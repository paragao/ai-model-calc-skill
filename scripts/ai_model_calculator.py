"""
AI Model Training Calculator - Self-contained implementation.
Implements all 6 phases of the LLM training infrastructure calculator.
No external dependencies, no subprocess, no git clone required.

Phases:
  1. Memory Analysis (MoE-aware with VP activation overhead)
  2. Batch Configuration
  3. Training Time
  4. ZeRO Communication Overhead
  5. MoE All-to-All Communication
  6. PP SendRecv Communication (NEW)
"""
import math
import csv
import json
import os

# ============================================================================
# DEFAULT ADVANCED CONFIGURATION
# ============================================================================

ADVANCED_DEFAULTS = {
    "TOKENS_PER_BATCH": 4e6,
    "ZERO_STRATEGY": [
        {"zero": 1, "eff": 1.0},
        {"zero": 2, "eff": 0.9},
        {"zero": 3, "eff": 0.8},
    ],
    "MICROS": [1, 2, 4, 8, 16],
    "GRAD_ACCUM_VALUES": [1, 2, 4, 8, 16, 32],
    "LAYER_NORM": 2,
    "FFN_WEIGHT_MATRICES": 3,
    "NUM_EMBEDDINGS_TABLES": 2,
    "OPTIM_BYTES": 6,
    "EMPIRICAL_ACT_MULTIPLIER": 12,
    "SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER": 0.45,
    "FWD_BWD_ROUTING_BUFF_PASSES": 2,
    "NCCL_MEM_BUF": 6.0,
    "NCCL_EP_SCALING_FACTOR": 1.0,
    "FRAGMENTATION_FACTOR": 1.24,
    "GPU_MEM_UTILIZATION_THRESHOLD": 0.92,
    "LOWER_BOUND_OPTIM_RANGE": 300_000,
    "UPPER_BOUND_OPTIM_RANGE": 800_000,
    "LOWER_BOUND_LOW_RANGE": 200_000,
    "UPPER_BOUND_LOW_RANGE": 300_000,
    "LOWER_BOUND_HIGH_RANGE": 800_000,
    "UPPER_BOUND_HIGH_RANGE": 1_200_000,
    "MIN_STEPS": 200_000,
    "MAX_STEPS": 1_200_000,
    "MAX_TOKENS_PER_BATCH": 600e6,
    "TIME_ESTIMATE_LOW_MULTIPLIER": 0.75,
    "TIME_ESTIMATE_HIGH_MULTIPLIER": 1.25,
    "EFA_PP_EFFICIENCY": 0.016,
    # Fraction of MoE all-to-all comm NOT overlapped with compute (Phase 3
    # exposed-comm term). 0.0 = fully overlapped, 1.0 = fully exposed.
    "A2A_EXPOSED_FRACTION": 0.5,
    # Attention FLOPs factor for the O(seq_len^2) attention term in training
    # time. 6ND omits the QK^T and softmax*V matmuls (O(seq_len^2 * d)/layer);
    # 12 = forward ~4 (2 matmuls * 2 FLOPs/MAC) x3 for fwd+bwd. 0 disables.
    "ATTENTION_FLOPS_FACTOR": 12,
}


def get_param_bytes(precision):
    """Get bytes per parameter based on precision."""
    return 1 if precision == "FP8" else 2


# ============================================================================
# PHASE 1: Memory Analysis (with MoE fixes + VP activation overhead)
# ============================================================================

def calculate_model_memory_gb(variant, layers_per_gpu, experts_per_gpu, param_bytes,
                              vocab, tp=1):
    """Calculate model memory in GB using the improved MoE-aware formula."""
    d = variant["d"]
    layers = variant["layers"]
    moe_layers = layers - variant.get("dense_layers", layers)

    # Distribute MoE layers proportionally
    moe_layers_per_gpu = int(layers_per_gpu * moe_layers / layers) if layers > 0 and moe_layers > 0 else 0

    attn_mem = layers_per_gpu * variant["attn_per_layer"] * param_bytes
    routed_mem = experts_per_gpu * moe_layers_per_gpu * variant["expert_each"] * param_bytes
    shared_mem = moe_layers_per_gpu * variant.get("shared_each", 0) * param_bytes
    router_mem = moe_layers_per_gpu * variant.get("router_each", 0) * param_bytes
    ln_mem = layers_per_gpu * 2 * d * param_bytes  # LAYER_NORM = 2
    dense_layers_per_gpu = layers_per_gpu - moe_layers_per_gpu
    dense_ffn_mem = dense_layers_per_gpu * 3 * d * variant["dense_ffn"] * param_bytes
    embed_mem = 2 * vocab * d * param_bytes  # NUM_EMBEDDINGS_TABLES = 2

    # TP shards attention, FFN, expert, and embedding. LN + router replicated.
    model_mem_bytes = (
        (attn_mem + routed_mem + shared_mem + dense_ffn_mem + embed_mem) / tp
        + ln_mem + router_mem
    )
    return model_mem_bytes / 1e9


def phase1_memory(variants, hardware, config):
    """Compute per-GPU memory breakdown with MoE fixes, VP overhead, and fragmentation."""
    results = []
    precision = config.get("PRECISION", "BF16")
    param_bytes = get_param_bytes(precision)
    default_seq_len = config.get("SEQ_LEN", 4096)
    pp = config.get("PP", 1)
    tp = config.get("TP", 1)
    cp = config.get("CP", 1)
    ep = config.get("EP", 1)
    vp = config.get("VP", 1)
    n_experts = config.get("N_EXPERTS", 0)
    topk = config.get("TOPK", 0)
    vocab = config.get("VOCAB", 128256)

    adv = {**ADVANCED_DEFAULTS, **config.get("advanced", {})}
    optim_bytes = adv["OPTIM_BYTES"]
    act_mult = adv["EMPIRICAL_ACT_MULTIPLIER"]
    act_ckpt_mult = adv["SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER"]
    nccl_buf = adv["NCCL_MEM_BUF"]
    nccl_ep_scaling = adv["NCCL_EP_SCALING_FACTOR"]
    frag_factor = adv["FRAGMENTATION_FACTOR"]
    util_threshold = adv["GPU_MEM_UTILIZATION_THRESHOLD"]
    micros = adv["MICROS"]
    zero_strategies = adv["ZERO_STRATEGY"]
    fwd_bwd_passes = adv["FWD_BWD_ROUTING_BUFF_PASSES"]

    for variant in variants:
        layers = variant["layers"]
        d = variant["d"]
        seq_len = variant.get("seq_len", default_seq_len)
        layers_per_gpu = layers // pp
        is_moe = variant.get("expert_ffn", 0) > 0
        moe_layers = layers - variant.get("dense_layers", layers)
        moe_layers_per_gpu = int(layers_per_gpu * moe_layers / layers) if layers > 0 and moe_layers > 0 else 0
        dense_layers_on_gpu = layers_per_gpu - moe_layers_per_gpu
        experts_per_gpu = n_experts // ep if ep > 0 and n_experts > 0 else 0

        for hw in hardware:
            gpus = hw["gpus"]
            gpu_mem = hw["mem_gb"]
            usable_mem = gpu_mem * util_threshold
            nodes = hw.get("nodes", gpus // 8)

            dp = gpus // (tp * pp * cp)

            # Model memory
            model_mem_gb = calculate_model_memory_gb(
                variant, layers_per_gpu, experts_per_gpu, param_bytes, vocab, tp
            )

            for zero_cfg in zero_strategies:
                zero_stage = zero_cfg["zero"]

                # --- EP/DP optimizer sharding fix ---
                if is_moe and ep > 1 and dp > ep:
                    eff_dp_moe = dp // ep
                else:
                    eff_dp_moe = dp

                # MoE fraction for optimizer splitting
                if is_moe and model_mem_gb > 0:
                    routed_mem_bytes = experts_per_gpu * moe_layers_per_gpu * variant["expert_each"] * param_bytes
                    total_model_bytes = model_mem_gb * 1e9
                    moe_frac = min(1.0, routed_mem_bytes / total_model_bytes) if total_model_bytes > 0 else 0
                else:
                    moe_frac = 0.0
                dense_frac = 1.0 - moe_frac

                # Gradient memory
                if zero_stage >= 2:
                    grad_mem_gb = model_mem_gb / dp
                else:
                    grad_mem_gb = model_mem_gb

                # Optimizer memory (dense sharded by full DP, MoE by eff_dp_moe)
                optim_dense = model_mem_gb * dense_frac * optim_bytes / dp
                optim_moe = model_mem_gb * moe_frac * optim_bytes / eff_dp_moe
                optim_mem_gb = optim_dense + optim_moe

                for micro in micros:
                    # --- MoE-aware activation memory ---
                    if is_moe and moe_layers_per_gpu > 0:
                        expert_ffn = variant.get("expert_ffn", 0)
                        moe_act_per_layer = seq_len * micro * (d * act_mult + topk * expert_ffn * 2 * param_bytes)
                        dense_act_per_layer = seq_len * micro * d * act_mult
                        act_total_full = (moe_layers_per_gpu * moe_act_per_layer +
                                          dense_layers_on_gpu * dense_act_per_layer)
                    else:
                        act_total_full = layers_per_gpu * seq_len * micro * d * act_mult

                    act_mem_gb = act_total_full * act_ckpt_mult / 1e9

                    # --- VP activation overhead ---
                    if vp > 1 and pp > 1:
                        layers_per_virtual_chunk = layers_per_gpu // vp
                        inflight_micros = (pp - 1) * vp
                        if is_moe and moe_layers_per_gpu > 0:
                            expert_ffn = variant.get("expert_ffn", 0)
                            avg_act_per_layer = seq_len * micro * (d * act_mult + topk * expert_ffn * 2 * param_bytes)
                        else:
                            avg_act_per_layer = seq_len * micro * d * act_mult
                        vp_overhead = inflight_micros * layers_per_virtual_chunk * avg_act_per_layer / 1e9
                        act_mem_gb += vp_overhead

                    # --- Buffer memory ---
                    # All-to-all routing buffers
                    a2a_buf = seq_len * micro * topk * d * fwd_bwd_passes * param_bytes / 1e9

                    # MoE dispatch buffers
                    moe_dispatch_buf = 0.0
                    if is_moe and experts_per_gpu > 0:
                        moe_dispatch_buf = (
                            moe_layers_per_gpu * 2 * micro * seq_len * d * param_bytes
                            * (topk / experts_per_gpu) / 1e9
                        )

                    # NCCL workspace: base + EP scaling
                    nccl_workspace = nccl_buf + nccl_ep_scaling * max(0, ep - 1)

                    # PP send/recv buffers
                    pp_buf = 0.0
                    if pp > 1:
                        pp_buf = 2 * micro * seq_len * d * param_bytes / 1e9

                    buf_mem_gb = a2a_buf + moe_dispatch_buf + nccl_workspace + pp_buf

                    # --- Total with fragmentation ---
                    raw_total = model_mem_gb + grad_mem_gb + optim_mem_gb + act_mem_gb + buf_mem_gb
                    total_mem = raw_total * frag_factor

                    headroom = usable_mem - total_mem
                    fits = headroom > 0
                    utilization = (total_mem / gpu_mem) * 100

                    results.append({
                        "variant": variant["name"],
                        "hardware": hw["name"],
                        "gpus": gpus,
                        "gpu_mem_gb": gpu_mem,
                        "zero_stage": zero_stage,
                        "micro_batch": micro,
                        "model_mem_gb": round(model_mem_gb, 2),
                        "grad_mem_gb": round(grad_mem_gb, 2),
                        "optim_mem_gb": round(optim_mem_gb, 2),
                        "act_mem_gb": round(act_mem_gb, 2),
                        "buf_mem_gb": round(buf_mem_gb, 2),
                        "total_mem_gb": round(total_mem, 2),
                        "headroom_gb": round(headroom, 2),
                        "utilization_pct": round(utilization, 1),
                        "fits": fits,
                        "dp": dp,
                    })

    return results


def find_best_memory_config(phase1_results):
    """Find minimum ZeRO stage and max micro batch that fits for each variant x hardware."""
    best = {}
    for r in phase1_results:
        if not r["fits"]:
            continue
        key = (r["variant"], r["hardware"])
        if key not in best:
            best[key] = r
        else:
            existing = best[key]
            if (r["zero_stage"] < existing["zero_stage"]) or \
               (r["zero_stage"] == existing["zero_stage"] and r["micro_batch"] > existing["micro_batch"]):
                best[key] = r
    return best


# ============================================================================
# PHASE 2: Batch Configuration
# ============================================================================

def phase2_batch(variants, hardware, config, best_memory):
    """Sweep micro batch x grad accumulation to find optimal batch configs."""
    results = []
    default_seq_len = config.get("SEQ_LEN", 4096)
    total_tokens = config.get("TOTAL_TOKENS", 15e12)

    adv = {**ADVANCED_DEFAULTS, **config.get("advanced", {})}
    grad_accums = adv["GRAD_ACCUM_VALUES"]

    for variant in variants:
        seq_len = variant.get("seq_len", default_seq_len)
        for hw in hardware:
            key = (variant["name"], hw["name"])
            if key not in best_memory:
                continue

            best = best_memory[key]
            max_micro = best["micro_batch"]
            dp = best["dp"]

            for micro in range(1, max_micro + 1):
                for accum in grad_accums:
                    tokens_per_batch = micro * seq_len * accum * dp

                    if tokens_per_batch > adv["MAX_TOKENS_PER_BATCH"]:
                        continue

                    training_steps = int(total_tokens / tokens_per_batch)

                    if training_steps < adv["MIN_STEPS"] or training_steps > adv["MAX_STEPS"]:
                        priority = 3
                    elif adv["LOWER_BOUND_OPTIM_RANGE"] <= training_steps <= adv["UPPER_BOUND_OPTIM_RANGE"]:
                        priority = 0
                    elif adv["LOWER_BOUND_LOW_RANGE"] <= training_steps < adv["LOWER_BOUND_OPTIM_RANGE"]:
                        priority = 1
                    elif adv["UPPER_BOUND_OPTIM_RANGE"] < training_steps <= adv["UPPER_BOUND_HIGH_RANGE"]:
                        priority = 1
                    else:
                        priority = 2

                    assessment = ["Optimal", "Good", "Acceptable", "Poor"][priority]

                    results.append({
                        "variant": variant["name"],
                        "hardware": hw["name"],
                        "micro_batch": micro,
                        "grad_accum": accum,
                        "tokens_per_batch": tokens_per_batch,
                        "training_steps": training_steps,
                        "priority": priority,
                        "assessment": assessment,
                        "dp": dp,
                    })

    results.sort(key=lambda x: (x["variant"], x["hardware"], x["priority"], -x["tokens_per_batch"]))
    return results


# ============================================================================
# PHASE 3: Training Time
# ============================================================================

def _phase3_overhead(variant, hw, config, adv, micro, gbs, dp):
    """Estimate PP bubble fraction and exposed comm ms/step for one config.

    Mirrors the Phase 6 (PP send/recv) and Phase 5 (MoE all-to-all) models so
    the overhead varies with the swept micro/accum (via gbs -> num_microbatches).

    Returns:
        (bubble_fraction, exposed_comm_ms_per_step)
    """
    pp = config.get("PP", 1)
    vp = config.get("VP", 1)
    ep = config.get("EP", 1)
    n_experts = config.get("N_EXPERTS", 0)
    seq_len = variant.get("seq_len", config.get("SEQ_LEN", 4096))
    d = variant["d"]
    dtype_bytes = get_param_bytes(config.get("PRECISION", "BF16"))

    num_microbatches = gbs // (micro * dp) if (micro * dp) > 0 else 1

    bubble_fraction = 0.0
    exposed_comm_ms = 0.0

    # --- Pipeline bubble + exposed PP send time (PP > 1) ---
    if pp > 1:
        effective_microbatches = num_microbatches * vp if vp > 1 else num_microbatches
        bubble_pct = ((pp - 1) / effective_microbatches * 100
                      if effective_microbatches > 0 else 100)
        bubble_fraction = min(0.99, bubble_pct / 100.0)

        gpus_per_node = hw["gpus"] // hw.get("nodes", max(1, hw["gpus"] // 8))
        is_inter_node = pp > gpus_per_node
        if is_inter_node:
            effective_bw = hw["inter_node_bw_gb"] * adv["EFA_PP_EFFICIENCY"]
        else:
            effective_bw = hw["intra_node_bw_gbps"]
        activation_size_bytes = micro * seq_len * d * dtype_bytes
        time_per_send_us = (
            (activation_size_bytes / 1e9) / effective_bw * 1e6 if effective_bw > 0 else 0
        )
        exposed_sends = 2 * (pp - 1)  # warmup + cooldown (serialized)
        exposed_comm_ms += exposed_sends * time_per_send_us / 1000

    # --- Exposed MoE all-to-all (MoE only) ---
    if n_experts > 0 and ep > 1 and variant.get("expert_ffn", 0) > 0:
        moe_layers = variant["layers"] - variant.get("dense_layers", 0)
        fwd_bwd_passes = adv["FWD_BWD_ROUTING_BUFF_PASSES"]
        intra_bw = hw["intra_node_bw_gbps"]
        tokens_per_micro = seq_len * micro
        volume_per_layer_gb = (tokens_per_micro * d * 2 * fwd_bwd_passes) / (1024 ** 3)
        # a2a ms per micro-batch across MoE layers, times micro-batches per step
        a2a_per_microbatch_ms = (
            (volume_per_layer_gb * moe_layers) / intra_bw * 1000 if intra_bw > 0 else 0
        )
        a2a_per_step_ms = a2a_per_microbatch_ms * num_microbatches
        exposed_comm_ms += adv["A2A_EXPOSED_FRACTION"] * a2a_per_step_ms

    return bubble_fraction, exposed_comm_ms


def phase3_training_time(variants, hardware, config, best_memory):
    """Estimate wall-clock training time over the micro x grad-accum grid.

    Gradient accumulation is an explicit user-controlled input (it affects
    convergence/perplexity), so GRAD_ACCUM_VALUES is swept rather than derived.
    Uses the single best-fit ZeRO stage per (variant, hardware) from Phase 1,
    and folds PP pipeline-bubble + exposed communication into the wall-clock
    time on top of the ideal FLOPs compute time.
    """
    results = []
    total_tokens = config.get("TOTAL_TOKENS", 15e12)
    mfu = config.get("MFU", 0.40)
    seq_len = config.get("SEQ_LEN", 4096)

    adv = {**ADVANCED_DEFAULTS, **config.get("advanced", {})}
    low_mult = adv["TIME_ESTIMATE_LOW_MULTIPLIER"]
    high_mult = adv["TIME_ESTIMATE_HIGH_MULTIPLIER"]
    grad_accums = adv["GRAD_ACCUM_VALUES"]
    max_tokens_per_batch = adv["MAX_TOKENS_PER_BATCH"]
    attn_factor = adv["ATTENTION_FLOPS_FACTOR"]
    zero_eff_map = {z["zero"]: z["eff"] for z in adv["ZERO_STRATEGY"]}
    _SECONDS_PER_MONTH = 86400 * 30.44

    for variant in variants:
        active_params = variant["active_params_B"]
        # Per-variant sequence length (falls back to the global default).
        v_seq_len = variant.get("seq_len", seq_len)
        # Linear-in-tokens compute (6ND) + O(seq_len^2) attention term. The
        # attention term is linear in seq_len at fixed total_tokens, so a 32K
        # context estimates more compute than a 4K one for the same tokens.
        linear_flops = 6 * active_params * 1e9 * total_tokens
        attention_flops = (
            attn_factor * variant["layers"] * v_seq_len * total_tokens * variant["d"]
            if attn_factor else 0.0
        )
        total_flops = linear_flops + attention_flops

        for hw in hardware:
            key = (variant["name"], hw["name"])
            if key not in best_memory:
                continue  # OOM on all configs for this variant/hardware

            best = best_memory[key]
            zero_stage = best["zero_stage"]
            zero_eff = zero_eff_map.get(zero_stage, 1.0)
            max_micro = best["micro_batch"]
            dp = best["dp"]

            gpus = hw["gpus"]
            peak_tflops = hw["peak_tflops_bf16"]
            effective_tflops = peak_tflops * mfu * zero_eff
            compute_seconds = total_flops / (gpus * effective_tflops * 1e12)

            for micro in range(1, max_micro + 1):
                for accum in grad_accums:
                    gbs = dp * micro * accum
                    tokens_per_batch = gbs * v_seq_len
                    if tokens_per_batch > max_tokens_per_batch:
                        continue
                    steps = int(total_tokens / tokens_per_batch) if tokens_per_batch > 0 else 0

                    # Overhead: PP bubble (fraction of compute) + exposed comm (ms/step)
                    bubble_fraction, exposed_comm_ms = _phase3_overhead(
                        variant, hw, config, adv, micro, gbs, dp
                    )
                    bubble_seconds = (
                        compute_seconds * bubble_fraction / (1.0 - bubble_fraction)
                        if bubble_fraction > 0 else 0.0
                    )
                    comm_seconds = (exposed_comm_ms / 1000.0) * steps
                    total_seconds = compute_seconds + bubble_seconds + comm_seconds

                    time_days = total_seconds / 86400
                    time_months = total_seconds / _SECONDS_PER_MONTH
                    bubble_pct = (bubble_seconds / total_seconds * 100) if total_seconds > 0 else 0.0
                    comm_pct = (comm_seconds / total_seconds * 100) if total_seconds > 0 else 0.0

                    results.append({
                        "variant": variant["name"],
                        "hardware": hw["name"],
                        "gpus": gpus,
                        "zero_stage": zero_stage,
                        "zero_efficiency": zero_eff,
                        "micro_batch": micro,
                        "grad_accum": accum,
                        "gbs": gbs,
                        "tokens_per_batch": tokens_per_batch,
                        "training_steps": steps,
                        "time_days": round(time_days, 2),
                        "time_days_low": round(time_days * low_mult, 2),
                        "time_days_high": round(time_days * high_mult, 2),
                        "time_months": round(time_months, 2),
                        "time_months_low": round(time_months * low_mult, 2),
                        "time_months_high": round(time_months * high_mult, 2),
                        "bubble_pct": round(bubble_pct, 1),
                        "comm_overhead_pct": round(comm_pct, 1),
                        "effective_tflops_per_gpu": round(effective_tflops, 1),
                        "total_pflops": round(gpus * effective_tflops / 1000, 1),
                    })

    return results


# ============================================================================
# PHASE 4: ZeRO Communication Overhead
# ============================================================================

def phase4_zero_comm(variants, hardware, config):
    """Compute ZeRO-1 (all-gather) and ZeRO-2 (reduce-scatter) communication overhead."""
    results = []
    precision = config.get("PRECISION", "BF16")
    param_bytes = get_param_bytes(precision)
    tp = config.get("TP", 1)
    pp = config.get("PP", 1)
    cp = config.get("CP", 1)
    n_experts = config.get("N_EXPERTS", 0)
    ep = config.get("EP", 1)

    for variant in variants:
        layers = variant["layers"]
        layers_per_gpu = layers // pp
        dense_layers = variant.get("dense_layers", layers)
        moe_layers = layers - dense_layers

        dense_layers_per_gpu = min(dense_layers, layers_per_gpu)
        moe_layers_per_gpu = layers_per_gpu - dense_layers_per_gpu

        model_params_per_gpu = variant["dense_layer_params"] * dense_layers_per_gpu / tp
        if moe_layers_per_gpu > 0 and n_experts > 0:
            experts_per_gpu = n_experts // ep if ep > 0 else 0
            expert_each = variant.get("expert_each", 0)
            shared_each = variant.get("shared_each", 0)
            model_params_per_gpu += (variant["attn_per_layer"] / tp + expert_each * experts_per_gpu + shared_each) * moe_layers_per_gpu

        model_size_bytes = model_params_per_gpu * param_bytes
        model_size_gb = model_size_bytes / (1024**3)

        for hw in hardware:
            gpus = hw["gpus"]
            inter_bw = hw["inter_node_bw_gb"]
            dp = gpus // (tp * pp * cp)

            zero2_volume_gb = model_size_gb * (dp - 1) / dp
            zero2_time_ms = (zero2_volume_gb / inter_bw) * 1000

            zero1_volume_gb = model_size_gb * (dp - 1) / dp
            zero1_time_ms = (zero1_volume_gb / inter_bw) * 1000

            results.append({
                "variant": variant["name"],
                "hardware": hw["name"],
                "gpus": gpus,
                "dp": dp,
                "model_size_gb": round(model_size_gb, 2),
                "zero2_reduce_scatter_volume_gb": round(zero2_volume_gb, 3),
                "zero2_reduce_scatter_time_ms": round(zero2_time_ms, 2),
                "zero1_all_gather_volume_gb": round(zero1_volume_gb, 3),
                "zero1_all_gather_time_ms": round(zero1_time_ms, 2),
                "inter_node_bw_gb": inter_bw,
            })

    return results


# ============================================================================
# PHASE 5: MoE All-to-All Communication
# ============================================================================

def phase5_alltoall(variants, hardware, config):
    """Compute MoE all-to-all routing communication overhead."""
    results = []
    n_experts = config.get("N_EXPERTS", 0)
    ep = config.get("EP", 1)
    default_seq_len = config.get("SEQ_LEN", 4096)

    if n_experts == 0 or ep <= 1:
        return results

    adv = {**ADVANCED_DEFAULTS, **config.get("advanced", {})}
    micros = adv["MICROS"]
    fwd_bwd_passes = adv["FWD_BWD_ROUTING_BUFF_PASSES"]

    for variant in variants:
        if variant.get("expert_ffn", 0) == 0:
            continue

        d = variant["d"]
        layers = variant["layers"]
        seq_len = variant.get("seq_len", default_seq_len)
        dense_layers = variant.get("dense_layers", 0)
        moe_layers = layers - dense_layers

        for hw in hardware:
            intra_bw = hw["intra_node_bw_gbps"]

            for micro in micros:
                tokens_per_micro = seq_len * micro
                volume_per_layer_bytes = tokens_per_micro * d * 2 * fwd_bwd_passes
                volume_per_layer_gb = volume_per_layer_bytes / (1024**3)

                total_volume_gb = volume_per_layer_gb * moe_layers
                total_time_ms = (total_volume_gb / intra_bw) * 1000

                results.append({
                    "variant": variant["name"],
                    "hardware": hw["name"],
                    "micro_batch": micro,
                    "moe_layers": moe_layers,
                    "volume_per_layer_gb": round(volume_per_layer_gb, 4),
                    "total_volume_gb": round(total_volume_gb, 3),
                    "total_time_ms": round(total_time_ms, 2),
                    "intra_node_bw_gbps": intra_bw,
                })

    return results


# ============================================================================
# PHASE 6: PP SendRecv Communication (NEW)
# ============================================================================

def phase6_pp_comm(variants, hardware, config):
    """
    Compute PP point-to-point communication overhead.
    Models intra-node (NVLink) vs inter-node (EFA) latency.
    """
    results = []
    pp = config.get("PP", 1)
    vp = config.get("VP", 1)
    tp = config.get("TP", 1)
    cp = config.get("CP", 1)
    ep = config.get("EP", 1)
    default_seq_len = config.get("SEQ_LEN", 4096)
    precision = config.get("PRECISION", "BF16")
    dtype_bytes = get_param_bytes(precision)
    total_tokens = config.get("TOTAL_TOKENS", 15e12)

    if pp <= 1:
        return results

    adv = {**ADVANCED_DEFAULTS, **config.get("advanced", {})}
    efa_pp_eff = adv["EFA_PP_EFFICIENCY"]
    micros = adv["MICROS"]

    for variant in variants:
        layers = variant["layers"]
        d = variant["d"]
        seq_len = variant.get("seq_len", default_seq_len)

        for hw in hardware:
            gpus = hw["gpus"]
            inter_bw = hw["inter_node_bw_gb"]
            intra_bw = hw["intra_node_bw_gbps"]
            gpus_per_node = gpus // hw.get("nodes", gpus // 8)
            dp = gpus // (tp * pp * cp)

            # Determine if PP crosses node boundary
            is_inter_node = pp > gpus_per_node
            comm_type = "inter-node (EFA)" if is_inter_node else "intra-node (NVLink)"

            for micro in micros:
                # Activation size per P2P send
                activation_size_bytes = micro * seq_len * d * dtype_bytes
                activation_size_mb = activation_size_bytes / 1e6

                # Sends per micro-batch
                if vp > 1:
                    sends_per_microbatch = 2 * (pp * vp - 1)
                else:
                    sends_per_microbatch = 2 * (pp - 1)

                # GBS approximation for tokens_per_batch / seq_len
                tokens_per_batch = adv["TOKENS_PER_BATCH"]
                gbs = int(tokens_per_batch / seq_len)
                num_microbatches = gbs // (micro * dp) if (micro * dp) > 0 else 1

                # Total sends and traffic
                total_sends = sends_per_microbatch * num_microbatches
                total_traffic_gb = total_sends * activation_size_bytes / 1e9

                # Pipeline bubble
                effective_microbatches = num_microbatches * vp if vp > 1 else num_microbatches
                bubble_pct = ((pp - 1) / effective_microbatches * 100
                              if effective_microbatches > 0 else 100)

                # Time per send
                if is_inter_node:
                    effective_bw = inter_bw * efa_pp_eff
                else:
                    effective_bw = intra_bw
                time_per_send_us = (activation_size_bytes / 1e9) / effective_bw * 1e6 if effective_bw > 0 else 0

                # Exposed PP time (warmup + cooldown sends)
                exposed_sends = 2 * (pp - 1)
                estimated_pp_time_ms = exposed_sends * time_per_send_us / 1000

                results.append({
                    "variant": variant["name"],
                    "hardware": hw["name"],
                    "micro_batch": micro,
                    "activation_size_mb": round(activation_size_mb, 2),
                    "sends_per_microbatch": sends_per_microbatch,
                    "num_microbatches": num_microbatches,
                    "total_sends_per_step": total_sends,
                    "total_traffic_gb": round(total_traffic_gb, 3),
                    "pipeline_bubble_pct": round(bubble_pct, 1),
                    "time_per_send_us": round(time_per_send_us, 1),
                    "estimated_pp_time_ms": round(estimated_pp_time_ms, 2),
                    "comm_type": comm_type,
                })

    return results


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def run_calculator(variants, hardware, config, output_dir=None):
    """
    Run all 6 phases and return structured results.

    Args:
        variants: list of model variant dicts
        hardware: list of hardware config dicts
        config: merged project + advanced config dict
        output_dir: optional directory to write CSV exports

    Returns:
        dict with phase1-phase6 results
    """
    # Phase 1: Memory
    p1 = phase1_memory(variants, hardware, config)
    best_mem = find_best_memory_config(p1)

    # Phase 2: Batch
    p2 = phase2_batch(variants, hardware, config, best_mem)

    # Phase 3: Training Time
    p3 = phase3_training_time(variants, hardware, config, best_mem)

    # Phase 4: ZeRO Communication
    p4 = phase4_zero_comm(variants, hardware, config)

    # Phase 5: MoE All-to-All
    p5 = phase5_alltoall(variants, hardware, config)

    # Phase 6: PP SendRecv Communication
    p6 = phase6_pp_comm(variants, hardware, config)

    results = {
        "phase1_memory": p1,
        "phase1_best": {k[0] + "|" + k[1]: v for k, v in best_mem.items()},
        "phase2_batch": p2,
        "phase3_training_time": p3,
        "phase4_zero_comm": p4,
        "phase5_alltoall": p5,
        "phase6_pp_comm": p6,
    }

    # Export CSVs if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        if p1:
            _write_csv(os.path.join(output_dir, "phase1_memory_results.csv"), p1)
        if p2:
            _write_csv(os.path.join(output_dir, "phase2_batch_results.csv"), p2)
        if p3:
            _write_csv(os.path.join(output_dir, "phase3_training_results.csv"), p3)
        if p4:
            _write_csv(os.path.join(output_dir, "phase4_zero_comm_results.csv"), p4)
        if p5:
            _write_csv(os.path.join(output_dir, "phase5_alltoall_results.csv"), p5)
        if p6:
            _write_csv(os.path.join(output_dir, "phase6_pp_comm_results.csv"), p6)

        # JSON export for Phase 1
        with open(os.path.join(output_dir, "phase1_memory_results.json"), "w") as f:
            json.dump(p1, f, indent=2)

    return results


def _write_csv(path, data):
    """Write list of dicts to CSV."""
    if not data:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
