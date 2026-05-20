"""
AI Model Training Calculator - Self-contained implementation.
Implements all 5 phases of the LLM training infrastructure calculator.
No external dependencies, no subprocess, no git clone required.
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
    "GRAD_ACCUM_VALUES": [4, 8, 16, 32, 64, 128],
    "LAYER_NORM": 2,
    "FFN_WEIGHT_MATRICES": 3,
    "NUM_EMBEDDINGS_TABLES": 2,
    "OPTIM_BYTES": 6,
    "EMPIRICAL_ACT_MULTIPLIER": 12,
    "SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER": 0.45,
    "FWD_BWD_ROUTING_BUFF_PASSES": 2,
    "NCCL_MEM_BUF": 4.0,
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
}


def get_param_bytes(precision):
    """Get bytes per parameter based on precision."""
    return 1 if precision == "FP8" else 2


# ============================================================================
# PHASE 1: Memory Analysis
# ============================================================================

def phase1_memory(variants, hardware, config):
    """
    Compute per-GPU memory breakdown for every (variant, hardware, ZeRO, micro) combo.
    Returns list of result dicts.
    """
    results = []
    precision = config.get("PRECISION", "BF16")
    param_bytes = get_param_bytes(precision)
    seq_len = config.get("SEQ_LEN", 4096)
    pp = config.get("PP", 1)
    tp = config.get("TP", 1)
    cp = config.get("CP", 1)
    ep = config.get("EP", 1)
    n_experts = config.get("N_EXPERTS", 0)
    vocab = config.get("VOCAB", 128256)

    adv = {**ADVANCED_DEFAULTS, **config.get("advanced", {})}
    optim_bytes = adv["OPTIM_BYTES"]
    act_mult = adv["EMPIRICAL_ACT_MULTIPLIER"]
    act_ckpt_mult = adv["SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER"]
    nccl_buf = adv["NCCL_MEM_BUF"]
    util_threshold = adv["GPU_MEM_UTILIZATION_THRESHOLD"]
    micros = adv["MICROS"]
    zero_strategies = adv["ZERO_STRATEGY"]

    for variant in variants:
        layers = variant["layers"]
        d = variant["d"]
        layers_per_gpu = layers // pp

        # Model parameter memory components
        attn_per_layer = variant["attn_per_layer"]
        dense_layer_params = variant["dense_layer_params"]
        moe_layer_params = variant.get("moe_layer_params", 0)
        dense_layers = variant.get("dense_layers", layers)
        moe_layers = layers - dense_layers

        # Embedding parameters
        embed_params = vocab * d * adv["NUM_EMBEDDINGS_TABLES"]

        # Layer norm + router (small, replicated across TP)
        ln_params = d * adv["LAYER_NORM"] * layers
        router_total = variant.get("router_each", 0) * moe_layers

        for hw in hardware:
            gpus = hw["gpus"]
            gpu_mem = hw["mem_gb"]
            usable_mem = gpu_mem * util_threshold
            gpus_per_node = gpus // hw.get("nodes", gpus // 8)

            dp = gpus // (tp * pp * cp)

            for zero_cfg in zero_strategies:
                zero_stage = zero_cfg["zero"]

                # Layers per GPU split between dense and MoE
                dense_layers_per_gpu = min(dense_layers, layers_per_gpu)
                moe_layers_per_gpu = layers_per_gpu - dense_layers_per_gpu

                # Attention + FFN sharded by TP
                model_params_dense = (dense_layer_params * dense_layers_per_gpu) / tp

                # MoE layer: attention/TP + experts/EP
                expert_each = variant.get("expert_each", 0)
                shared_each = variant.get("shared_each", 0)
                if moe_layers_per_gpu > 0 and n_experts > 0:
                    experts_per_gpu = n_experts // ep
                    moe_attn = attn_per_layer * moe_layers_per_gpu / tp
                    moe_expert = (expert_each * experts_per_gpu + shared_each) * moe_layers_per_gpu
                    model_params_moe = moe_attn + moe_expert
                else:
                    model_params_moe = 0

                # Embeddings sharded by TP
                embed_per_gpu = embed_params / tp

                # LN + router replicated (small)
                ln_router_per_gpu = (ln_params + router_total) / pp

                total_model_params = model_params_dense + model_params_moe + embed_per_gpu + ln_router_per_gpu

                # ZeRO-3 further shards model params by DP
                if zero_stage == 3:
                    total_model_params /= dp

                model_mem_gb = (total_model_params * param_bytes) / (1024**3)

                # Gradient memory
                if zero_stage >= 2:
                    grad_mem_gb = (total_model_params * param_bytes) / dp / (1024**3)
                else:
                    grad_mem_gb = model_mem_gb

                # Optimizer memory (always sharded by DP for all ZeRO stages)
                if zero_stage == 3:
                    base_params_for_optim = (model_params_dense + model_params_moe + embed_per_gpu + ln_router_per_gpu)
                    optim_mem_gb = (base_params_for_optim * optim_bytes) / dp / (1024**3)
                else:
                    optim_mem_gb = (total_model_params * optim_bytes) / dp / (1024**3)

                for micro in micros:
                    # Activation memory
                    act_mem_gb = (seq_len * micro * d * act_mult * layers_per_gpu * act_ckpt_mult) / (1024**3)

                    # Communication buffers
                    buf_mem_gb = nccl_buf
                    if n_experts > 0 and moe_layers_per_gpu > 0:
                        routing_buf = (seq_len * micro * d * adv["FWD_BWD_ROUTING_BUFF_PASSES"]) / (1024**3)
                        buf_mem_gb += routing_buf

                    total_mem = model_mem_gb + grad_mem_gb + optim_mem_gb + act_mem_gb + buf_mem_gb
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
    """
    Sweep micro batch x grad accumulation to find optimal batch configs.
    Returns list of result dicts.
    """
    results = []
    seq_len = config.get("SEQ_LEN", 4096)
    total_tokens = config.get("TOTAL_TOKENS", 15e12)
    tp = config.get("TP", 1)
    pp = config.get("PP", 1)
    cp = config.get("CP", 1)

    adv = {**ADVANCED_DEFAULTS, **config.get("advanced", {})}
    grad_accums = adv["GRAD_ACCUM_VALUES"]

    for variant in variants:
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

def phase3_training_time(variants, hardware, config):
    """
    Estimate wall-clock training time.
    Returns list of result dicts.
    """
    results = []
    total_tokens = config.get("TOTAL_TOKENS", 15e12)
    mfu = config.get("MFU", 0.40)

    adv = {**ADVANCED_DEFAULTS, **config.get("advanced", {})}
    low_mult = adv["TIME_ESTIMATE_LOW_MULTIPLIER"]
    high_mult = adv["TIME_ESTIMATE_HIGH_MULTIPLIER"]
    zero_strategies = adv["ZERO_STRATEGY"]

    for variant in variants:
        active_params = variant["active_params_B"]
        flops_per_token = 6 * active_params * 1e9
        total_flops = flops_per_token * total_tokens

        for hw in hardware:
            gpus = hw["gpus"]
            peak_tflops = hw["peak_tflops_bf16"]

            for zero_cfg in zero_strategies:
                zero_stage = zero_cfg["zero"]
                zero_eff = zero_cfg["eff"]

                effective_tflops = peak_tflops * mfu * zero_eff
                time_seconds = total_flops / (gpus * effective_tflops * 1e12)

                time_days = time_seconds / 86400
                time_low = time_days * low_mult
                time_high = time_days * high_mult
                time_months = time_days / 30.44

                results.append({
                    "variant": variant["name"],
                    "hardware": hw["name"],
                    "gpus": gpus,
                    "zero_stage": zero_stage,
                    "zero_efficiency": zero_eff,
                    "time_seconds": round(time_seconds, 1),
                    "time_days": round(time_days, 2),
                    "time_days_low": round(time_low, 2),
                    "time_days_high": round(time_high, 2),
                    "time_months": round(time_months, 2),
                    "time_months_low": round(time_months * low_mult, 2),
                    "time_months_high": round(time_months * high_mult, 2),
                    "effective_tflops_per_gpu": round(effective_tflops, 1),
                    "total_pflops": round(gpus * effective_tflops / 1000, 1),
                })

    return results


# ============================================================================
# PHASE 4: ZeRO Communication Overhead
# ============================================================================

def phase4_zero_comm(variants, hardware, config):
    """
    Compute ZeRO-1 (all-gather) and ZeRO-2 (reduce-scatter) communication overhead.
    Returns list of result dicts.
    """
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
    """
    Compute MoE all-to-all routing communication overhead.
    Returns list of result dicts (empty for dense models).
    """
    results = []
    n_experts = config.get("N_EXPERTS", 0)
    ep = config.get("EP", 1)
    seq_len = config.get("SEQ_LEN", 4096)

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
# MAIN ORCHESTRATOR
# ============================================================================

def run_calculator(variants, hardware, config, output_dir=None):
    """
    Run all 5 phases and return structured results.

    Args:
        variants: list of model variant dicts
        hardware: list of hardware config dicts
        config: merged project + advanced config dict
        output_dir: optional directory to write CSV exports

    Returns:
        dict with phase1-phase5 results
    """
    # Phase 1: Memory
    p1 = phase1_memory(variants, hardware, config)
    best_mem = find_best_memory_config(p1)

    # Phase 2: Batch
    p2 = phase2_batch(variants, hardware, config, best_mem)

    # Phase 3: Training Time
    p3 = phase3_training_time(variants, hardware, config)

    # Phase 4: ZeRO Communication
    p4 = phase4_zero_comm(variants, hardware, config)

    # Phase 5: MoE All-to-All
    p5 = phase5_alltoall(variants, hardware, config)

    results = {
        "phase1_memory": p1,
        "phase1_best": {k[0] + "|" + k[1]: v for k, v in best_mem.items()},
        "phase2_batch": p2,
        "phase3_training_time": p3,
        "phase4_zero_comm": p4,
        "phase5_alltoall": p5,
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
