---
name: ai-model-calc
display_name: AI Model Training Calculator
description: "Analyze GPU memory, batch configuration, training time, and communication overhead for LLM training on AWS hardware. Activate when the user asks about training infrastructure requirements, GPU memory for models, training time estimates, instance sizing for LLM training, or comparing hardware platforms for model training."
icon: "🖥️"
trigger: calculate training infrastructure
inputs:
  - name: mode
    description: "Analysis mode: A (how many instances for a time target), B (how long with fixed instances), or C (minimum instances to fit a model)"
    type: string
    required: true
  - name: model
    description: "Model(s) to analyze — e.g. 'Llama-3.1-70B', 'DeepSeek-V3', 'Qwen3-235B-A22B', or 'custom'"
    type: string
    required: true
  - name: hardware
    description: "AWS hardware platform — e.g. 'p5', 'p5en', 'p6-b200', 'p6-b300', 'p6e-gb200', 'g5', 'g6', 'g6e'"
    type: string
    required: true
  - name: dataset_tokens
    description: "Dataset size in tokens (e.g. '15T', '30B', '1e12'). Required for modes A and B only."
    type: string
    required: false
  - name: target_time
    description: "Target training time (e.g. '2 months', '30 days'). Required for mode A only."
    type: string
    required: false
  - name: node_count
    description: "Number of instances (e.g. 64, 128, 256, 512). Required for mode B only."
    type: number
    required: false
tools: [run_python, run_python_with_write]
scripts: [ai_model_calculator.py]
---

## Overview

This skill runs a 5-phase LLM training infrastructure calculator that computes GPU memory requirements, optimal batch configurations, wall-clock training time estimates, and communication overhead (ZeRO + MoE all-to-all) for large language models on AWS GPU instances. It supports dense and MoE architectures across p5 (H100), p5en (H200), p6-b200 (B200), p6-b300 (B300 Ultra), p6e-gb200 (GB200 NVL), and G-family instances.

The calculator is **fully self-contained and portable** — no git clone, no subprocess, no external dependencies, no hardcoded paths. The bundled `ai_model_calculator.py` script is auto-imported when the skill loads. Works on any platform (macOS, Windows, Linux) for any user. Simply `from ai_model_calculator import run_calculator`.

## Supported Models

### Dense Models
Llama-3.1-8B, Llama-3.1-70B, Llama-3.1-405B, Llama-3.2-3B, Llama-3.3-70B, Qwen2.5-7B, Qwen2.5-32B, Qwen2.5-72B, Qwen3-8B, Qwen3-32B, Qwen3.5-27B

### MoE Models
Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3.5-35B-A3B, Qwen3.5-122B-A10B, Qwen3.5-397B-A17B, Llama-4-Scout-17B-16E, Llama-4-Maverick-17B-128E, DeepSeek-V3, DeepSeek-V3.1, DeepSeek-V3.2, DeepSeek-V4

### Custom
Users can provide architecture details manually (layers, hidden dim, FFN dim, heads, etc.)

## Supported Hardware

| Platform | GPU | Memory/GPU | Peak BF16 TFLOPS | GPUs/node |
|----------|-----|-----------|-----------------|-----------|
| p5.48xlarge | H100 SXM | 80 GB | 989 | 8 |
| p5en.48xlarge | H200 SXM | 141 GB | 989 | 8 |
| p6-b200.48xlarge | B200 | 180 GB | 2250 | 8 |
| p6-b300.48xlarge | B300 Ultra | 268 GB | 3375 | 8 |
| p6e-gb200.36xlarge | GB200 NVL | 185 GB | 2500 | 4 |
| g5.48xlarge | A10G | 24 GB | 35 | 8 |
| g6.48xlarge | L4 | 24 GB | 121 | 8 |
| g6e.48xlarge | L40S | 48 GB | 366 | 8 |

## Workflow

### Step 0: Determine Mode
- **Mode**: `agentic`
- **Input**: `{{mode}}` from user (or infer from their question)
- **Output**: Confirmed mode (A, B, or C) with required parameters identified
- **Validate**: Mode is one of A/B/C and all required inputs for that mode are collected
- **On failure**: Ask the user to clarify their question

**Mode A** — "How many instances do I need to finish in time X?" Requires: model, hardware type, dataset_tokens, target_time.
**Mode B** — "How long will training take?" Requires: model, hardware type, dataset_tokens, node_count.
**Mode C** — "What's the minimum to fit this model?" Requires: model, hardware type. No dataset size needed.

### Step 1: Select Model Architecture
- **Mode**: `agentic`
- **Input**: `{{model}}` from user
- **Output**: Complete model variant configuration dict(s)
- **Validate**: All required fields populated (layers, d, q_heads, kv_heads, dense_ffn, expert_ffn, total_params_B, active_params_B, dense_layers, attn_per_layer, expert_each, shared_each, router_each, moe_layer_params, dense_layer_params, valid_pp)
- **On failure**: If model not in catalog, ask user for custom architecture parameters

For pre-defined models, use the variant dicts from the Model Catalog section below. For custom models, gather: name, layers, d, q_heads, kv_heads, head_dim, dense_ffn, total_params_B, active_params_B, expert_ffn, dense_layers. Then compute derived fields:
- `attn_per_layer = 2 * d * (q_heads + kv_heads) * head_dim`
- `expert_each = d * expert_ffn * 3`
- `shared_each = d * shared_expert_ffn * 3` (if shared experts) or `0`
- `router_each = d * N_EXPERTS` (or `0` for dense)
- `moe_layer_params = (N_EXPERTS * expert_each) + shared_each + router_each`
- `dense_layer_params = attn_per_layer + (d * dense_ffn * 3)`
- `valid_pp` = all divisors of `layers`

### Step 2: Select Hardware Platform
- **Mode**: `agentic`
- **Input**: `{{hardware}}`, `{{node_count}}` from user
- **Output**: Complete hardware configuration dict(s)
- **Validate**: All required fields populated (name, nodes, gpus, mem_gb, peak_tflops_bf16, inter_node_bw_gb, intra_node_bw_gbps)
- **On failure**: If custom hardware, ask user for 7 hardware parameters

Standard node counts: 64, 128, 256, 512 for P-family; 1, 4, 8, 16 for G-family.
For Mode A: include ALL standard node counts to sweep. For Mode B: use the specified node_count.

Hardware config template:
```python
{"name": "256 p5en", "gpus": 2048, "nodes": 256, "mem_gb": 141, "inter_node_bw_gb": 400, "intra_node_bw_gbps": 900, "peak_tflops_bf16": 989}
```

### Step 3: Configure Training Parameters
- **Mode**: `agentic`
- **Input**: Model family, `{{dataset_tokens}}`, user overrides
- **Output**: Complete config dict with smart defaults applied
- **Validate**: TOTAL_TOKENS set, PP divides layers evenly, EP divides N_EXPERTS evenly
- **On failure**: Suggest valid PP/EP values that are divisors of layers/N_EXPERTS

Apply smart defaults automatically based on model family:
- **Llama 3.1/3.2/3.3 dense**: VOCAB=128256, EP=1, N_EXPERTS=0, TOPK=0
- **Llama 4 MoE**: VOCAB=202048, EP=8 (Scout: N_EXPERTS=16, TOPK=1; Maverick: N_EXPERTS=128, TOPK=1)
- **Qwen2.5 dense**: VOCAB=152064, EP=1, N_EXPERTS=0, TOPK=0
- **Qwen3 dense**: VOCAB=151936, EP=1, N_EXPERTS=0, TOPK=0
- **Qwen3 MoE**: VOCAB=151936, N_EXPERTS=128, TOPK=8, EP=8
- **Qwen3.5 dense**: VOCAB=248320, EP=1, N_EXPERTS=0, TOPK=0
- **Qwen3.5 MoE**: VOCAB=248320, EP=8 (35B: N_EXPERTS=256, TOPK=8; 122B: N_EXPERTS=256, TOPK=8; 397B: N_EXPERTS=512, TOPK=10)
- **DeepSeek-V3/V3.1/V3.2**: VOCAB=129280, N_EXPERTS=256, TOPK=8, EP=8
- **DeepSeek-V4**: VOCAB=129280, N_EXPERTS=384, TOPK=6, EP=8

Other defaults: SEQ_LEN=4096, PP=1, TP=1, CP=1, PRECISION="BF16", MFU=0.40, TOKENS_PER_BATCH=4e6

**Important**: For large models (>70B dense params), TP=8 is typically needed (one full node). Check Phase 1 memory fit — if OOM with TP=1, retry with TP=8.

### Step 4: Run Calculator
- **Mode**: `deterministic`
- **Tool**: `run_python`
- **Input**: Validated model variants, hardware configs, and training config from Steps 1-3
- **Output**: Structured results dict with all 5 phases + CSV exports
- **Validate**: Results contain phase1_memory, phase2_batch, phase3_training_time keys with non-empty lists
- **On failure**: Check validation — PP must divide layers, EP must divide N_EXPERTS. Fix config and retry.

The bundled `ai_model_calculator.py` is automatically importable after this skill is loaded (no file paths needed — works on any OS). Simply import and call:

```python
from ai_model_calculator import run_calculator

variants = [...]   # Model variant dicts from Step 1
hardware = [...]   # Hardware config dicts from Step 2
config = {         # Merged config from Step 3
    "VOCAB": 128256,
    "SEQ_LEN": 4096,
    "N_EXPERTS": 0,
    "TOPK": 0,
    "PP": 1,
    "TP": 8,
    "CP": 1,
    "EP": 1,
    "TOTAL_TOKENS": 15e12,
    "PRECISION": "BF16",
    "MFU": 0.40,
}

results = run_calculator(variants, hardware, config, output_dir=f"{WORKSPACE_DIR}/artifacts")
```

The script exports CSVs to the output_dir automatically.

### Step 5: Present Results
- **Mode**: `agentic`
- **Input**: Results dict from Step 4, user's mode
- **Output**: Structured markdown summary with tables, recommendations, and actionable insights
- **Validate**: All relevant phases presented, mode-specific answer given first
- **On failure**: Inspect the results dict directly for troubleshooting

**Mode A presentation** — Lead with: "To complete training in ≤ X time, you need **N nodes (M GPUs)** of instance-type." Show comparison table of all node counts vs time. Identify the minimum config that meets the target.

**Mode B presentation** — Lead with: "With N nodes (M GPUs), training will take approximately **X days (Y-Z range)**."

**IMPORTANT — MFU disclaimer**: Always begin the results output with a disclaimer stating the MFU value used. Example: "> ⚠️ **Assumptions**: MFU = 0.40 (40% of peak GPU FLOPS). This is a conservative planning estimate. Actual MFU varies by workload — well-optimized training can achieve 0.45–0.55. Results scale linearly with MFU."

**Mode C presentation** — Show which configs fit vs OOM. Report minimum nodes/GPUs needed with the best ZeRO stage.

**All modes include:**
- Phase 1: Memory breakdown table (model, gradient, optimizer, activation, buffer, total, headroom)
- Phase 2: Top 3-5 batch configurations (micro batch, grad accum, tokens/batch, steps, assessment)
- Phase 3: Training time estimates with confidence range
- Phase 4: ZeRO communication overhead
- Phase 5: MoE all-to-all routing (skip for dense models)

### Step 6: Generate HTML Report (Optional)
- **Mode**: `agentic`
- **Input**: Results dict and CSV export files from Step 4
- **Output**: AWS-branded HTML report with charts
- **Validate**: HTML file generated and renders correctly
- **On failure**: Skip report generation — the markdown results are sufficient

Only generate if user explicitly requests an HTML report.

## Calculation Formulas

### Phase 1: Memory Analysis
- Model memory: `(attn + FFN + expert + embed) * PARAM_BYTES / (TP * PP)` (ZeRO-3 further divides by DP)
- Gradients: same size as model params; sharded by DP under ZeRO ≥ 2
- Optimizer: `model_params_per_gpu * 6 bytes`; sharded by DP under all ZeRO stages
- Activations: `SEQ_LEN * micro * d * 12 * layers_per_gpu * 0.45`
- Buffers: `4 GB (NCCL)` + MoE routing buffers
- Usable threshold: `GPU_MEM * 0.92`

### Phase 3: Training Time
- `flops_per_token = 6 * active_params_B * 1e9`
- `total_flops = flops_per_token * TOTAL_TOKENS`
- `effective_tflops = peak_tflops * MFU * zero_efficiency`
- `time_seconds = total_flops / (gpus * effective_tflops * 1e12)`
- Range: `[time * 0.75, time * 1.25]`

### ZeRO Efficiency
- ZeRO-1: 1.0 (optimizer states only)
- ZeRO-2: 0.9 (optimizer + gradients)
- ZeRO-3: 0.8 (full sharding)

## Output

The final deliverable includes:
1. **Markdown summary** in chat with structured tables for each phase
2. **CSV/JSON exports** in `artifacts/` directory
3. **HTML report** (optional, on request) with charts and metric cards

## Lessons Learned

### Do
- Apply smart defaults silently based on model family — users expect this
- Present the mode-specific answer FIRST before detailed phase breakdowns
- Validate PP divides layers and EP divides N_EXPERTS BEFORE running
- For Mode A, sweep all node counts and present a comparison table
- For MoE models, use total_params_B (not active) for memory calculations
- If Phase 1 shows OOM with TP=1, automatically retry with TP=8 (standard for models >70B)
- Offer to re-run with different parameters after presenting results

### Don't
- Don't use subprocess, git clone, os.chdir, or sys.path — they're blocked by the sandbox
- Don't use hardcoded paths or tilde (~) expansion — the skill must be portable across users and platforms
- Don't ask the user for derived fields (attn_per_layer, expert_each, etc.) — compute them
- Don't dump raw data — parse and present in clean markdown tables
- Don't run without confirming model and dataset size (Modes A/B)
- Don't assume node counts for Mode B — always ask

### Common Failures
- **OOM in Phase 1 with TP=1**: Large models (>70B) need TP=8. Retry automatically.
- **PP does not divide layers**: Check `layers % PP == 0`. Suggest valid PP values (divisors of layers).
- **EP does not divide N_EXPERTS**: Check `N_EXPERTS % EP == 0`. Suggest valid EP values.
- **EP too small causes OOM for MoE**: If model memory exceeds GPU after EP sharding, increase EP (e.g., EP=16 or EP=64 for DeepSeek-V3).

### When to Ask the User
- Which mode (A/B/C) if not clear from their question
- Dataset size in tokens (Modes A/B) — never assume a default
- Target time (Mode A) — must be explicit
- Node count (Mode B) — must be explicit
- Custom model parameters — if model not in the pre-defined catalog
- Whether to change training parameter defaults (offer but don't require)
