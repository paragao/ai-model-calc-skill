# AI Model Training Calculator — Amazon Quick Skill

A self-contained Amazon Quick skill that calculates GPU memory, batch configuration, training time, and communication overhead for LLM training on AWS hardware.

## Supported Models

Dense: Llama 3.1/3.2/3.3, Qwen 2.5/3/3.5  
MoE: DeepSeek V3/V3.1/V3.2/V4, Qwen3-MoE, Llama 4 Scout/Maverick

## Supported Hardware

p5 (H100), p5en (H200), p6-b200 (B200), p6-b300 (B300 Ultra), p6e-gb200 (GB200 NVL), g5, g6, g6e

## Installation

1. Copy the `amazon-quick-skill/` folder (this directory) to your Amazon Quick skills path:

   ```bash
   cp -r amazon-quick-skill ~/.quickwork/profiles/federate-prod/skills/ai-model-calc
   ```

2. Restart Amazon Quick or start a new conversation.

3. The skill activates automatically when you ask about training infrastructure — for example:
   - *"How many p5en nodes do I need to train Llama-3.1-70B on 15T tokens in 2 months?"*
   - *"How long would it take to train DeepSeek-V3 on 256 p6-b200 instances?"*
   - *"What's the minimum hardware to fit a 405B model?"*

## Contents

```
amazon-quick-skill/
├── SKILL.md                        # Skill definition
├── scripts/
│   └── ai_model_calculator.py      # Calculator engine (auto-imported)
└── README.md                       # This file
```

## Requirements

- Amazon Quick desktop app
- No external dependencies — fully self-contained, works on macOS, Windows, and Linux
