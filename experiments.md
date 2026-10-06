# Experiments log

<!-- Format: date | commit hash | config summary | result -->
2026-10-02 | d75985c | P0 Gate 0 repro: upstream finetune, 1xT4 fp32, mb4/ga16/ep4, effective batch 64, ~2h | base acc 0.362 / ft acc 0.7765 (published 0.766) | ft Brier 0.051, ECE 0.143, within_1_level 0.991, p50 130ms | temps [1.060, 1.011, 1.161]
2026-10-06 | 7d64b5d | P1 Route A baseline (no training): Qwen2-VL-2B @895c3a4 describe, prompt a2, native res, greedy, max 192 tokens -> Gate-0 ft Laya zero-shot; val comparison set 2,000 (seed 11) | acc 0.501 (majority floor 0.0645) | VLM p50 7.5 s, Laya p50 41 ms, 7.6 GPU-s/image | 590/2000 descriptions hit the token cap, 0 errors
