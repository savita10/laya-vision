# Experiments log

<!-- Format: date | commit hash | config summary | result -->
2026-10-02 | d75985c | P0 Gate 0 repro: upstream finetune, 1xT4 fp32, mb4/ga16/ep4, effective batch 64, ~2h | base acc 0.362 / ft acc 0.7765 (published 0.766) | ft Brier 0.051, ECE 0.143, within_1_level 0.991, p50 130ms | temps [1.060, 1.011, 1.161]
