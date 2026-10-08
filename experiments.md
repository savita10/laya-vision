# Experiments log

<!-- Format: date | commit hash | config summary | result -->
2026-10-02 | d75985c | P0 Gate 0 repro: upstream finetune, 1xT4 fp32, mb4/ga16/ep4, effective batch 64, ~2h | base acc 0.362 / ft acc 0.7765 (published 0.766) | ft Brier 0.051, ECE 0.143, within_1_level 0.991, p50 130ms | temps [1.060, 1.011, 1.161]
2026-10-06 | 7d64b5d | P1 Route A baseline (no training): Qwen2-VL-2B @895c3a4 describe, prompt a2, native res, greedy, max 192 tokens -> Gate-0 ft Laya zero-shot; val comparison set 2,000 (seed 11) | acc 0.501 (majority floor 0.0645) | VLM p50 7.5 s, Laya p50 41 ms, 7.6 GPU-s/image | 590/2000 descriptions hit the token cap, 0 errors
2026-10-08 | e2ecf73 | P2 precompute (no training): p2v1, SigLIP base-384 @41aec1c8 8x8 fp16, Tesseract LSTM psm3 eng; 1xL4; train 60k / val 5k / test 5k, 36 shards, 7.0 GB | 0 errors; 8.24 img/s overall (SigLIP 32, Tesseract 7.3 on 12 workers); train ~2 h, val ~17 min, test ~15 min | val NN check (32-token view, mean): 1-NN 0.756 (12x chance), 5-NN 0.776; 7 near-dup pairs, 0 cross-class
