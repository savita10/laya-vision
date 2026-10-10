# Model card: Laya-Vision

## Comparison to published results

RVL-CDIP document-type accuracy (16 classes), as published. Laya-Vision and
Route A were evaluated in this work; every other row is copied from the
cited source.

| Model | Accuracy (%) | Input | Test set | Source |
|---|---|---|---|---|
| Bi-VLDoc | 97.12 | image + text + layout | full RVL-CDIP test (40k) | Luo et al. 2022, via Larson et al. 2023, Table 1 |
| ERNIE-Layout-large | 96.27 | image + text + layout | full RVL-CDIP test (40k) | Larson et al. 2023, Table 1 |
| LayoutLMv3-large | 95.93 | image + text + layout | full RVL-CDIP test (40k) | Larson et al. 2023, Table 1 |
| DocXClassifierFPN-XL | 94.19 | image only | full RVL-CDIP test (40k) | Saifullah et al. 2024 |
| VGG-16 | 90.97 | image only | full RVL-CDIP test (40k) | Afzal et al. 2017 |
| Harley et al. 2015 (the dataset paper) | 89.80 | image only | full RVL-CDIP test (40k) | Harley et al. 2015 |
| **Laya-Vision (combined view)** | **86.24** | image + OCR text | **our 5,000-image stratified subsample of the RVL-CDIP test split** | this work |
| Route A (Qwen2-VL-2B describe → Laya, zero-shot) | 51.0 | image (via generated description) | 1,000 images from our 5,000-image test subsample | this work |

**Laya-Vision is not evaluated on the full RVL-CDIP test split.** Its 86.24 is
on a 5,000-image stratified subsample (about 313 images per class) of the
40,000-image test split, scored once. The published rows use the full test
split. Route A's 51.0 is on a pre-registered 1,000-image stratified subset of
that 5,000-image subsample. On the same 1,000 images, Laya-Vision (combined)
scores 87.3.

GPT-4-Vision is deliberately not listed: its reported 69.9 is on a 160 × 5
subset, which is not comparable to the results above.

### Caveats
1. **Training data.** Laya-Vision is trained on a 60,000-image stratified
   subsample of RVL-CDIP train. The published models train on the full
   320,000 images.
2. **Model capacity and input.** Laya-Vision keeps both encoders frozen:
   SigLIP base at 384 × 384 px for the image, and the Laya backbone for the
   text. Only a 28M-parameter adapter is trained (projector + decision
   head). The top published models are full fine-tunes, and the top three
   also use layout features (OCR word positions).
3. **Different objective.** Laya-Vision is built for calibrated typed
   decisions at about 15 ms per page (L4, batched, model forward only). On
   this test subsample it reaches ECE 0.0145 after temperature scaling.
   Neither calibration nor cost is measured by the leaderboard, which
   ranks top-1 accuracy alone.

### Label noise
Larson et al. (2023) estimate that **8.1% of the RVL-CDIP test labels are
wrong**, so the top leaderboard scores partly fit label noise. The same
noise is in our test subsample, and it caps every model's measurable
accuracy.

### Planned for v1.1
- Evaluate on the full 40,000-image RVL-CDIP test split, so the result is
  directly comparable with the rows above.
- Train on the full 320,000-image training split.

### References
- Afzal, M. Z., Kölsch, A., Ahmed, S., & Liwicki, M. (2017). Cutting the
  error by half: Investigation of very deep CNN and advanced training
  strategies for document image classification. *Proceedings of the 14th
  IAPR International Conference on Document Analysis and Recognition
  (ICDAR 2017)*. arXiv:1704.03557.
- Harley, A. W., Ufkes, A., & Derpanis, K. G. (2015). Evaluation of deep
  convolutional nets for document image classification and retrieval.
  *Proceedings of the 13th International Conference on Document Analysis
  and Recognition (ICDAR 2015)*.
- Huang, Y., Lv, T., Cui, L., Lu, Y., & Wei, F. (2022). LayoutLMv3:
  Pre-training for document AI with unified text and image masking.
  *Proceedings of the 30th ACM International Conference on Multimedia
  (ACM MM 2022)*. Cited for the LayoutLMv3-large model; the 95.93 is
  taken from Larson et al. (2023), Table 1.
- Larson, S., Lim, G., & Leach, K. (2023). On evaluation of document
  classification with RVL-CDIP. *Proceedings of the 17th Conference of
  the European Chapter of the Association for Computational Linguistics
  (EACL 2023)*.
- Luo, C., et al. (2022). Bi-VLDoc: Bidirectional vision-language
  modeling for visually-rich document understanding. arXiv preprint.
  The 97.12 is taken from Larson et al. (2023), Table 1.
- Peng, Q., et al. (2022). ERNIE-Layout: Layout knowledge enhanced
  pre-training for visually-rich document understanding. *Findings of
  the Association for Computational Linguistics: EMNLP 2022*. Cited for
  the ERNIE-Layout-large model; the 96.27 is taken from Larson et al.
  (2023), Table 1.
- Saifullah, S., Agne, S., Dengel, A., & Ahmed, S. (2024).
  DocXClassifier: Towards a robust and interpretable deep neural network
  for document image classification. *International Journal on Document
  Analysis and Recognition (IJDAR)*, 27(3), 447–473.
  https://doi.org/10.1007/s10032-024-00483-w
