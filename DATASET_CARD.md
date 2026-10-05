# Laya-Vision RVL-CDIP dataset card (draft)

Draft for the P1 gate; sections are filled in as P1 closes.

## Source
RVL-CDIP (aharley/rvl_cdip, pinned revision 03f14a4), 16 document classes.
Stratified subsample: 60k train / 5k val / 5k test. The test split is
frozen and read only by final_eval.py.

## Primary label: document type
16-class choice, taken from the RVL-CDIP labels. Hand audit of 200
stratified train samples: 96.5% correct, 2% mislabelled (concentrated in
form / email / invoice), 1.5% too unclear to judge (DECISIONS.md 2026-10-03).

## Secondary labels
- **Handwriting (noul): distilled, train-only auxiliary.** Soft label from
  Qwen2-VL-2B-Instruct (prompt v3, a direct question plus an inverse
  question for debiasing, signatures excluded; transformers 4.49.0;
  mid-res input). It is used only as an auxiliary training target and is
  not a reported evaluation metric. It detects dominant handwriting but
  misses sparse or marginal handwriting at RVL-CDIP's ~90 dpi: on 16 val
  images it ranked 5 of 6 handwritten documents in the top 5, but 4 of 6
  scored 0.27-0.40, overlapping clean printed pages.
- **Legibility (score): dropped.** The distilled label was degenerate
  (14/16 one level, probabilities 0.34-0.60) and carried no training
  signal.
