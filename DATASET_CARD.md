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

Document type is the only labelled task in v1. It covers all 60k / 5k / 5k
records, straight from the converter.

## Secondary labels (explored, not in v1)
- **Handwriting (noul): explored and deferred to v2.** Distilled with
  Qwen2-VL-2B-Instruct (prompt v3: a direct question plus an inverse
  question to cancel the model's yes-bias, signatures excluded;
  transformers 4.49.0). The label is marginal. It detects dominant
  handwriting but misses sparse or marginal handwriting: on 16 val images
  at mid-res it ranked 5 of 6 handwritten documents in the top 5 (native
  resolution: 2 of 6), but 4 of 6 scored 0.27-0.40, overlapping clean
  printed pages. The limit appears to be RVL-CDIP's ~90 dpi scans
  combined with a 2B-parameter VLM. Distilling 60k images would take
  ~59 h on an L4 (0.284 img/s; batching runs out of memory), too much
  GPU time for a weak auxiliary signal, so v1 ships no handwriting labels.
- **Legibility (score): dropped.** The distilled label was degenerate
  (14/16 one level, probabilities 0.34-0.60) and carried no training
  signal. A future version should use a deterministic sharpness/contrast
  proxy rather than a VLM label.
