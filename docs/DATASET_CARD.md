# Laya-Vision RVL-CDIP dataset card

The data used to train and evaluate Laya-Vision v1 (document routing).
Final for Gate 1 (DECISIONS.md 2026-10-06).

## Source
- RVL-CDIP (Harley, Ufkes and Derpanis, "Evaluation of Deep Convolutional
  Nets for Document Image Classification and Retrieval", ICDAR 2015), via
  Hugging Face `aharley/rvl_cdip` at pinned revision
  `03f14a4ad0a32413eff51ca10f9f511545f2bd5b`.
- 400,000 grayscale scans of business documents (320k train / 40k val /
  40k test in the source), 16 classes: letter, form, email, handwritten,
  advertisement, scientific report, scientific publication,
  specification, file folder, news article, budget, invoice, presentation,
  questionnaire, resume, memo. The scans are low resolution (~90 dpi,
  longest side ~1000 px).

## Licence and terms of use
- The Hugging Face card lists the licence as "other": "RVL-CDIP is a subset
  of IIT-CDIP, which came from the Legacy Tobacco Document Library, for
  which license information can be found here"
  (https://www.industrydocuments.ucsf.edu/help/copyright/).
- This project treats RVL-CDIP as research-use data and uses it for
  research and benchmarking only. The repo redistributes no images; it
  holds record ids, labels and split files that point into the source
  dataset. Anyone reusing the data or the models trained on it should
  check the UCSF Industry Documents Library terms above.
- Laya-Vision's own code and weights are Apache-2.0 (Phase 5). That
  licence does not extend to the RVL-CDIP images.

## Splits
- Built by `scripts/build_dataset.py` from the source label files only (no
  images read), seed `20261003`. Each source split is sampled
  independently and stratified by class: train from train, val from val,
  test from test. Output: `<split>.jsonl` + `splits_manifest.json`
  (source sha256s, per-class counts, script git hash).

| Split | Records | Per class | Source |
|-------|---------|-----------|--------|
| train | 60,000 | 3,732–3,770 | RVL-CDIP train (320k) |
| val | 5,000 | 303–322 | RVL-CDIP val (40k) |
| test | 5,000 | stratified | RVL-CDIP test (40k) |

- **Test is frozen** (2026-10-03) and read only by `final_eval.py`, once,
  in Phase 4.
- **Fixed comparison set:** `splits/val_comparison_2000_seed11.json`, a
  stratified 2,000-image val subset (seed 11, 121–129 per class; records
  the sha256 of the val.jsonl it was drawn from). Route A and Laya-Vision
  are both scored on these ids. Majority-class floor: 0.0645 (invoice).

## Labelled task (v1): document type
- **Document type is v1's only labelled task:** a 16-class choice question
  (`doc_type`, "What type of document is this?"). Labels come straight
  from RVL-CDIP and cover every train / val / test record.
- **Hand audit** (200 stratified train samples): 96.5% correct, 2%
  mislabelled, 1.5% too unclear to judge. Mislabels concentrate in
  structured business documents (form, email, invoice). Read per-class
  confusion among those classes against this noise. Noise plus
  illegibility give a ~3.5% effective ceiling: test accuracy above ~98%
  would suggest leakage or noise-fitting, not skill.

## Explored and deferred (not in v1)
- **Handwriting (noul): deferred to v2.** Distilled with
  Qwen2-VL-2B-Instruct (prompt v3: a direct question plus an inverse
  question to cancel the model's yes-bias, signatures excluded;
  transformers 4.49.0). The label is marginal. It detects dominant
  handwriting but misses sparse or marginal handwriting: on 16 val images
  at mid-res it ranked 5 of 6 handwritten documents in the top 5 (native
  resolution: 2 of 6), but 4 of 6 scored 0.27-0.40, overlapping clean
  printed pages. The limit appears to be the ~90 dpi scans combined with a
  2B-parameter VLM. Distilling 60k images would take ~59 h on an L4
  (0.284 img/s; batching runs out of memory), too much GPU time for a weak
  auxiliary signal. Evidence: /content/drive/MyDrive/laya-vision/phase1/.
- **Legibility (score): dropped.** The distilled label was degenerate
  (14/16 one level, probabilities 0.34-0.60) and carried no training
  signal. A future version should use a deterministic sharpness/contrast
  proxy rather than a VLM label.

## Baselines on the comparison set (val, 2,000)
| Baseline | Accuracy | Latency p50 | GPU-s/image |
|----------|----------|-------------|-------------|
| Majority class | 0.0645 | – | – |
| Route A: Qwen2-VL-2B description → Gate-0 Laya, zero-shot | 0.501 | VLM 7.5 s + Laya 41 ms | 7.6 |

In Route A, 590/2,000 descriptions hit the 192-token cap. The main
confusions are among letter / email / memo, form / questionnaire and
scientific report / scientific publication: classes told apart mainly by
layout, which a text description loses.

## Known limitations
- Low-resolution, grayscale scans of mostly English US tobacco-industry
  business documents. Not representative of modern, colour, multilingual
  or born-digital documents.
- About 2% label noise (above), concentrated in form / email / invoice.
- Prior work reports label noise and near-duplicate documents across
  RVL-CDIP's source splits (Larson et al., "On Evaluation of Document
  Classification using RVL-CDIP", EACL 2023). This project does not
  deduplicate across splits.
