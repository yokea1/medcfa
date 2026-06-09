# MedCFA

MedCFA is a training-free counterfactual evaluation framework for medical vision-language models.

## Overview

MedCFA compares model predictions under:

- Original image
- Matched-Patch lesion replacement
- Zero masking
- Blur masking

If a model changes from `Yes` to `No` after the lesion region is removed or corrupted, the prediction is considered sensitive to localized pathology evidence.

## Supported Datasets

- CheXlocalize
- ChestX-Det10

Raw medical images are **not included** due to dataset licenses and storage size.

## Repository Structure

```text
MedCFA/
├── scripts/
├── configs/
├── data/
├── results/
├── docs/
├── README.md
├── requirements.txt
├── environment.yml
└── .gitignore
```

## What to Upload

```text
scripts/
configs/
data/*.jsonl
results/*.csv
results/*.txt
results/*_results.jsonl
docs/
README.md
requirements.txt
environment.yml
.gitignore
```

## What Not to Upload

Do not upload:

```text
raw medical images
*.png / *.jpg / *.jpeg / *.dcm
temporary masked images
model weights
HuggingFace cache
Conda environment folders
```

## Main Metrics

### MP Delta Flip

```text
MP Delta Flip = Yes-to-No / Original Yes
```

### SCI

```text
SCI = Recall - MP Delta Flip
```

### BalSCI

```text
BalSCI = Balanced Accuracy - MP Delta Flip
```

### F1SCI

```text
F1SCI = F1 - MP Delta Flip
```

## Reproduce Evaluation

```bash
python scripts/eval_classification_with_negative.py
python scripts/eval_classification_with_negative.py --dataset-prefix chestxdet10
python scripts/eval_chestxdet10_flip_all.py
python scripts/eval_chestxdet10_sci.py
python scripts/bootstrap_ci.py
python scripts/bootstrap_ci.py --dataset-prefix chestxdet10
python scripts/diagnose_yes_rate.py
python scripts/diagnose_bbox_size.py
```

## Run Inference Example

```bash
PAIRS=/path/to/chestxdet10_pairs_test.jsonl

python scripts/run_qwen_vl.py \
  --config configs/qwen3vl.yaml \
  --condition original \
  --pairs $PAIRS \
  --split chestxdet10
```

## Reproducibility

All random sampling uses `SEED = 42`.

## Notes

This is a lightweight handoff package. Raw images should be downloaded separately from official dataset sources.

