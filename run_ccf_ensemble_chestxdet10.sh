#!/bin/bash
set -e

models=(
qwen25vl
qwen3vl
huatuogpt_vision
medgemma
internvl3
llava_med
)

for m in "${models[@]}"; do
  echo "=============================="
  echo "Running All-3 Ensemble CCF for $m"
  echo "=============================="

  python code/cfa_ccf_ensemble.py \
    --original results/${m}_chestxdet10_original_results.jsonl \
    --matched-patch results/${m}_chestxdet10_matched_patch_results.jsonl \
    --zero results/${m}_chestxdet10_zero_results.jsonl \
    --blur results/${m}_chestxdet10_blur_results.jsonl \
    --output results/${m}_ccf_ensemble.jsonl \
    --summary results/${m}_ccf_ensemble_summary.csv
done
