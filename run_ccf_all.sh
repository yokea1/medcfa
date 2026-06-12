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
  echo "Running CCF for $m"
  echo "=============================="

  python code/cfa_ccf.py \
    --original results/${m}_chestxdet10_original_results.jsonl \
    --counterfactual results/${m}_chestxdet10_matched_patch_results.jsonl \
    --output results/${m}_ccf.jsonl \
    --operator matched_patch

  python code/cfa_ccf_summary.py \
    --input results/${m}_ccf.jsonl \
    --output results/${m}_ccf_summary.csv
done
