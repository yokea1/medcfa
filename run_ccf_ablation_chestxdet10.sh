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

ops=(
matched_patch
zero
blur
)

for op in "${ops[@]}"; do
  for m in "${models[@]}"; do
    echo "Running CCF: model=$m op=$op"

    python code/cfa_ccf.py \
      --original results/${m}_chestxdet10_original_results.jsonl \
      --counterfactual results/${m}_chestxdet10_${op}_results.jsonl \
      --output results/${m}_ccf_${op}.jsonl \
      --operator ${op}

    python code/cfa_ccf_summary.py \
      --input results/${m}_ccf_${op}.jsonl \
      --output results/${m}_ccf_${op}_summary.csv
  done
done
