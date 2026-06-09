#!/bin/bash
set -e

cd /home/217885@student.upm.edu.my/handoff_pkg

PAIRS=/home/217885@student.upm.edu.my/medcfa/data/chestxdet10_pairs_test.jsonl

rm -f results/qwen3vl_chestxdet10_original_results.jsonl
rm -f results/qwen3vl_chestxdet10_matched_patch_results.jsonl
rm -f results/qwen3vl_chestxdet10_zero_results.jsonl
rm -f results/qwen3vl_chestxdet10_blur_results.jsonl

python scripts/run_qwen_vl.py --config configs/qwen3vl.yaml --condition original --pairs $PAIRS --split chestxdet10
python scripts/run_qwen_vl.py --config configs/qwen3vl.yaml --condition matched_patch --pairs $PAIRS --split chestxdet10
python scripts/run_qwen_vl.py --config configs/qwen3vl.yaml --condition zero --pairs $PAIRS --split chestxdet10
python scripts/run_qwen_vl.py --config configs/qwen3vl.yaml --condition blur --pairs $PAIRS --split chestxdet10

wc -l results/qwen3vl_chestxdet10_*_results.jsonl
