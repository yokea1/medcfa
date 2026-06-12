#!/bin/bash
set -e

PAIR_FILE="data/crs_chestxdet10_100_positive.jsonl"
GRID=4

declare -A CONFIGS
CONFIGS[qwen25vl]="configs/qwen25vl.yaml"
CONFIGS[qwen3vl]="configs/qwen3vl.yaml"
CONFIGS[huatuogpt_vision]="configs/huatuogpt_vision.yaml"
CONFIGS[medgemma]="configs/medgemma.yaml"
CONFIGS[internvl3]="configs/internvl3.yaml"
CONFIGS[llava_med]="configs/llava_med.yaml"

for MODEL in qwen25vl qwen3vl huatuogpt_vision medgemma internvl3 llava_med
do
    echo "======================================"
    echo "Running CRS for $MODEL"
    echo "======================================"

    python scripts/run_qwen_vl_crs.py \
        --config "${CONFIGS[$MODEL]}" \
        --pairs "$PAIR_FILE" \
        --grid-size "$GRID" \
        --split chestxdet10_crs_100

    python code/cfa_crs_metrics.py \
        --input "/home/217885@student.upm.edu.my/medcfa/results/${MODEL}_chestxdet10_crs_100_grid4_crs.jsonl" \
        --output "results/${MODEL}_crs_metrics.csv"
done
