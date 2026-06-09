#!/bin/bash
set -e

cd /home/217885@student.upm.edu.my/handoff_pkg

PAIRS=/home/217885@student.upm.edu.my/medcfa/data/chestxdet10_pairs_test.jsonl

# Qwen2.5-VL
python scripts/run_qwen_vl.py --config configs/qwen25vl.yaml --condition original --pairs $PAIRS --split chestxdet10
python scripts/run_qwen_vl.py --config configs/qwen25vl.yaml --condition matched_patch --pairs $PAIRS --split chestxdet10
python scripts/run_qwen_vl.py --config configs/qwen25vl.yaml --condition zero --pairs $PAIRS --split chestxdet10
python scripts/run_qwen_vl.py --config configs/qwen25vl.yaml --condition blur --pairs $PAIRS --split chestxdet10

# HuatuoGPT-Vision
python scripts/run_qwen_vl.py --config configs/huatuogpt_vision.yaml --condition original --pairs $PAIRS --split chestxdet10
python scripts/run_qwen_vl.py --config configs/huatuogpt_vision.yaml --condition matched_patch --pairs $PAIRS --split chestxdet10
python scripts/run_qwen_vl.py --config configs/huatuogpt_vision.yaml --condition zero --pairs $PAIRS --split chestxdet10
python scripts/run_qwen_vl.py --config configs/huatuogpt_vision.yaml --condition blur --pairs $PAIRS --split chestxdet10

# MedGemma
python scripts/run_medgemma.py --condition original --pairs $PAIRS --split chestxdet10
python scripts/run_medgemma.py --condition matched_patch --pairs $PAIRS --split chestxdet10
python scripts/run_medgemma.py --condition zero --pairs $PAIRS --split chestxdet10
python scripts/run_medgemma.py --condition blur --pairs $PAIRS --split chestxdet10

# InternVL3
python scripts/run_internvl3.py --condition original --pairs $PAIRS --split chestxdet10
python scripts/run_internvl3.py --condition matched_patch --pairs $PAIRS --split chestxdet10
python scripts/run_internvl3.py --condition zero --pairs $PAIRS --split chestxdet10
python scripts/run_internvl3.py --condition blur --pairs $PAIRS --split chestxdet10

# LLaVA-Med
python scripts/run_llava_med_official.py --condition original --pairs $PAIRS --split chestxdet10
python scripts/run_llava_med_official.py --condition matched_patch --pairs $PAIRS --split chestxdet10
python scripts/run_llava_med_official.py --condition zero --pairs $PAIRS --split chestxdet10
python scripts/run_llava_med_official.py --condition blur --pairs $PAIRS --split chestxdet10

wc -l results/*_chestxdet10_*_results.jsonl
