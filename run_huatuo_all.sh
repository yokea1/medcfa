#!/bin/bash
cd /home/217885@student.upm.edu.my/handoff_pkg

python scripts/run_qwen_vl.py --config configs/huatuogpt_vision.yaml --condition original
python scripts/run_qwen_vl.py --config configs/huatuogpt_vision.yaml --condition matched_patch
python scripts/run_qwen_vl.py --config configs/huatuogpt_vision.yaml --condition zero
python scripts/run_qwen_vl.py --config configs/huatuogpt_vision.yaml --condition blur
