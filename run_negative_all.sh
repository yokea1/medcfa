#!/bin/bash
cd /home/217885@student.upm.edu.my/handoff_pkg

NEG=/home/217885@student.upm.edu.my/medcfa/data/medcfa_pairs_negative_test.jsonl

python scripts/run_qwen_vl.py --config configs/qwen25vl.yaml --condition original --pairs $NEG --split negative
python scripts/run_qwen_vl.py --config configs/huatuogpt_vision.yaml --condition original --pairs $NEG --split negative

python scripts/run_medgemma.py --condition original --pairs $NEG --split negative
python scripts/run_internvl3.py --condition original --pairs $NEG --split negative
python scripts/run_llava_med_official.py --condition original --pairs $NEG --split negative
