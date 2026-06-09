#!/bin/bash
cd /home/217885@student.upm.edu.my/handoff_pkg

python scripts/run_llava_med_official.py --condition original
python scripts/run_llava_med_official.py --condition matched_patch
python scripts/run_llava_med_official.py --condition zero
python scripts/run_llava_med_official.py --condition blur
