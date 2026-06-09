#!/bin/bash
cd /home/217885@student.upm.edu.my/handoff_pkg

python scripts/run_medgemma.py --condition original
python scripts/run_medgemma.py --condition matched_patch
python scripts/run_medgemma.py --condition zero
python scripts/run_medgemma.py --condition blur
