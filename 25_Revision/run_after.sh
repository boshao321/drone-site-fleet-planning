#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
while pgrep -f "run_exps.sh" > /dev/null; do sleep 5; done
python e13_rolling.py STRICT > outputs/e13_strict.log 2>&1
echo "== e13 STRICT end $(date +%T)" >> outputs/chain.log
