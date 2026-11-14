#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
python e03_sensitivity.py thr=P50 thr=P90 thr=all > outputs/e03_thr.log 2>&1
echo "== thresholds end $(date +%T)" >> outputs/chain.log
python tune_sweep.py --models LSTM STGCN DCRNN GraphWaveNet AGCRN AMSGCN --hidden 64 16 32 >> outputs/e15b.log 2>&1
