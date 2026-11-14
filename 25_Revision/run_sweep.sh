#!/bin/bash
# after the threshold forecasts are done, stop the remaining chain and run the tuning sweep
cd "$(dirname "$0")"
until grep -q STAGE2_DONE folds/train.log; do sleep 10; done
pkill -f run_chain.sh; pkill -f train_fold.py
echo SWEEP_START >> folds/train.log
python tune_sweep.py --models AMSGCN AGCRN GraphWaveNet DCRNN STGCN LSTM --hidden 64 16 32 >> outputs/e15.log 2>&1
