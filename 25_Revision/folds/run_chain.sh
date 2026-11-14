#!/bin/bash
cd "$(dirname "$0")/.."
T=${1:-1}
DET="HA SeasonalNaive Ridge"
python train_fold.py --tag THR90  --train-days 5 --last-day 7 --threads $T --hv-thr 92.70 --models $DET AMSGCN
python train_fold.py --tag THR0   --train-days 5 --last-day 7 --threads $T --hv-thr 0     --models $DET AMSGCN
echo STAGE2_DONE
python train_fold.py --tag W      --train-days 3 --last-day 5 --threads $T
python train_fold.py --tag S1     --train-days 4 --last-day 6 --threads $T
python train_fold.py --tag STRICT --train-days 4 --last-day 7 --threads $T
echo CHAIN_DONE
