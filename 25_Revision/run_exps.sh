#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
for s in "$@"; do
  echo "== $s start $(date +%T)" >> outputs/chain.log
  python $s.py > outputs/$s.log 2>&1 || echo "== $s FAILED" >> outputs/chain.log
  echo "== $s end $(date +%T)" >> outputs/chain.log
done
echo EXPS_DONE >> outputs/chain.log
