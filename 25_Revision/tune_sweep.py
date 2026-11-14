"""E15 (R2.9): were the neural baselines given the same tuning as AMSGCN?

All six neural models share one protocol (Adam, learning rate 1e-3, weight decay 1e-4,
hidden size 32, batch 32, patience 20, gradient clipping 5). This sweep retrains every
model at hidden sizes 16, 32 and 64 with one seed on the main split (days 1 to 5 for
training, day 6 for early stopping and for choosing the size, day 7 for testing) and
records the cell MAE on both days, so each model can be compared at its own best size.

    python tune_sweep.py --models LSTM STGCN DCRNN --hidden 16 32 64
Output: outputs/e15_tuning.csv (appended, one row per model and size)
"""
import argparse, sys, time, csv
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "09_Forecasting_Rebuilt")); sys.path.insert(0, str(ROOT / "10_Allocation" / "scripts"))
import torch
from core import config as C, data as D
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=C.NEURAL_MODELS)
ap.add_argument("--hidden", nargs="*", type=int, default=[16, 32, 64])
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--threads", type=int, default=1)
args = ap.parse_args()
torch.set_num_threads(args.threads)
thr = 53.90
D.high_value_cut = lambda default=53.90: thr
df = D.load_orders(); sp = D.build_spatial(df, 0.010, include_dropoff=True)
bundle = D.build_bundle(df, sp, biz=None, need_dynamic=True, high_value_only=True)
x = bundle.raw; S = C.SLOTS_PER_DAY
def day_starts(d): return np.arange((d - 1) * S - C.SEQ_LEN, d * S - C.PRED_LEN - C.SEQ_LEN + 1)
vs, ts = day_starts(6), day_starts(7)
def truth(starts): return np.stack([x[s + C.SEQ_LEN:s + C.SEQ_LEN + C.PRED_LEN] for s in starts])
from core.engine import train_model, make_loader, collect
from core.data import WindowDataset
from core.models import baselines as B
from core.models.amsgcn import AMSGCN
mk = lambda idx: WindowDataset(bundle.train.x, idx, bundle.train.tod, bundle.train.dow, bundle.train.adj)
n = bundle.n_nodes
BUILD = {"LSTM": lambda h: B.LSTMBaseline(n, hidden=h), "STGCN": lambda h: B.STGCN(n, bundle.adj_static, hidden=h),
         "DCRNN": lambda h: B.DCRNN(n, bundle.adj_static, hidden=h), "GraphWaveNet": lambda h: B.GraphWaveNet(n, bundle.adj_static, hidden=h),
         "AGCRN": lambda h: B.AGCRN(n, hidden=h), "AMSGCN": lambda h: AMSGCN(n, hidden=h)}
out = Path(__file__).resolve().parent / "outputs" / "e15_tuning.csv"
done = set()
if out.exists():
    done = {(r["model"], int(r["hidden"])) for r in csv.DictReader(open(out))}
for name in args.models:
    for h in args.hidden:
        if (name, h) in done: continue
        t0 = time.time(); C.seed_everything(args.seed)
        model = BUILD[name](h); npar = sum(p.numel() for p in model.parameters())
        train_model(model, bundle, args.seed, max_epochs=150)
        v, _ = collect(model, make_loader(mk(vs), False), bundle.scaler, C.DEVICE)
        t, _ = collect(model, make_loader(mk(ts), False), bundle.scaler, C.DEVICE)
        ev, et = np.abs(v - truth(vs)), np.abs(t - truth(ts))
        row = dict(model=name, hidden=h, seed=args.seed, params=npar, val_mae_all=float(ev.mean()), val_mae_60=float(ev[:, 3].mean()),
                   test_mae_all=float(et.mean()), test_mae_15=float(et[:, 0].mean()), test_mae_60=float(et[:, 3].mean()), sec=round(time.time() - t0, 1))
        new = not out.exists()
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row)); 
            if new: w.writeheader()
            w.writerow(row)
        print(row, flush=True)
print("SWEEP_DONE", flush=True)
