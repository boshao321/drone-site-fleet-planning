"""Forecasts for a chronological fold (revision, 2026-11).

    python train_fold.py --tag W  --train-days 3 --last-day 5
    python train_fold.py --tag S1 --train-days 4 --last-day 6
    python train_fold.py --tag STRICT --train-days 4 --last-day 7

Neural models are fitted on days 1..train_days and stopped early on day
train_days+1. Predictions are written for every later day. The deterministic
models (HA, seasonal naive, ridge) are fitted on all days before the
calibration day, which is day last_day-1. The high value threshold is the 75th
percentile of order value on the days before the test day (last_day), so no
step sees the scored day.

Output (folds/<tag>/forecasts): the same files 10_Allocation writes, so
core_pd.Study can read them: demand_full.npy, node_coords.npy, val_starts.npy
(calibration day), test_starts.npy (test day), valpred_<m>.npy, pred_<m>.npy.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "09_Forecasting_Rebuilt")); sys.path.insert(0, str(ROOT / "10_Allocation" / "scripts"))
import torch
from core import config as C, data as D

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--train-days", type=int, required=True)
ap.add_argument("--last-day", type=int, required=True)
ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
ap.add_argument("--max-epochs", type=int, default=150)
ap.add_argument("--models", nargs="*", default=None)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--target", default="high_value", choices=["high_value", "all"])
ap.add_argument("--hv-thr", type=float, default=None, help="override the value threshold")
args = ap.parse_args()
torch.set_num_threads(args.threads)

n_tr, last = args.train_days, args.last_day
cal_day, test_day = last - 1, last                      # 1-based
C.TRAIN_DAYS, C.VAL_DAYS, C.N_DAYS = n_tr, 1, last
C.TEST_DAYS = last - n_tr - 1
C.TOTAL_SLOTS = C.SLOTS_PER_DAY * C.N_DAYS

df_all = D.load_orders()                                # days 1..last only (slot filter)
thr = args.hv_thr if args.hv_thr is not None else float(np.percentile(df_all[df_all["dt_seq_id"] < test_day]["package_price"], 75))
D.high_value_cut = lambda default=53.90: thr

# node set from the full week, so cell indices mean the same place in every fold
C.TOTAL_SLOTS = C.SLOTS_PER_DAY * 7
sp = D.build_spatial(D.load_orders(), 0.010, include_dropoff=True)
C.TOTAL_SLOTS = C.SLOTS_PER_DAY * C.N_DAYS

out = Path(__file__).resolve().parent / "folds" / args.tag / ("forecasts" if args.target == "high_value" else "forecasts_all")
out.mkdir(parents=True, exist_ok=True)
bundle = D.build_bundle(df_all, sp, biz=None, need_dynamic=True, high_value_only=(args.target == "high_value"))
x = bundle.raw
S = C.SLOTS_PER_DAY
def day_starts(d):                                      # d 1-based; windows whose four targets lie in day d
    return np.arange((d - 1) * S - C.SEQ_LEN, d * S - C.PRED_LEN - C.SEQ_LEN + 1)
val_starts, test_starts = day_starts(cal_day), day_starts(test_day)
np.save(out / "node_coords.npy", sp.coords); np.save(out / "demand_full.npy", x)
np.save(out / "val_starts.npy", val_starts); np.save(out / "test_starts.npy", test_starts)
json.dump({"tag": args.tag, "train_days": n_tr, "es_day": n_tr + 1, "cal_day": cal_day, "test_day": test_day,
           "hv_threshold": thr, "seeds": args.seeds, "target": args.target}, open(out / "meta.json", "w"), indent=1)
print(f"fold {args.tag}: neural train 1..{n_tr}, early stop day {n_tr+1}, calibrate day {cal_day}, test day {test_day}, HV cut {thr:.2f}", flush=True)

from core.engine import train_model, make_loader, collect
from core.models.statistical import HistoricalAverage, RidgeLinear, SeasonalNaive
import run_unified_forecasts as R

tod = (np.arange(C.TOTAL_SLOTS) % S).astype(np.int64)
det_train = np.arange(0, (cal_day - 1) * S)             # all days before the calibration day
def truth(starts): return np.stack([x[s + C.SEQ_LEN:s + C.SEQ_LEN + C.PRED_LEN] for s in starts])
def mae60(p, starts): return float(np.abs(p[:, 3] - truth(starts)[:, 3]).mean())

models = args.models or C.ALL_MODELS
log = json.load(open(out / 'accuracy.json')) if (out / 'accuracy.json').exists() else {}
for name in models:
    if (out / f"pred_{name}.npy").exists() and (out / f"valpred_{name}.npy").exists():
        print(f"  {name}: on disk, skipped", flush=True); continue
    t0 = time.time()
    if name == "HA":
        m = HistoricalAverage().fit(x[det_train], tod[det_train]); vp, tp = m.predict(val_starts, tod), m.predict(test_starts, tod)
    elif name == "SeasonalNaive":
        m = SeasonalNaive().fit(x); vp, tp = m.predict(val_starts), m.predict(test_starts)
    elif name == "Ridge":
        tr_starts = np.arange(0, (cal_day - 1) * S - C.PRED_LEN - C.SEQ_LEN + 1)
        m = RidgeLinear().fit(x, tr_starts); vp, tp = m.predict_from(x, val_starts, sp.n), m.predict_from(x, test_starts, sp.n)
    else:
        from core.data import WindowDataset
        mk = lambda idx: WindowDataset(bundle.train.x, idx, bundle.train.tod, bundle.train.dow, bundle.train.adj)
        vps, tps = [], []
        for seed in args.seeds:
            fs = out / f"_seed_{name}_{seed}.npz"                              # per seed file, so a stopped run resumes
            if fs.exists():
                z = np.load(fs); vps.append(z["v"]); tps.append(z["t"]); continue
            C.seed_everything(seed)
            model = R.build_neural(name, bundle)
            train_model(model, bundle, seed, max_epochs=args.max_epochs)      # early stopping on bundle.val (day n_tr+1)
            v, _ = collect(model, make_loader(mk(val_starts), False), bundle.scaler, C.DEVICE)
            t, _ = collect(model, make_loader(mk(test_starts), False), bundle.scaler, C.DEVICE)
            np.savez(fs, v=v.astype(np.float32), t=t.astype(np.float32))
            vps.append(v); tps.append(t)
            print(f"    {name} seed {seed} done ({time.time()-t0:.0f} s)", flush=True)
        vp, tp = np.mean(vps, axis=0), np.mean(tps, axis=0)
    np.save(out / f"valpred_{name}.npy", vp.astype(np.float32)); np.save(out / f"pred_{name}.npy", tp.astype(np.float32))
    log[name] = {"cal_MAE60": mae60(vp, val_starts), "test_MAE60": mae60(tp, test_starts), "sec": round(time.time() - t0, 1)}
    print(f"  {name:<14} cal MAE@t+60 {log[name]['cal_MAE60']:.4f}  test MAE@t+60 {log[name]['test_MAE60']:.4f}  ({log[name]['sec']} s)", flush=True)
    json.dump(log, open(out / "accuracy.json", "w"), indent=1)
print("done", flush=True)
