"""Stage 1 of the predict-then-optimize pipeline: forecasts on the siting node set.

The forecasting stage in 09 used the 93 pickup grids while the siting stage used
the 120-grid union of pickup and dropoff endpoints. A forecast defined on one
universe cannot be handed to an optimizer defined on the other, so this script
re-runs the forecasting stage on the union, and saves the predictions themselves
rather than only their error metrics.

The target is total order origination per grid per 15 minutes, pooled over the
four business lines. That is the quantity that consumes aircraft time at a
launch site, so it is the quantity the allocation stage needs. Per-business-line
accuracy remains the subject of the forecasting section; this run exists to feed
the optimizer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "09_Forecasting_Rebuilt"))

from core import config as C            # noqa: E402
from core import data as D              # noqa: E402
from core.engine import train_model, make_loader, collect   # noqa: E402
from core.metrics import per_horizon    # noqa: E402
from core.models import baselines as B  # noqa: E402
from core.models.amsgcn import AMSGCN   # noqa: E402
from core.models.statistical import HistoricalAverage, RidgeLinear, SeasonalNaive  # noqa: E402

OUT = ROOT / "10_Allocation" / "outputs"
(OUT / "forecasts").mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(parents=True, exist_ok=True)


def build_neural(name, bundle):
    n = bundle.n_nodes
    return {
        "LSTM": lambda: B.LSTMBaseline(n),
        "STGCN": lambda: B.STGCN(n, bundle.adj_static),
        "DCRNN": lambda: B.DCRNN(n, bundle.adj_static),
        "GraphWaveNet": lambda: B.GraphWaveNet(n, bundle.adj_static),
        "AGCRN": lambda: B.AGCRN(n),
        "AMSGCN": lambda: AMSGCN(n),
    }[name]()


def deterministic_preds(name, bundle, x_raw, tod, starts):
    if name == "HA":
        train_slots = np.arange(0, C.TRAIN_DAYS * C.SLOTS_PER_DAY)
        m = HistoricalAverage().fit(x_raw[train_slots], tod[train_slots])
        return m.predict(starts, tod)
    if name == "SeasonalNaive":
        return SeasonalNaive().fit(x_raw).predict(starts)
    if name == "Ridge":
        tr = bundle.train.starts
        return RidgeLinear().fit(x_raw, tr).predict_from(x_raw, starts, bundle.n_nodes)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=float, default=C.MAIN_GRID_STEP)
    ap.add_argument("--models", nargs="*", default=C.ALL_MODELS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--max-epochs", type=int, default=150)
    ap.add_argument("--target", choices=["all", "high_value"], default="all",
                    help="all = every order; high_value = orders at or above the "
                         "75th value percentile, matching the allocation objective")
    args = ap.parse_args()

    global OUT
    if args.target == "high_value":
        OUT = OUT.parent / "outputs_hv"
        (OUT / "forecasts").mkdir(parents=True, exist_ok=True)
        (OUT / "tables").mkdir(parents=True, exist_ok=True)

    df = D.load_orders()
    sp = D.build_spatial(df, args.grid, include_dropoff=True)
    print(f"Unified node set: {sp.n} grids at {args.grid} degrees "
          f"(pickup-only set would be {D.build_spatial(df, args.grid).n})")
    print(f"Forecast target: {args.target}")

    bundle = D.build_bundle(df, sp, biz=None, need_dynamic=True,
                            high_value_only=(args.target == "high_value"))
    print(f"Aggregate demand: train/val/test windows = "
          f"{len(bundle.train)}/{len(bundle.val)}/{len(bundle.test)}; "
          f"test mean {bundle.zero_baseline['test_mean_demand']:.3f}, "
          f"nonzero {bundle.zero_baseline['test_nonzero_share_pct']:.1f}%")

    slot_idx = np.arange(C.TOTAL_SLOTS)
    tod = (slot_idx % C.SLOTS_PER_DAY).astype(np.int64)
    test_starts = bundle.test.starts

    # ground truth for the test windows, on the original demand scale
    y_true = np.stack([bundle.raw[s + C.SEQ_LEN:s + C.SEQ_LEN + C.PRED_LEN]
                       for s in test_starts])
    np.save(OUT / "forecasts" / "y_true.npy", y_true)
    np.save(OUT / "forecasts" / "test_starts.npy", test_starts)
    np.save(OUT / "forecasts" / "node_coords.npy", sp.coords)
    np.save(OUT / "forecasts" / "demand_full.npy", bundle.raw)
    val_starts = bundle.val.starts
    np.save(OUT / "forecasts" / "val_starts.npy", val_starts)
    np.save(OUT / "forecasts" / "y_val.npy",
            np.stack([bundle.raw[s + C.SEQ_LEN:s + C.SEQ_LEN + C.PRED_LEN] for s in val_starts]))

    rows = []
    acc_path = OUT / "tables" / "unified_forecast_accuracy.csv"
    if acc_path.exists():
        rows = pd.read_csv(acc_path).to_dict("records")
    for name in args.models:
        if (OUT / "forecasts" / f"pred_{name}.npy").exists() and (OUT / "forecasts" / f"valpred_{name}.npy").exists():
            print(f"  {name:<14} forecast already on disk, skipped")
            continue
        if name in C.DETERMINISTIC_MODELS:
            p = deterministic_preds(name, bundle, bundle.raw, tod, test_starts)
            np.save(OUT / "forecasts" / f"pred_{name}.npy", p.astype(np.float32))
            vp = deterministic_preds(name, bundle, bundle.raw, tod, bundle.val.starts)
            np.save(OUT / "forecasts" / f"valpred_{name}.npy", vp.astype(np.float32))
            m = per_horizon(p.astype(np.float64), y_true.astype(np.float64))
            rows.append({"model": name, "seed": -1, **m})
            print(f"  {name:<14} MAE {m['MAE']:.4f}  MAE@t+60 {m['MAE@t+60']:.4f}")
        else:
            preds, vpreds = [], []
            for seed in args.seeds:
                C.seed_everything(seed)
                model = build_neural(name, bundle)
                m, p, _ = train_model(model, bundle, seed, max_epochs=args.max_epochs)
                preds.append(p)
                # train_model restores the best checkpoint before returning, so the
                # model in hand is the one whose test predictions were just taken.
                # Collecting on the validation split as well gives a selection
                # signal that never touches the test day.
                vp, _ = collect(model, make_loader(bundle.val, False), bundle.scaler, C.DEVICE)
                vpreds.append(vp)
                rows.append({"model": name, "seed": seed, **m})
            p = np.mean(preds, axis=0)          # seed-averaged forecast
            np.save(OUT / "forecasts" / f"pred_{name}.npy", p.astype(np.float32))
            np.save(OUT / "forecasts" / f"valpred_{name}.npy",
                    np.mean(vpreds, axis=0).astype(np.float32))
            mm = per_horizon(p.astype(np.float64), y_true.astype(np.float64))
            print(f"  {name:<14} MAE {np.mean([r['MAE'] for r in rows if r['model']==name]):.4f}"
                  f" (per-seed mean), seed-averaged forecast MAE {mm['MAE']:.4f},"
                  f" MAE@t+60 {mm['MAE@t+60']:.4f}")
            rows.append({"model": name, "seed": -2, **mm})   # -2 marks the seed-averaged forecast
        pd.DataFrame(rows).to_csv(OUT / "tables" / "unified_forecast_accuracy.csv",
                                  index=False, encoding="utf-8-sig")

    zb = bundle.zero_baseline
    pd.DataFrame([{"model": "ZeroBaseline", "seed": -1, **zb}]).to_csv(
        OUT / "tables" / "unified_zero_baseline.csv", index=False, encoding="utf-8-sig")
    print(f"\nForecasts written to {OUT / 'forecasts'}")


if __name__ == "__main__":
    main()
