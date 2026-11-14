"""Main experiment: 4 business lines x 2 grid scales x 9 models x 5 seeds.

Deterministic baselines (HA, SeasonalNaive, Ridge) are run once, since a seed
cannot change their output. Neural models are run under every seed and reported
as mean and standard deviation, so that a difference between models can be
judged against the spread of repeated training runs of the same model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import config as C
from core import data as D
from core.engine import train_model
from core.metrics import per_horizon
from core.models import baselines as B
from core.models.amsgcn import AMSGCN
from core.models.statistical import HistoricalAverage, RidgeLinear, SeasonalNaive

TABLE_DIR = C.OUT_ROOT / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def build_neural(name, bundle):
    n = bundle.n_nodes
    if name == "LSTM":
        return B.LSTMBaseline(n)
    if name == "STGCN":
        return B.STGCN(n, bundle.adj_static)
    if name == "DCRNN":
        return B.DCRNN(n, bundle.adj_static)
    if name == "GraphWaveNet":
        return B.GraphWaveNet(n, bundle.adj_static)
    if name == "AGCRN":
        return B.AGCRN(n)
    if name == "AMSGCN":
        return AMSGCN(n)
    raise ValueError(name)


def run_deterministic(name, bundle, x_raw, tod, dow):
    tr, te = bundle.train.starts, bundle.test.starts
    y = np.stack([x_raw[s + C.SEQ_LEN:s + C.SEQ_LEN + C.PRED_LEN] for s in te])
    if name == "HA":
        train_slots = np.arange(0, C.TRAIN_DAYS * C.SLOTS_PER_DAY)
        m = HistoricalAverage().fit(x_raw[train_slots], tod[train_slots])
        p = m.predict(te, tod)
    elif name == "SeasonalNaive":
        p = SeasonalNaive().fit(x_raw).predict(te)
    elif name == "Ridge":
        m = RidgeLinear().fit(x_raw, tr)
        p = m.predict_from(x_raw, te, bundle.n_nodes)
    else:
        raise ValueError(name)
    out = per_horizon(p.astype(np.float64), y.astype(np.float64))
    out.update({"epochs": 0, "train_sec": 0.0, "params": 0, "val_MAE": float("nan")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="*", type=float, default=C.GRID_STEPS)
    ap.add_argument("--biz", nargs="*", default=C.BIZ_LINES)
    ap.add_argument("--models", nargs="*", default=C.ALL_MODELS)
    ap.add_argument("--seeds", nargs="*", type=int, default=C.SEEDS)
    ap.add_argument("--max-epochs", type=int, default=C.MAX_EPOCHS)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    print("Loading orders ...")
    df = D.load_orders()
    print(f"  {len(df):,} orders over {C.N_DAYS} days at {C.RESOLUTION_MIN}-minute resolution")

    slot_idx = np.arange(C.TOTAL_SLOTS)
    tod = (slot_idx % C.SLOTS_PER_DAY).astype(np.int64)
    dow = (slot_idx // C.SLOTS_PER_DAY).astype(np.int64)

    out_path = TABLE_DIR / f"raw_runs_{args.tag}.csv"

    # Resume support: a long run can be interrupted, so completed
    # (grid, business line, model, seed) combinations are reloaded and skipped
    # rather than recomputed.
    rows = []
    done = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        rows = prev.to_dict("records")
        done = {(round(float(r["grid_step"]), 6), r["biz"], r["model"], int(r["seed"]))
                for r in rows}
        print(f"  resuming: {len(rows)} runs already on disk")

    def flush():
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")

    for step in args.grids:
        sp = D.build_spatial(df, step)
        print(f"\n=== grid {step}° (~{step * 111:.2f} km), {sp.n} nodes shared by all business lines ===")
        need_dyn = "AMSGCN" in args.models
        for biz in args.biz:
            bundle = D.build_bundle(df, sp, biz, need_dynamic=need_dyn)
            zb = bundle.zero_baseline
            print(f"\n-- {biz} ({C.BIZ_LABELS[biz]}): train/val/test windows = "
                  f"{len(bundle.train)}/{len(bundle.val)}/{len(bundle.test)}; "
                  f"test mean demand {zb['test_mean_demand']:.3f}, "
                  f"nonzero {zb['test_nonzero_share_pct']:.1f}%, zero-baseline MAE {zb['MAE']:.4f}")
            key = round(float(step), 6)
            if (key, biz, "ZeroBaseline", -1) not in done:
                rows.append({"grid_step": step, "n_nodes": sp.n, "biz": biz,
                             "model": "ZeroBaseline", "seed": -1, **zb})
                flush()

            for name in args.models:
                if name in C.DETERMINISTIC_MODELS:
                    if (key, biz, name, -1) in done:
                        print(f"   {name:<14} already done, skipped")
                        continue
                    m = run_deterministic(name, bundle, bundle.raw, tod, dow)
                    rows.append({"grid_step": step, "n_nodes": sp.n, "biz": biz,
                                 "model": name, "seed": -1, **m})
                    flush()
                    print(f"   {name:<14} MAE {m['MAE']:.4f}  RMSE {m['RMSE']:.4f}  (deterministic)")
                else:
                    maes, ran = [], 0
                    for seed in args.seeds:
                        if (key, biz, name, seed) in done:
                            maes.append(float(next(r["MAE"] for r in rows
                                                   if (round(float(r["grid_step"]), 6), r["biz"],
                                                       r["model"], int(r["seed"])) == (key, biz, name, seed))))
                            continue
                        C.seed_everything(seed)
                        model = build_neural(name, bundle)
                        mt, _, _ = train_model(model, bundle, seed, max_epochs=args.max_epochs)
                        rows.append({"grid_step": step, "n_nodes": sp.n, "biz": biz,
                                     "model": name, "seed": seed, **mt})
                        maes.append(mt["MAE"])
                        ran += 1
                        flush()
                    tail = f" ({rows[-1]['params']:,} params, {rows[-1]['train_sec']:.0f}s/run)" if ran else " (all seeds already on disk)"
                    print(f"   {name:<14} MAE {np.mean(maes):.4f} ± {np.std(maes, ddof=1):.4f} "
                          f"over {len(maes)} seeds{tail}")

    flush()
    print(f"\nRaw runs written to {out_path}")


if __name__ == "__main__":
    main()
