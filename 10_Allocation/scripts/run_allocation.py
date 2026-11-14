"""Run the allocation experiment: four policies across a sweep of fleet sizes.

Reads the K* sites produced by the siting stage and the t+60 forecasts produced
by the unified forecasting stage, both defined on the same 120-grid node set at
0.010 degrees, and reports what each policy achieves on the held-out test day.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "09_Forecasting_Rebuilt"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import config as C        # noqa: E402
from core import data as D          # noqa: E402
import allocation as A              # noqa: E402

OUT = ROOT / "10_Allocation" / "outputs"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
FC = OUT / "forecasts"

FLEET_SIZES = [10, 20, 30, 50, 75, 100, 150, 200, 300]


def load_sites(coords):
    p = ROOT / "07_NetworkBenefit_KStar_NoWeights" / "outputs" / "tables" / "selected_kstar_grids_no_weights.csv"
    sel = pd.read_csv(p)
    index = {(round(a, 6), round(b, 6)): i for i, (a, b) in enumerate(coords)}
    idx, missing = [], 0
    for a, b in zip(sel["lat"], sel["lng"]):
        k = (round(float(a), 6), round(float(b), 6))
        if k in index:
            idx.append(index[k])
        else:
            missing += 1
    if missing:
        print(f"  warning: {missing} selected sites not found in the node set")
    return np.array(sorted(set(idx))), sel


def prepare_orders(step=0.010):
    df = D.load_orders()
    df = df.copy()
    df["p_lat_snap"] = D.snap(df["pickup_latitude"], step)
    df["p_lng_snap"] = D.snap(df["pickup_longitude"], step)
    df["od_km"] = A.haversine_km(df["pickup_latitude"].to_numpy(),
                                 df["pickup_longitude"].to_numpy(),
                                 df["dropoff_latitude"].to_numpy(),
                                 df["dropoff_longitude"].to_numpy())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--radius", type=float, default=A.SERVICE_RADIUS_KM)
    ap.add_argument("--speed", type=float, default=A.V_CRUISE_KMH)
    ap.add_argument("--service-min", type=float, default=A.T_SERVICE_MIN)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--forecast-dir", default=None,
                    help="directory holding pred_*.npy; defaults to the high-value "
                         "forecasts, which are the ones whose target matches the objective")
    args = ap.parse_args()

    global FC
    if args.forecast_dir:
        FC = Path(args.forecast_dir)
    elif (OUT.parent / "outputs_hv" / "forecasts" / "demand_full.npy").exists():
        FC = OUT.parent / "outputs_hv" / "forecasts"
    print(f"Forecast source: {FC}")

    coords = np.load(FC / "node_coords.npy")
    demand_full = np.load(FC / "demand_full.npy")       # high-value demand when using outputs_hv
    test_starts = np.load(FC / "test_starts.npy")
    y_true = np.load(FC / "y_true.npy")                 # [S, 4, N]

    site_idx, sel_tbl = load_sites(coords)
    print(f"Node set {len(coords)} grids; selected sites {len(site_idx)}")

    orders = prepare_orders()
    # target slots are the t+60 targets of each test window
    target_slots = np.array([s + C.SEQ_LEN + C.PRED_LEN - 1 for s in test_starts])

    sp = D.build_spatial(D.load_orders(), 0.010, include_dropoff=True)
    sc = A.build_scenario(orders, coords, site_idx, target_slots,
                          radius_km=args.radius, v_kmh=args.speed,
                          t_service=args.service_min, grid_index=sp.index)

    grid_to_site, dmat = A.assign_grids_to_sites(coords, site_idx, args.radius)
    covered = (grid_to_site >= 0)
    print(f"Grids within {args.radius} km of a site: {covered.sum()} of {len(coords)}")
    print(f"Test-day servable orders at these sites: {sc.req_orders.sum():,.0f}; "
          f"aircraft-minutes required: {sc.req_minutes.sum():,.0f}")

    # average aircraft-minutes per order, estimated on TRAINING days only, used to
    # convert a predicted order count into a predicted aircraft-minute requirement
    train_slots = np.arange(0, C.TRAIN_DAYS * C.SLOTS_PER_DAY)
    tr_orders = orders[orders["slot"].isin(train_slots)]
    tr_orders = tr_orders[tr_orders["od_km"] <= A.MAX_SORTIE_RANGE_KM]
    tr_hv = tr_orders[tr_orders["package_price"] >= A.HIGH_VALUE_THRESHOLD]
    minutes_per_order = float(A.sortie_minutes(tr_hv["od_km"].to_numpy(),
                                               args.speed, args.service_min).mean())
    print(f"Mean aircraft-minutes per high-value sortie (training days): {minutes_per_order:.2f}")
    print(f"Test-day high-value orders at these sites: {sc.req_orders_hv.sum():,.0f}; "
          f"aircraft-minutes required: {sc.req_minutes_hv.sum():,.0f}")

    models = args.models or [p.stem.replace("pred_", "") for p in sorted(FC.glob("pred_*.npy"))]
    print(f"Forecast models available: {models}")

    rows = []
    for fleet in FLEET_SIZES:
        # policies that do not depend on a forecast model
        est = A.policy_estimates(sc, demand_full, grid_to_site, np.zeros_like(sc.req_minutes),
                                 train_slots, minutes_per_order)
        for pol in ("static", "lagged", "foresight"):
            x = A.allocate(est[pol], fleet)
            r = A.evaluate(x, sc)
            rows.append({"fleet_size": fleet, "policy": pol, "model": "-", **r})

        # the ceiling is the largest served high-value count the same fleet can
        # reach on the realised book, so it bounds every row above rather than
        # describing one more policy
        _, x_star = A.attainable_ceiling(sc, fleet)
        rows.append({"fleet_size": fleet, "policy": "ceiling", "model": "-",
                     **A.evaluate(x_star, sc)})

        for m in models:
            pred = np.load(FC / f"pred_{m}.npy")          # [S, 4, N]
            pred_t60_nodes = pred[:, -1, :]               # the t+60 horizon
            pred_sites = A.nodes_to_sites(pred_t60_nodes, grid_to_site, sc.n_sites)
            x = A.allocate(pred_sites * minutes_per_order, fleet)
            r = A.evaluate(x, sc)
            rows.append({"fleet_size": fleet, "policy": "forecast", "model": m, **r})

        done = [r for r in rows if r["fleet_size"] == fleet]
        best = max((r for r in done if r["policy"] == "forecast"),
                   key=lambda r: r["served_high_value_share_pct"], default=None)
        f = next(r for r in done if r["policy"] == "foresight")
        c = next(r for r in done if r["policy"] == "ceiling")
        s = next(r for r in done if r["policy"] == "static")
        assert c["served_high_value_share_pct"] >= max(
            r["served_high_value_share_pct"] for r in done) - 1e-9, \
            "a policy beat the ceiling, so the ceiling is not a bound"
        print(f"  fleet {fleet:>4}: static {s['served_high_value_share_pct']:.1f}%  "
              f"best forecast {best['served_high_value_share_pct']:.1f}% ({best['model']})  "
              f"foresight {f['served_high_value_share_pct']:.1f}%  "
              f"ceiling {c['served_high_value_share_pct']:.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / f"allocation_{args.tag}.csv", index=False, encoding="utf-8-sig")

    # value of the forecast: share of the static-to-ceiling gap that is closed.
    # The denominator is the attainable ceiling, not perfect foresight passed
    # through the allocation rule, so it cannot be beaten by any policy and the
    # share it reports is a share of something real.
    recs = []
    for fleet, g in df.groupby("fleet_size"):
        s = g[g.policy == "static"]["served_high_value_share_pct"].iloc[0]
        c = g[g.policy == "ceiling"]["served_high_value_share_pct"].iloc[0]
        fo = g[g.policy == "foresight"]["served_high_value_share_pct"].iloc[0]
        l = g[g.policy == "lagged"]["served_high_value_share_pct"].iloc[0]
        gap = c - s
        for _, r in g[g.policy == "forecast"].iterrows():
            recs.append({"fleet_size": fleet, "model": r["model"],
                         "static_pct": s, "lagged_pct": l,
                         "forecast_pct": r["served_high_value_share_pct"],
                         "foresight_pct": fo, "ceiling_pct": c,
                         "gain_over_static_pp": r["served_high_value_share_pct"] - s,
                         "gain_over_lagged_pp": r["served_high_value_share_pct"] - l,
                         "share_of_attainable_gap_pct": (
                             (r["served_high_value_share_pct"] - s) / gap * 100 if gap > 1e-9 else np.nan),
                         "foresight_share_of_attainable_gap_pct": (
                             (fo - s) / gap * 100 if gap > 1e-9 else np.nan)})
    vf = pd.DataFrame(recs)
    vf.to_csv(OUT / "tables" / f"value_of_forecast_{args.tag}.csv", index=False, encoding="utf-8-sig")
    print(f"\nWritten to {OUT / 'tables'}")


if __name__ == "__main__":
    main()
