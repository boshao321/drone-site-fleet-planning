"""The base scenario: at most ten aircraft per pad in a slot.

Writes, for the paper's main figures and tables,

    base_mechanism.csv  per plan: how noisy one pad's demand is, and how much
                        high-value requirement the floor and the ceiling leave
                        uncovered at fleets of 100, 200 and 300
    base_currency.csv   every lever priced in aircraft at fleets of 200 and 300,
                        against the paper's 38 demand-ranked pads and against
                        the floor's best plan (45 demand-ranked pads)
    pad_sets.csv        the 120 candidates with their coordinates, high-value
                        volume, rank, and membership of the plans the paper shows

Usage: python3 analyze_base.py [mechanism] [currency] [pads]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import core_pd as P
from analyze_caps import curve_on_usable, usable, worth

A = P.A
CAP = 10
KS = [5, 10, 15, 20, 25, 30, 38, 45, 55, 70, 90, 120]


def mechanism(st):
    rows = []
    for rule in ("proposed", "pmedian"):
        for K in KS:
            pl = st.plan(rule, K)
            truth, est = pl.truth["test"], pl.static["test"]
            nz = truth > 0
            rel = (est - truth)[nz] / truth[nz]
            rec = dict(rule=rule, K=K, tau=pl.tau, mean_req_min=float(truth.mean()),
                       share_slots_below_one_aircraft=float((truth < P.SLOT).mean() * 100),
                       static_rel_err_sd=float(rel.std(ddof=1)))
            for M in (100, 200, 300):
                x = P.allocate_point(est, M, CAP)
                xc = P.ceiling_alloc(pl.sc["test"], M, CAP)[1]
                rec[f"usable_M{M}"] = int(min(M, K * CAP))
                rec[f"floor_unmet_pct_M{M}"] = 100 * np.maximum(truth - x * P.SLOT, 0).sum() / truth.sum()
                rec[f"ceiling_unmet_pct_M{M}"] = 100 * np.maximum(truth - xc * P.SLOT, 0).sum() / truth.sum()
            rows.append(rec)
            print(f"  mechanism {rule} K={K}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(P.OUT / "base_mechanism.csv", index=False)
    return df


def currency(ref=("proposed", 45), fleets=(200, 300)):
    """Every lever priced in aircraft on the reference plan (45 demand-ranked
    pads, the floor's best plan at 300 aircraft), at two fleet sizes. A lever
    whose service no fleet on this plan can match (the floor's curve flattens
    once every pad is full) is reported as infinite."""
    g = pd.read_csv(P.OUT / "cap_grid.csv", dtype={"cap": str})
    c = pd.read_csv(P.OUT / "cap_floor_curve.csv", dtype={"cap": str})
    f = pd.read_csv(P.OUT / "flexible_c10_ref.csv")
    g, c = g[g.cap == str(CAP)], c[c.cap == str(CAP)]
    base_rule, base_K = ref
    ck = c[(c.rule == base_rule) & (c.K == base_K)].sort_values("M")
    cm, cs = curve_on_usable(ck.M, ck.floor, base_K, str(CAP))
    rows = []
    for M in fleets:
        gm = g[g.M == M]
        cell = lambda r, k, col: float(gm[(gm.rule == r) & (gm.K == k)][col].iloc[0])
        s0 = cell(base_rule, base_K, "floor")
        best = gm.loc[gm.floor.idxmax()]
        levers = {}
        if (best.rule, int(best.K)) != ref:
            levers[f"plan: best pads ({best.rule} K={int(best.K)})"] = float(best.floor)
        other = "pmedian" if base_rule == "proposed" else "proposed"
        levers["plan: same K, other siting rule"] = cell(other, base_K, "floor")
        fr = f[(f.rule == base_rule) & (f.K == base_K) & (f.M == M)]
        levers["operations: pool pads within reach"] = s0 + float(
            fr.floor_pooled.iloc[0] - fr.floor_zoned.iloc[0])
        for col, nm in (("unc_static", "dispatch: proposed rule, no forecast"),
                        ("point", "dispatch: point forecast (AMSGCN)"),
                        ("method", "dispatch: proposed rule (AMSGCN)"),
                        ("foresight", "dispatch: perfect forecast"),
                        ("ceiling", "dispatch: attainable ceiling")):
            levers[nm] = cell(base_rule, base_K, col)
        Mu = float(usable(M, base_K, str(CAP)))
        for nm, s1 in levers.items():
            w = worth(cm, cs, Mu, s1)
            rows.append(dict(M=M, baseline=f"{base_rule} K={base_K}", lever=nm,
                             floor=s0, service=s1, service_pp=s1 - s0, aircraft=w,
                             pct_of_fleet=100 * w / M if np.isfinite(w) else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(P.OUT / "base_currency.csv", index=False)
    return df


def pads(st):
    cand = st.cand.copy()
    cand["node"] = st.cand_node
    rank = {n: i + 1 for i, n in enumerate(st.rank_order)}
    cand["rank"] = cand["node"].map(rank)
    for nm, (rule, K) in {"proposed38": ("proposed", 38), "proposed45": ("proposed", 45),
                          "pmedian15": ("pmedian", 15), "pmedian45": ("pmedian", 45)}.items():
        cand[nm] = cand["node"].isin(set(st.sites(rule, K).tolist()))
    keep = ["node", "lat", "lng", "high_value_orders", "rank",
            "proposed38", "proposed45", "pmedian15", "pmedian45"]
    cand[keep].to_csv(P.OUT / "pad_sets.csv", index=False)
    return cand[keep]


if __name__ == "__main__":
    which = sys.argv[1:] or ["mechanism", "currency", "pads"]
    pd.set_option("display.width", 250)
    st = P.Study(positioning=True) if ("mechanism" in which or "pads" in which) else None
    if "pads" in which:
        print(pads(st).head())
    if "mechanism" in which:
        print(mechanism(st).round(2).to_string(index=False))
    if "currency" in which:
        print(currency().round(2).to_string(index=False))
