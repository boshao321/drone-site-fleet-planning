"""Mechanism and the common currency.

1. Why service falls as zoned pads are added: for each plan, the mean
   high-value requirement a site faces in a slot, how far the time-of-day
   estimate misses it, and how much of the fleet's time the floor leaves idle.

2. Every lever priced in aircraft. For a lever that moves service from S0 to S1
   at fleet M, its worth is the number of extra aircraft the baseline plan's
   time-of-day floor would need to reach S1, read off that plan's floor curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import core_pd as P
from analyze_planning import equiv

A = P.A


def mechanism():
    st = P.Study(positioning=True)
    rows = []
    for rule in ("pmedian", "proposed"):
        for K in (5, 10, 15, 20, 25, 30, 38, 45, 55, 70, 90, 120):
            pl = st.plan(rule, K)
            truth, est = pl.truth["test"], pl.static["test"]
            nz = truth > 0
            rel = (est - truth)[nz] / truth[nz]
            rec = dict(rule=rule, K=K, tau=pl.tau,
                       mean_req_min=float(truth.mean()),
                       share_slots_below_one_aircraft=float((truth < P.SLOT).mean() * 100),
                       static_rel_err_sd=float(rel.std(ddof=1)),
                       static_abs_err_share=float(np.abs(est - truth).sum() / truth.sum() * 100))
            for M in (100, 300):
                x = A.allocate(est, M)
                cap = x * P.SLOT
                # requirement the floor leaves uncovered, and aircraft it stands
                # where the realised requirement leaves them with nothing to do
                rec[f"floor_unmet_pct_M{M}"] = 100 * np.maximum(truth - cap, 0).sum() / truth.sum()
                rec[f"floor_idle_aircraft_pct_M{M}"] = 100 * np.floor(
                    np.maximum(cap - truth, 0) / P.SLOT).sum() / x.sum()
                xc = A.attainable_ceiling(pl.sc["test"], M)[1]
                rec[f"ceiling_unmet_pct_M{M}"] = 100 * np.maximum(truth - xc * P.SLOT, 0).sum() / truth.sum()
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(P.OUT / "mechanism.csv", index=False)
    return df


def currency():
    g = pd.read_csv(P.OUT / "grid.csv")
    c = pd.read_csv(P.OUT / "floor_curve.csv")
    f = pd.read_csv(P.OUT / "flexible.csv")

    def curve(rule, K):
        ck = c[(c.rule == rule) & (c.K == K)].sort_values("M")
        return ck.M.to_numpy(float), ck.floor.to_numpy(float)

    def cell(rule, K, M, col):
        return float(g[(g.rule == rule) & (g.K == K) & (g.M == M)][col].iloc[0])

    rows = []
    for M in (100, 200, 300):
        for base_rule, base_K in (("proposed", 38), ("pmedian", 15)):
            cm, cs = curve(base_rule, base_K)
            s0 = cell(base_rule, base_K, M, "floor")

            def worth(s1):
                m = equiv(cm, cs, s1)
                return (m - M) if np.isfinite(m) else np.nan

            levers = {}
            # planning: move to the best zoned plan for this fleet under the floor
            gm = g[g.M == M]
            b = gm.loc[gm.floor.idxmax()]
            levers[f"plan: best pads ({b.rule} K={int(b.K)})"] = b.floor
            levers["plan: same K, other siting rule"] = cell(
                "pmedian" if base_rule == "proposed" else "proposed", base_K, M, "floor")
            # operating mode: pooled execution of the same floor allocation
            fr = f[(f.rule == base_rule) & (f.K == base_K) & (f.M == M)]
            if len(fr):
                levers["operations: pool pads within reach"] = s0 + float(
                    fr.floor_pooled.iloc[0] - fr.floor_zoned.iloc[0])
            # dispatch
            for col, nm in (("unc_static", "dispatch: uncertainty-aware rule, no forecast"),
                            ("point", "dispatch: point forecast (AMSGCN)"),
                            ("method", "dispatch: proposed method (AMSGCN)"),
                            ("foresight", "dispatch: perfect forecast"),
                            ("ceiling", "dispatch: attainable ceiling")):
                levers[nm] = cell(base_rule, base_K, M, col)
            for nm, s1 in levers.items():
                w = worth(s1)
                rows.append(dict(M=M, baseline=f"{base_rule} K={base_K}", lever=nm,
                                 service_pp=s1 - s0, aircraft=w,
                                 pct_of_fleet=100 * w / M if np.isfinite(w) else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(P.OUT / "currency.csv", index=False)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    m = mechanism()
    print(m.round(2).to_string(index=False))
    if "--mechanism-only" in __import__("sys").argv:
        raise SystemExit
    cu = currency()
    print(cu.round(2).to_string(index=False))
