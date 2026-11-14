"""The planning grid: every pad count K, fleet size M, siting rule and dispatch
rule, scored on the evaluation day against one district-wide denominator.

Outputs
    outputs/grid.csv          one row per (rule, K, M): service under each policy
    outputs/floor_curve.csv   the time-of-day floor at fine fleet resolution for
                              every plan, used to express any gain as the number
                              of extra aircraft that would buy the same service
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import core_pd as P

KS = [5, 10, 15, 20, 25, 30, 38, 45, 55, 70, 90, 120]
MS = [25, 50, 75, 100, 150, 200, 250, 300, 400, 500]
RULES = ["proposed", "pmedian"]
FINE_M = list(range(5, 805, 5))
MODEL = "AMSGCN"
CAL, EV = ("test", "val") if "--swap" in sys.argv else ("val", "test")
SUFFIX = "_swap" if "--swap" in sys.argv else ""

_ST = None


def _study():
    global _ST
    if _ST is None:
        _ST = P.Study(positioning=True)
    return _ST


def one_plan(args):
    rule, K = args
    t0 = time.time()
    st = _study()
    pl = st.plan(rule, K)
    ec, ee = pl.forecast(MODEL, CAL), pl.forecast(MODEL, EV)
    rows = []
    for M in MS:
        rec = {"rule": rule, "K": K, "M": M, "tau": pl.tau,
               "reach_pct": 100 * pl.reach[EV] / st.district[EV]}
        for name in ("floor", "unc_static", "foresight", "ceiling"):
            rec[name] = pl.share(pl.policy(name, M, CAL, EV)[0], EV)
        rec["point"] = pl.share(pl.policy("point", M, CAL, EV, ec, ee)[0], EV)
        s, lam = pl.policy("blend", M, CAL, EV, ec, ee)
        rec["blend"], rec["lambda"] = pl.share(s, EV), lam
        rec["method"] = pl.share(pl.policy("method", M, CAL, EV, ec, ee, lam=lam)[0], EV)
        rows.append(rec)
    curve = [{"rule": rule, "K": K, "M": M,
              "floor": pl.share(pl.policy("floor", M, CAL, EV)[0], EV)} for M in FINE_M]
    print(f"  {rule:8s} K={K:3d} done in {time.time() - t0:5.1f}s "
          f"(floor@300 {rows[MS.index(300)]['floor']:.2f}, method@300 {rows[MS.index(300)]['method']:.2f})",
          flush=True)
    return rows, curve


def main():
    jobs = [(r, k) for r in RULES for k in KS]
    with Pool(2) as pool:
        res = pool.map(one_plan, jobs, chunksize=1)
    grid = pd.DataFrame([r for rows, _ in res for r in rows])
    curve = pd.DataFrame([c for _, cs in res for c in cs])
    grid.to_csv(P.OUT / f"grid{SUFFIX}.csv", index=False)
    curve.to_csv(P.OUT / f"floor_curve{SUFFIX}.csv", index=False)
    print(f"wrote {len(grid)} grid rows and {len(curve)} floor-curve rows")


if __name__ == "__main__":
    main()
