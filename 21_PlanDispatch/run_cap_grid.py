"""The planning grid with a limit on the aircraft one pad can hold.

A pad is a physical facility: it can launch, recover and turn round only so
many aircraft in a 15-minute slot. With no limit, as in `run_grid.py`, the
planner can stand any fleet on a handful of pads, and pads matter only through
reach and tour length. With a limit c, a fleet of M aircraft needs at least
M / c pads to fly at all, so the pad count and the fleet are tied, and a
larger fleet forces a denser network.

Every policy is scored on one common fleet grid, so that the fleet a policy
needs to reach a service target is read off curves of the same resolution.

Outputs
    outputs/cap_grid{_swap}.csv         (rule, K, cap, M) x policy service
    outputs/cap_floor_curve{_swap}.csv  the floor at 5-aircraft resolution
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import core_pd as P

KS = [5, 10, 15, 20, 25, 30, 38, 45, 55, 70, 90, 120]
MS = [25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600]
CAPS = [5, 10, 20, None]
RULES = ["pmedian", "proposed"]
FINE_M = list(range(5, 605, 5))
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
    rows, curve = [], []
    for cap in CAPS:
        ctag = "none" if cap is None else int(cap)
        for M in MS:
            rec = {"rule": rule, "K": K, "cap": ctag, "M": M, "tau": pl.tau,
                   "usable_fleet": M if cap is None else min(M, K * cap),
                   "reach_pct": 100 * pl.reach[EV] / st.district[EV]}
            for name in ("floor", "unc_static", "foresight", "ceiling"):
                rec[name] = pl.share(pl.policy(name, M, CAL, EV, cap=cap)[0], EV)
            rec["point"] = pl.share(pl.policy("point", M, CAL, EV, ec, ee, cap=cap)[0], EV)
            s, lam = pl.policy("method", M, CAL, EV, ec, ee, cap=cap)
            rec["method"], rec["lambda"] = pl.share(s, EV), lam
            rows.append(rec)
        curve += [{"rule": rule, "K": K, "cap": ctag, "M": M,
                   "floor": pl.share(pl.policy("floor", M, CAL, EV, cap=cap)[0], EV)}
                  for M in FINE_M]
    print(f"  {rule:8s} K={K:3d} done in {time.time() - t0:5.1f}s", flush=True)
    return rows, curve


def main():
    jobs = [(r, k) for r in RULES for k in KS]
    # largest plans first so the two workers finish together
    jobs.sort(key=lambda j: -j[1])
    with Pool(2) as pool:
        res = pool.map(one_plan, jobs, chunksize=1)
    grid = pd.DataFrame([r for rows, _ in res for r in rows])
    curve = pd.DataFrame([c for _, cs in res for c in cs])
    grid.to_csv(P.OUT / f"cap_grid{SUFFIX}.csv", index=False)
    curve.to_csv(P.OUT / f"cap_floor_curve{SUFFIX}.csv", index=False)
    print(f"wrote {len(grid)} grid rows and {len(curve)} floor-curve rows")


if __name__ == "__main__":
    main()
