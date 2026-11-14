"""Sanity checks on every output of the planning-and-dispatch study.

    1  no policy beats the attainable ceiling, in any grid or fold
    2  floor and ceiling never fall when the fleet grows
    3  perfect foresight never beats the ceiling; the floor never beats foresight
       by more than rounding (foresight is an exact optimum of the minute program,
       not of served orders, so it can in principle lose; we report if it does)
    4  the capped grid with no limit reproduces the uncapped grid exactly
    5  the historical-average forecast reproduces the floor (it is the same
       estimate), which checks the forecast-to-site conversion end to end
    6  a pad limit never raises the attainable ceiling
"""
import numpy as np
import pandas as pd

import core_pd as P

O = P.OUT
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)
    print(("ok   " if cond else "FAIL ") + msg)


POL = ["floor", "unc_static", "point", "method", "foresight"]
for f in ("grid", "grid_swap", "cap_grid", "cap_grid_swap"):
    p = O / f"{f}.csv"
    if not p.exists():
        print("skip", f)
        continue
    g = pd.read_csv(p)
    keys = ["rule", "K"] + (["cap"] if "cap" in g else [])
    worst = max(float((g[c] - g.ceiling).max()) for c in POL)
    check(worst <= 1e-9, f"{f}: no policy above the ceiling (max excess {worst:.2e})")
    dec = 0.0
    for _, d in g.groupby(keys):
        d = d.sort_values("M")
        for c in ("floor", "ceiling"):
            dec = min(dec, float(np.diff(d[c].to_numpy()).min()))
    check(dec >= -1e-9, f"{f}: floor and ceiling non-decreasing in M (worst step {dec:.3f})")
    fl = float((g.floor - g.foresight).max())
    print(f"     {f}: largest floor-over-foresight excess {fl:.3f} pp")

g = pd.read_csv(O / "grid.csv")
c = pd.read_csv(O / "cap_grid.csv")
c = c[c.cap.astype(str) == "none"]
m = g.merge(c, on=["rule", "K", "M"], suffixes=("", "_c"))
# grid.csv predates the expected-gain tie-break fix of 2026-10-08 and is
# superseded by cap_grid.csv with cap "none"; only the policies that do not use
# the expected-gain allocation can still be compared with it.
for col in ("floor", "point", "foresight", "ceiling"):
    d = float((m[col] - m[f"{col}_c"]).abs().max())
    check(d < 1e-9, f"cap grid without a limit reproduces grid.csv for {col} ({len(m)} cells, max diff {d:.2e})")
print("     grid.csv predates the tie-break fix; unc_static and method are compared on cap_grid only")

for tag in ("proposed38", "pmedian15"):
    d = pd.read_csv(O / f"d1_models_{tag}.csv")
    h = d[d.model == "HA"]
    # the forecasting stage computes its historical average in its own
    # normalised units, so the two differ by a handful of orders at most
    check(float(h.point_vs_floor_pp.abs().max()) < 0.05,
          f"{tag}: historical-average point forecast within 0.05 pp of the floor in every cell "
          f"(max {float(h.point_vs_floor_pp.abs().max()):.3f})")

c = pd.read_csv(O / "cap_grid.csv")
c["cap"] = c.cap.astype(str)
base = c[c.cap == "none"].set_index(["rule", "K", "M"])
for cap in ("5", "10", "20"):
    d = c[c.cap == cap].set_index(["rule", "K", "M"])
    exc = float((d["ceiling"] - base.loc[d.index, "ceiling"]).max())
    check(exc <= 1e-9, f"limit {cap}: the ceiling never rises under a pad limit (max {exc:.2e})")
    # the estimate-driven rules are not optima of served orders, so a limit that
    # forces aircraft apart can help them a little; report how much
    for q in ("floor", "point", "method"):
        print(f"     limit {cap}: largest gain of {q} from the limit "
              f"{float((d[q] - base.loc[d.index, q]).max()):.3f} pp")

print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED")
