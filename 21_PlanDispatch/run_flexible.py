"""Is the loss from adding pads a property of zoned operation?

In the main model every demand grid is served by its nearest pad, and the
aircraft standing at a pad serve only that pad's catchment. Under that zoned
operation a denser network splits the same demand into smaller, noisier
catchments, and aircraft committed to one of them cannot help a neighbour.

This experiment executes the same allocations two ways, with one greedy
dispatcher so that only the pooling differs:

    zoned     an order may be flown only from its nearest pad
    pooled    an order may be flown from any pad within the service radius;
              among those with enough minutes left it goes to the one with the
              shortest tour

High-value orders are taken in value order, as in the main evaluation. Only
high-value orders are counted, and they precede every other order in value
order, so the other orders cannot change the count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import core_pd as P

A = P.A
KS = [5, 10, 15, 20, 25, 30, 38, 45, 55, 70, 90, 120]
MS = [100, 300]
import sys
CAP = next((int(a.split('=')[1]) for a in sys.argv if a.startswith('--cap=')), None)
SUFFIX = f'_c{CAP}' if CAP else ''


class Executor:
    def __init__(self, st, pl, day):
        o = st.orders
        sl = st.slots[day]
        pos = {int(s): i for i, s in enumerate(sl)}
        d = o[(o["node"] >= 0) & (o["od_km"] <= A.MAX_SORTIE_RANGE_KM)
              & (o["package_price"] >= A.HIGH_VALUE_THRESHOLD) & o["slot"].isin(pos)]
        d = d.sort_values("package_price", ascending=False, kind="stable")
        self.t = d["slot"].map(pos).to_numpy()
        nodes = d["node"].to_numpy()
        site_xy = st.coords[pl.site_idx]
        g = st.coords[nodes]
        dist_grid = A.haversine_km(g[:, 0][:, None], g[:, 1][:, None],
                                   site_xy[:, 0][None, :], site_xy[:, 1][None, :])
        self.reach = dist_grid <= st.radius                       # [orders, sites]
        self.nearest = np.where(self.reach.any(axis=1),
                                np.argmin(np.where(self.reach, dist_grid, np.inf), axis=1), -1)
        # nearest-site assignment must match the main model's zoning exactly
        assert (self.nearest == pl.g2s[nodes]).all()
        d_sp = A.haversine_km(site_xy[:, 0][None, :], site_xy[:, 1][None, :],
                              d["pickup_latitude"].to_numpy()[:, None],
                              d["pickup_longitude"].to_numpy()[:, None])
        d_ds = A.haversine_km(d["dropoff_latitude"].to_numpy()[:, None],
                              d["dropoff_longitude"].to_numpy()[:, None],
                              site_xy[:, 0][None, :], site_xy[:, 1][None, :])
        self.tau = A.tour_minutes(d_sp, d["od_km"].to_numpy()[:, None], d_ds)   # [orders, sites]
        self.T = len(sl)

    def serve(self, x, pooled):
        cap = x.astype(float) * P.SLOT
        served = 0
        for i in range(len(self.t)):
            t = self.t[i]
            if pooled:
                cand = np.where(self.reach[i] & (cap[t] >= self.tau[i]))[0]
                if cand.size == 0:
                    continue
                j = cand[np.argmin(self.tau[i, cand])]
            else:
                j = self.nearest[i]
                if j < 0 or cap[t, j] < self.tau[i, j]:
                    continue
            cap[t, j] -= self.tau[i, j]
            served += 1
        return served


def main():
    st = P.Study(positioning=True)
    rows = []
    for rule in ("pmedian", "proposed"):
        for K in KS:
            pl = st.plan(rule, K)
            ex = Executor(st, pl, "test")
            ec, ee = pl.forecast("AMSGCN", "val"), pl.forecast("AMSGCN", "test")
            for M in MS:
                allocs = {"floor": P.allocate_point(pl.static["test"], M, CAP),
                          "foresight": P.allocate_point(pl.truth["test"], M, CAP),
                          "ceiling_alloc": P.ceiling_alloc(pl.sc["test"], M, CAP)[1]}
                lam = pl.choose_lambda(ec, M, "val", CAP)
                bc = (1 - lam) * pl.static["val"] + lam * ec
                be = (1 - lam) * pl.static["test"] + lam * ee
                allocs["method"] = P.allocate_expected(
                    P.residual_sampler(bc, pl.truth["val"])(be), M, CAP)
                rec = dict(rule=rule, K=K, M=M)
                for name, x in allocs.items():
                    for mode in ("zoned", "pooled"):
                        rec[f"{name}_{mode}"] = 100 * ex.serve(x, mode == "pooled") / st.district["test"]
                rows.append(rec)
            print(f"  {rule} K={K} done: floor zoned/pooled @300 "
                  f"{rows[-1]['floor_zoned']:.2f}/{rows[-1]['floor_pooled']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(P.OUT / f"flexible{SUFFIX}.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
