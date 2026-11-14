"""E16 (R1.3, R2.3): a deployment fixed for the whole day.

With a move budget of zero the fleet cannot follow the time of day. The static
deployment places the fleet once, by program P on the day mean of the historical
profile (training days only), and keeps it for all slots. It needs no forecast, no
calibration and no repositioning, so it is a second baseline next to the floor.

Outputs: e16_grid.csv (by number of sites), e16_ref.csv (reference plans, intervals)
"""
import sys, time
import numpy as np, pandas as pd
import revlib as R
P = R.P
REPO_V, REPO_H = 60.0, 1.0
t0 = time.time()
st = R.make_study()

def static_alloc(pl, ev, M):
    e = pl.static[ev]
    return np.repeat(P.allocate_point(e.mean(axis=0, keepdims=True), M, R.CAP), len(e), axis=0)

grid, ref = [], []
for K in R.KS:
    pl = st.plan("proposed", K)
    for cal, ev in (("val", "test"), ("test", "val")):
        N = st.district[ev]; books = R.hv_books(pl, ev); fold = f"{cal}->{ev}"
        sim = lambda x: R.simulate_aircraft(st, pl, x, ev, books=books, repo_speed=REPO_V, repo_handle=REPO_H)
        for M in (100, 200, 300):
            xs = static_alloc(pl, ev, M); xf = P.allocate_point(pl.static[ev], M, R.CAP)
            row = dict(K=K, fold=fold, M=M, static_agg=100 * R.served_slots(pl, xs, ev).sum() / N, floor_agg=100 * R.served_slots(pl, xf, ev).sum() / N)
            row["static_air"] = 100 * sim(xs).sum() / N; row["floor_air"] = 100 * sim(xf).sum() / N
            grid.append(row)
            if K in (38, 45) and M <= K * R.CAP:
                al = R.policy_allocs(pl, M, R.CAP, cal, ev, which=["point", "foresight"])
                ms = [R.policy_allocs(pl, M, R.CAP, cal, ev, seed=s, which=["method"])["method"] for s in range(5)]
                for mode, fn in (("aggregate", lambda x: R.served_slots(pl, x, ev)), ("aircraft_repo", sim)):
                    lev = dict(static=fn(xs), floor=fn(xf), point=fn(al["point"]), perfect=fn(al["foresight"]), method=np.mean([fn(x) for x in ms], axis=0))
                    rec = dict(K=K, fold=fold, M=M, mode=mode, **{k: 100 * v.sum() / N for k, v in lev.items()})
                    for a, b in (("static", "floor"), ("method", "static"), ("point", "static"), ("perfect", "static"), ("method", "floor")):
                        d = lev[a] - lev[b]; (lo, hi), _ = R.boot_ci(d)
                        rec[f"{a}-{b}"] = 100 * d.sum() / N; rec[f"{a}-{b}_lo"] = 100 * lo / N; rec[f"{a}-{b}_hi"] = 100 * hi / N
                    ref.append(rec)
    print(K, round(time.time() - t0), flush=True)
    pd.DataFrame(grid).to_csv(R.OUT / "e16_grid.csv", index=False); pd.DataFrame(ref).to_csv(R.OUT / "e16_ref.csv", index=False)
print("ALL_DONE")
