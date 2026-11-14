"""E3/E4 (R1.4, R1.5): closed-tour range limit and operating parameters.

One setting is changed at a time from the base case (v = 60 km/h, t_s = 4 min,
rho = 3 km, OD limit 8 km, no limit on the closed tour). For every setting the
whole planning grid is recomputed: site sets by the same ranking, tours,
historical mean, allocations of every policy, and the floor's fleet curve.

Outputs
  e03_grid.csv      service by setting, fold, number of sites, fleet and policy
  e03_summary.csv   per setting: best number of sites of each policy at 200 and
                    300 aircraft, gain of the proposed rule over the floor and
                    over the point forecast at 45 sites with paired block
                    bootstrap intervals, and worth in aircraft
  e03_tours.csv     distribution of closed-tour length of the high value orders
"""
import sys, time
import numpy as np, pandas as pd
from pathlib import Path
import revlib as R
P, A = R.P, R.A
HERE = Path(__file__).resolve().parent
POL = ["floor", "point", "method", "foresight", "ceiling"]
FLEETS = np.arange(5, 805, 5)
SETTINGS = [("base", {}),
            ("tour<=10km", dict(tour_cap=10.0)), ("tour<=12km", dict(tour_cap=12.0)),
            ("tour<=15km", dict(tour_cap=15.0)), ("tour<=20km", dict(tour_cap=20.0)),
            ("v=40", dict(v=40.0)), ("v=80", dict(v=80.0)),
            ("ts=2", dict(ts=2.0)), ("ts=6", dict(ts=6.0)),
            ("rho=2", dict(rho=2.0)), ("rho=4", dict(rho=4.0)),
            ("od<=5", dict(od=5.0)), ("od<=10", dict(od=10.0)),
            # E7 (R1.12): other definitions of a high value order. Forecasts retrained on the new target
            # (folds/THR*), sites ranked again by the endpoints of the orders above the new threshold.
            ("thr=P50", dict(fc=HERE / "folds/THR50/forecasts", thr=29.40, rank_days=range(1, 7))),
            ("thr=P90", dict(fc=HERE / "folds/THR90/forecasts", thr=92.70, rank_days=range(1, 7))),
            ("thr=all", dict(fc=HERE / "folds/THR0/forecasts", thr=0.0, rank_days=range(1, 7))),
            # R1.5: combined settings, all adverse or all favorable at once
            ("adverse", dict(v=40.0, ts=6.0, tour_cap=10.0)), ("favorable", dict(v=80.0, ts=2.0))]
only = sys.argv[1:]
gp, sp_, tp = R.OUT / "e03_grid.csv", R.OUT / "e03_summary.csv", R.OUT / "e03_tours.csv"
grid = pd.read_csv(gp).to_dict("records") if gp.exists() else []
summ = pd.read_csv(sp_).to_dict("records") if sp_.exists() else []
tours = pd.read_csv(tp).to_dict("records") if tp.exists() else []
done = {r["setting"] for r in summ}
t0 = time.time()
for name, kw in SETTINGS:
    if name in done or (only and name not in only):
        continue
    st = R.make_study(**kw)
    v, ts = R.ENV["v"], R.ENV["ts"]
    for cal, ev in (("val", "test"), ("test", "val")):
        N = st.district[ev]; fold = f"{cal}->{ev}"
        for K in R.KS:
            pl = st.plan("proposed", K)
            for M in (200, 300):
                al = R.policy_allocs(pl, M, R.CAP, cal, ev, which=POL)
                for p in POL:
                    grid.append(dict(setting=name, fold=fold, K=K, M=M, policy=p, service=100 * R.served_slots(pl, al[p], ev).sum() / N,
                                     lam=al["_lambda"] if p == "method" else np.nan, tau=pl.tau, reach=100 * pl.reach[ev] / N, N=N))
        pl = st.plan("proposed", 45)
        g = pd.DataFrame([r for r in grid if r["setting"] == name and r["fold"] == fold])
        curve = 100 * R.floor_curve(pl, ev, FLEETS) / N
        for M in (200, 300):
            row = dict(setting=name, fold=fold, M=M, N=N, tau45=pl.tau, reach45=100 * pl.reach[ev] / N)
            for p in POL:
                gg = g[(g.M == M) & (g.policy == p)].sort_values("K")
                row[f"bestK_{p}"] = int(gg.K.iloc[int(np.argmax(gg.service.to_numpy()))])
                row[f"svc45_{p}"] = float(gg[gg.K == 45].service.iloc[0])
            base = R.policy_allocs(pl, M, R.CAP, cal, ev, which=["floor", "point", "foresight"])
            lev = {p: R.served_slots(pl, base[p], ev) for p in base}
            ms = [R.policy_allocs(pl, M, R.CAP, cal, ev, seed=s, which=["method"]) for s in range(5)]
            lev["method"] = np.mean([R.served_slots(pl, m["method"], ev) for m in ms], axis=0)
            row["lam"] = ms[0]["_lambda"]
            for a, b in (("method", "floor"), ("method", "point"), ("point", "floor")):
                d = lev[a] - lev[b]; (lo, hi), _ = R.boot_ci(d)
                row[f"{a}-{b}"] = 100 * d.sum() / N; row[f"{a}-{b}_lo"] = 100 * lo / N; row[f"{a}-{b}_hi"] = 100 * hi / N
            for p in ("point", "method", "foresight"):
                row[f"worth_{p}"] = R.equiv_aircraft(FLEETS, curve, 100 * lev[p].sum() / N, M)
            summ.append(row)
        # closed-tour length of the high value orders the 45 sites reach on the scored day
        tk = np.concatenate([a[a[:, 1] >= A.HIGH_VALUE_THRESHOLD, 0] for a in pl.sc[ev].order_book.values()])
        km = (tk - ts) * v / 60.0
        tours.append(dict(setting=name, fold=fold, n=len(km), mean=km.mean(), p50=np.percentile(km, 50), p90=np.percentile(km, 90),
                          p99=np.percentile(km, 99), max=km.max(), share_gt10=100 * (km > 10).mean(), share_gt12=100 * (km > 12).mean(),
                          share_gt15=100 * (km > 15).mean(), mean_min=tk.mean(), p99_min=np.percentile(tk, 99), max_min=tk.max()))
    pd.DataFrame(grid).to_csv(gp, index=False); pd.DataFrame(summ).to_csv(sp_, index=False); pd.DataFrame(tours).to_csv(tp, index=False)
    print(f"{name} done {time.time()-t0:.0f}s", flush=True)
print("ALL_DONE", flush=True)
