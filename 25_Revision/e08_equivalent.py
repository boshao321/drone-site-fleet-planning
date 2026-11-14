"""E8 (R1.13, R2.4, R3.4): how firm is the worth in aircraft?

The worth of a dispatch rule is the number of extra aircraft the floor would
need to match it, read from the floor's fleet curve. Here the curve and the
rule are resampled together (paired block bootstrap over one-hour blocks of the
scored day), which gives an interval for the worth and for the ratio between
the proposed rule and the point forecast. The reading is repeated with fleet
increments of 1, 5, 10 and 25 aircraft and with step, linear and monotone
cubic interpolation.

Outputs
  e08_intervals.csv     worth in aircraft with 95% intervals, both evaluation rules
  e08_interpolation.csv worth by fleet increment and interpolation rule
"""
import sys, time
import numpy as np, pandas as pd
from scipy.interpolate import PchipInterpolator
import revlib as R
P = R.P
B, BLOCK = 5000, 4
REPO_V, REPO_H = 60.0, 1.0
t0 = time.time()
st = R.make_study()

def worth(fleets, phi, sv, M, how="linear"):
    """aircraft the floor needs beyond M to reach service sv; phi made monotone."""
    phi = np.maximum.accumulate(phi)
    if sv > phi[-1]:
        return np.inf
    if sv <= phi[0]:
        return float(fleets[0] - M)
    i = int(np.argmax(phi >= sv))
    if how == "step":
        return float(fleets[i] - M)
    if how == "linear":
        return float(np.interp(sv, phi[i - 1:i + 1], fleets[i - 1:i + 1]) - M)
    # monotone cubic through the curve, inverted on a fine grid
    f = PchipInterpolator(fleets, phi)
    g = np.linspace(fleets[i - 1], fleets[i], 501)
    return float(g[int(np.argmax(f(g) >= sv))] - M)

rows_i, rows_p, rows_s = [], [], []
for K in (45, 38):
    pl = st.plan("proposed", K)
    for cal, ev in (("val", "test"), ("test", "val")):
        N = st.district[ev]; books = R.hv_books(pl, ev); fold = f"{cal}->{ev}"
        T = len(st.slots[ev])
        blocks = [np.arange(i, min(i + BLOCK, T)) for i in range(0, T, BLOCK)]
        rng = np.random.default_rng(7)
        draws = rng.integers(0, len(blocks), size=(B, len(blocks)))
        bsum = lambda v: np.array([v[..., b].sum(axis=-1) for b in blocks])      # [n_blocks, ...]
        for mode in ("aggregate", "aircraft_repo"):
            if mode == "aircraft_repo" and K != 45:
                continue
            ev_fn = (lambda x: R.served_slots(pl, x, ev)) if mode == "aggregate" else \
                    (lambda x: R.simulate_aircraft(st, pl, x, ev, books=books, repo_speed=REPO_V, repo_handle=REPO_H))
            step = 1 if mode == "aggregate" else 5
            fleets = np.arange(step, min(K * R.CAP, 800) + 1, step)
            F = np.stack([ev_fn(P.allocate_point(pl.static[ev], int(m), R.CAP)) for m in fleets])   # [n_fleets, T]
            Fb = bsum(F)                                                                           # [n_blocks, n_fleets]
            for M in (100, 200, 300):
                if M > K * R.CAP:
                    continue
                al = R.policy_allocs(pl, M, R.CAP, cal, ev, which=["point", "foresight"])
                lev = {p: ev_fn(al[p]) for p in al}
                per_seed = [ev_fn(R.policy_allocs(pl, M, R.CAP, cal, ev, seed=s, which=["method"])["method"]) for s in range(5)]
                lev["method"] = per_seed[0]                       # the sampling seed of the paper's planning grid
                lev["method_mean5"] = np.mean(per_seed, axis=0)
                for s_, v_ in enumerate(per_seed):
                    rows_s.append(dict(K=K, fold=fold, mode=mode, M=M, seed=s_, service=100 * v_.sum() / N,
                                       worth=worth(fleets, 100 * F.sum(axis=1) / N, 100 * v_.sum() / N, M)))
                tot = 100 * F.sum(axis=1) / N
                point_est = {p: worth(fleets, tot, 100 * lev[p].sum() / N, M) for p in lev}
                # paired bootstrap: the floor curve and the rule are resampled on the same blocks
                bw = {p: np.empty(B) for p in lev}
                Lb = {p: bsum(lev[p]) for p in lev}
                for b in range(B):
                    curve = 100 * Fb[draws[b]].sum(axis=0) / N
                    for p in lev:
                        bw[p][b] = worth(fleets, curve, 100 * Lb[p][draws[b]].sum() / N, M)
                for p in lev:
                    lo, hi = np.percentile(bw[p], [2.5, 97.5])
                    rows_i.append(dict(K=K, fold=fold, mode=mode, M=M, policy=p, service=100 * lev[p].sum() / N,
                                       floor=float(np.interp(M, fleets, tot)), worth=point_est[p], lo=lo, hi=hi,
                                       share_inf=float(np.isinf(bw[p]).mean())))
                ok = (bw["point"] > 0) & np.isfinite(bw["method"]) & np.isfinite(bw["point"])
                ratio = bw["method"][ok] / bw["point"][ok]
                d = bw["method"] - bw["point"]
                rows_i.append(dict(K=K, fold=fold, mode=mode, M=M, policy="method/point ratio",
                                   worth=point_est["method"] / point_est["point"] if point_est["point"] > 0 else np.nan,
                                   lo=np.percentile(ratio, 2.5) if ok.sum() > 100 else np.nan,
                                   hi=np.percentile(ratio, 97.5) if ok.sum() > 100 else np.nan, share_inf=1 - ok.mean()))
                rows_i.append(dict(K=K, fold=fold, mode=mode, M=M, policy="method-point aircraft", worth=point_est["method"] - point_est["point"],
                                   lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), share_inf=float((d <= 0).mean())))
                if mode == "aggregate":
                    for inc in (1, 5, 10, 25):
                        sel = (fleets % inc == 0)
                        for how in ("step", "linear", "pchip"):
                            rows_p.append(dict(K=K, fold=fold, M=M, increment=inc, interpolation=how,
                                               **{p: worth(fleets[sel], tot[sel], 100 * lev[p].sum() / N, M, how) for p in ("point", "method", "foresight")}))
            print(f"K={K} {fold} {mode} done {time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(rows_i).to_csv(R.OUT / "e08_intervals.csv", index=False)
            pd.DataFrame(rows_p).to_csv(R.OUT / "e08_interpolation.csv", index=False)
            pd.DataFrame(rows_s).to_csv(R.OUT / "e08_seeds.csv", index=False)
print("ALL_DONE")
