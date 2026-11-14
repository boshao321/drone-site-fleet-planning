"""E17 (R1 follow-up): replay with order release times. Outputs e17_K_sweep.csv, e17_ref_ci.csv."""
import time
import numpy as np, pandas as pd
import revlib as R, causal as Cz
P = R.P
POL = ["floor", "point", "method", "foresight"]
MODES = {"causal_wait15": dict(wait=15), "causal_own_slot": dict(wait=15, own_slot=True)}
st = P.Study(); t0 = time.time()
rows = []
for K in R.KS:
    pl = st.plan("proposed", K)
    for cal, ev in (("val", "test"), ("test", "val")):
        N = st.district[ev]; tb = Cz.timed_books(st, pl, ev)
        for M in (200, 300):
            al = R.policy_allocs(pl, M, R.CAP, cal, ev, which=POL)
            for p in POL:
                r = dict(K=K, fold=f"{cal}->{ev}", M=M, policy=p)
                for m, kw in MODES.items():
                    r[m] = 100 * Cz.simulate_causal(st, pl, al[p], ev, tb, **kw).sum() / N
                rows.append(r)
    print(f"K={K} {time.time()-t0:.0f}s", flush=True)
pd.DataFrame(rows).to_csv(R.OUT / "e17_K_sweep.csv", index=False)
pl = st.plan("proposed", 45); rows = []
for cal, ev in (("val", "test"), ("test", "val")):
    N = st.district[ev]; tb = Cz.timed_books(st, pl, ev)
    for M in (200, 300):
        base = R.policy_allocs(pl, M, R.CAP, cal, ev, which=["floor", "point", "foresight"])
        ms = [R.policy_allocs(pl, M, R.CAP, cal, ev, seed=s, which=["method"])["method"] for s in range(5)]
        for m, kw in MODES.items():
            lev = {p: Cz.simulate_causal(st, pl, base[p], ev, tb, **kw) for p in base}
            lev["method"] = np.mean([Cz.simulate_causal(st, pl, x, ev, tb, **kw) for x in ms], axis=0)
            for a, b in (("method", "floor"), ("point", "floor"), ("foresight", "floor"), ("method", "point")):
                d = lev[a] - lev[b]; (lo, hi), _ = R.boot_ci(d)
                rows.append(dict(fold=f"{cal}->{ev}", M=M, mode=m, comparison=f"{a}-{b}", diff_pp=100 * d.sum() / N,
                                 lo=100 * lo / N, hi=100 * hi / N, a_level=100 * lev[a].sum() / N, b_level=100 * lev[b].sum() / N))
pd.DataFrame(rows).to_csv(R.OUT / "e17_ref_ci.csv", index=False)
print("DONE", round(time.time() - t0), flush=True)
