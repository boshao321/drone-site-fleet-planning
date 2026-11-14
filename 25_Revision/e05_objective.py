"""E5 (R1.9): aircraft-minute objective against a served-order objective.

The rule of the paper places aircraft to maximise the expected aircraft minutes
that are used, sum_s E[min(15 x_s, R_s)]. Service is counted in orders. Three
allocations are compared on the same predictive samples, same lambda:

  minutes    the paper's rule
  weighted   marginal minutes divided by the site's mean tour, i.e. expected
             orders per aircraft when every tour at a site had the mean length
  orders     expected number of high value orders served. Each predictive
             sample of a site's minutes is filled with tours drawn from that
             site's training-day tours, the orders that fit into 15 k minutes
             are counted for k = 0..c, and the fleet is placed by an exact
             dynamic programme over sites

With a perfect forecast the same contrast is the perfect-forecast policy (minutes
on the realised requirement) against the ceiling (orders on the realised book).

Output: e05_objective.csv
"""
import sys, time
import numpy as np, pandas as pd
import revlib as R
P, A = R.P, R.A
SLOT, L = P.SLOT, 40
REPO_V, REPO_H = 60.0, 1.0
t0 = time.time()
st = R.make_study()

def tour_pools(pl):
    tr = R._build_scenario(st.orders, st.coords, pl.site_idx, st.train, radius_km=st.radius, grid_index=st.sp.index, positioning=True)
    pools = [[] for _ in range(pl.S)]
    for (t, j), arr in tr.order_book.items():
        pools[j].append(arr[arr[:, 1] >= A.HIGH_VALUE_THRESHOLD, 0])
    pools = [np.concatenate(p) if len(p) else np.zeros(0) for p in pools]
    allp = np.concatenate(pools)
    return [p if len(p) >= 20 else allp for p in pools], np.array([p.mean() if len(p) >= 20 else allp.mean() for p in pools])

def dp_alloc(F, M, e):
    """max sum_s F[s, x_s]  s.t. sum x_s <= M; F[s, 0..c]. Spare aircraft by D'Hondt."""
    S, c1 = F.shape
    dp = np.zeros(M + 1); args = []
    for s in range(S):
        new = dp.copy(); arg = np.zeros(M + 1, dtype=int)
        for k in range(1, c1):
            if F[s, k] <= F[s, k - 1] + 1e-12 and k > 1:
                pass
            cand = dp[:M + 1 - k] + F[s, k]
            better = cand > new[k:] + 1e-12
            new[k:][better] = cand[better]; arg[k:][better] = k
        dp = new; args.append(arg)
    x = np.zeros(S, dtype=int); m = M
    for s in range(S - 1, -1, -1):
        x[s] = args[s][m]; m -= x[s]
    r = M - int(x.sum())
    if r > 0:
        P._dhondt_fill(x, e, r, c1 - 1)
    return x

def allocs(pl, M, cal, ev, model, seed, pools, taubar):
    cap = R.CAP
    ec, ee = pl.forecast(model, cal), pl.forecast(model, ev)
    lam = pl.choose_lambda(ec, M, cal, cap)
    bc = (1 - lam) * pl.static[cal] + lam * ec; be = (1 - lam) * pl.static[ev] + lam * ee
    smp = P.residual_sampler(bc, pl.truth[cal], seed=seed)(be)                # [T, S, N]
    T, S, N = smp.shape
    out = {"minutes": P.allocate_expected(smp, M, cap)}
    k = np.arange(cap)
    g = np.stack([np.clip(smp - SLOT * kk, 0.0, SLOT).mean(axis=2) for kk in k], axis=2)      # [T, S, cap]
    gw = g / taubar[None, :, None]
    xw = np.zeros((T, S), dtype=int); Mu = min(M, S * cap)
    unmet = smp.mean(axis=2)[:, :, None] - SLOT * k[None, None, :]
    for t in range(T):
        flat = gw[t].ravel(); top = np.lexsort((-unmet[t].ravel(), -flat))[:Mu]; top = top[flat[top] > 0]
        np.add.at(xw[t], top // cap, 1)
        r = Mu - int(xw[t].sum())
        if r > 0: P._dhondt_fill(xw[t], smp[t].mean(axis=1), r, cap)
    out["weighted"] = xw
    rng = np.random.default_rng(1000 + seed)
    F = np.zeros((T, S, cap + 1))
    for s in range(S):
        tours = rng.choice(pools[s], size=(T, N, L)); cum = np.cumsum(tours, axis=2)
        for kk in range(1, cap + 1):
            lim = np.minimum(SLOT * kk, smp[:, s, :])[..., None]
            F[:, s, kk] = (cum <= lim).sum(axis=2).mean(axis=1)
    xo = np.zeros((T, S), dtype=int)
    for t in range(T):
        xo[t] = dp_alloc(F[t], Mu, smp[t].mean(axis=1))
    out["orders"] = xo
    return out, lam

rows = []
for K in (45, 38):
    pl = st.plan("proposed", K); pools, taubar = tour_pools(pl)
    for cal, ev in (("val", "test"), ("test", "val")):
        N = st.district[ev]; books = R.hv_books(pl, ev); fold = f"{cal}->{ev}"
        sim = lambda x: R.simulate_aircraft(st, pl, x, ev, books=books, repo_speed=REPO_V, repo_handle=REPO_H)
        for M in (100, 200, 300):
            if M > K * R.CAP: continue
            res = {}
            for seed in range(5):
                al, lam = allocs(pl, M, cal, ev, "AMSGCN", seed, pools, taubar)
                for nm, x in al.items():
                    used = np.minimum(SLOT * x, pl.truth[ev]).sum() / (SLOT * x.sum()) * 100
                    res.setdefault(nm, []).append((R.served_slots(pl, x, ev), sim(x), used))
            lev = {nm: (np.mean([r[0] for r in v], axis=0), np.mean([r[1] for r in v], axis=0), np.mean([r[2] for r in v])) for nm, v in res.items()}
            xf = P.allocate_point(pl.static[ev], M, R.CAP); xp = P.allocate_point(pl.truth[ev], M, R.CAP); xc = P.ceiling_alloc(pl.sc[ev], M, R.CAP)[1]
            for nm, x in (("floor", xf), ("perfect (minutes)", xp), ("ceiling (orders)", xc)):
                lev[nm] = (R.served_slots(pl, x, ev), sim(x), np.minimum(SLOT * x, pl.truth[ev]).sum() / (SLOT * x.sum()) * 100)
            for nm, (a, v, u) in lev.items():
                ref = "perfect (minutes)" if nm.startswith("ceiling") else "minutes"
                d = a - lev[ref][0]; (lo, hi), _ = R.boot_ci(d)
                d2 = v - lev[ref][1]; (lo2, hi2), _ = R.boot_ci(d2)
                rows.append(dict(K=K, fold=fold, M=M, lam=lam, objective=nm, service=100 * a.sum() / N, service_aircraft=100 * v.sum() / N, minutes_used_pct=u,
                                 vs_ref=ref, diff=100 * d.sum() / N, lo=100 * lo / N, hi=100 * hi / N, diff_aircraft=100 * d2.sum() / N, lo_a=100 * lo2 / N, hi_a=100 * hi2 / N))
            print(f"K={K} {fold} M={M} done {time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(R.OUT / "e05_objective.csv", index=False)
print("ALL_DONE")
