"""E12 (R1.11, R2.7, R3.6): what pooled service asks of the operation.

The paper's pooled service lets an order fly from any site within the service
radius of its pickup cell. Here the same executor also records

  cross      share of served orders flown from a site other than the nearest
  minutes    mean tour of the served orders, zoned and pooled
  penalty    pooled service when each cross-site sortie costs delta extra
             minutes (handover between site operators, deconfliction)
  radius     pooled service when a second site may be used only within a
             smaller radius, or only the two nearest sites
  bound      an upper bound on pooled service for the fleet: the linear
             relaxation in which aircraft minutes are divisible, each site
             offers at most 15 c minutes per slot and the fleet 15 M

Outputs: e12_pooling.csv, e12_bound.csv
"""
import sys, time
import numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack
import revlib as R
P, A = R.P, R.A
SLOT = P.SLOT
t0 = time.time()
st = R.make_study()


class Exec:
    def __init__(self, pl, day):
        o = st.orders; sl = st.slots[day]; pos = {int(s): i for i, s in enumerate(sl)}
        d = o[(o["node"] >= 0) & (o["od_km"] <= A.MAX_SORTIE_RANGE_KM) & (o["package_price"] >= A.HIGH_VALUE_THRESHOLD) & o["slot"].isin(pos)]
        d = d.sort_values("package_price", ascending=False, kind="stable")
        self.t = d["slot"].map(pos).to_numpy(); nodes = d["node"].to_numpy()
        xy = st.coords[pl.site_idx]; g = st.coords[nodes]
        self.dist = A.haversine_km(g[:, 0][:, None], g[:, 1][:, None], xy[:, 0][None, :], xy[:, 1][None, :])
        self.reach = self.dist <= st.radius
        self.nearest = np.where(self.reach.any(axis=1), np.argmin(np.where(self.reach, self.dist, np.inf), axis=1), -1)
        assert (self.nearest == pl.g2s[nodes]).all()
        d_sp = A.haversine_km(xy[:, 0][None, :], xy[:, 1][None, :], d["pickup_latitude"].to_numpy()[:, None], d["pickup_longitude"].to_numpy()[:, None])
        d_ds = A.haversine_km(d["dropoff_latitude"].to_numpy()[:, None], d["dropoff_longitude"].to_numpy()[:, None], xy[:, 0][None, :], xy[:, 1][None, :])
        self.tau = A.tour_minutes(d_sp, d["od_km"].to_numpy()[:, None], d_ds, R.ENV["v"], R.ENV["ts"])
        self.T = len(sl); self.n = len(self.t)
        self.rank = np.argsort(np.argsort(np.where(self.reach, self.dist, np.inf), axis=1), axis=1)   # 0 = nearest

    def serve(self, x, pooled, delta=0.0, radius=None, two_nearest=False):
        cap = x.astype(float) * SLOT
        served = cross = 0; minutes = 0.0; zmin = 0.0
        for i in range(self.n):
            t = self.t[i]; jn = self.nearest[i]
            if jn < 0: continue
            if pooled:
                ok = self.reach[i].copy()
                if radius is not None: ok &= (self.dist[i] <= radius)
                if two_nearest: ok &= (self.rank[i] <= 1)
                ok[jn] = True
                tau = self.tau[i] + delta; tau[jn] -= delta
                cand = np.where(ok & (cap[t] >= tau))[0]
                if cand.size == 0: continue
                j = cand[np.argmin(tau[cand])]; used = tau[j]
            else:
                j = jn; used = self.tau[i, j]
                if cap[t, j] < used: continue
            cap[t, j] -= used; served += 1; minutes += self.tau[i, j]; zmin += self.tau[i, jn]; cross += int(j != jn)
        return dict(served=served, cross=cross, minutes=minutes, nearest_minutes=zmin)

    def bound(self, M, cap_site):
        """Linear relaxation of pooled service, slot by slot."""
        tot = 0.0; S = self.reach.shape[1]
        for t in range(self.T):
            idx = np.where((self.t == t) & (self.nearest >= 0))[0]
            if len(idx) == 0: continue
            oo, ss = np.where(self.reach[idx]); nv = len(oo); tau = self.tau[idx][oo, ss]
            A1 = coo_matrix((np.ones(nv), (oo, np.arange(nv))), shape=(len(idx), nv))
            A2 = coo_matrix((tau, (ss, np.arange(nv))), shape=(S, nv))
            A3 = coo_matrix((tau, (np.zeros(nv, dtype=int), np.arange(nv))), shape=(1, nv))
            b = np.concatenate([np.ones(len(idx)), np.full(S, SLOT * cap_site), [SLOT * min(M, S * cap_site)]])
            r = linprog(-np.ones(nv), A_ub=vstack([A1, A2, A3]).tocsr(), b_ub=b, bounds=(0, 1), method="highs")
            tot += -r.fun
        return tot


rows, brow = [], []
for K in R.KS:
    pl = st.plan("proposed", K)
    for cal, ev in ((("val", "test"), ("test", "val")) if K in (38, 45) else (("val", "test"),)):
        ex = Exec(pl, ev); N = st.district[ev]; fold = f"{cal}->{ev}"
        for M in (100, 200, 300):
            al = R.policy_allocs(pl, M, R.CAP, cal, ev, which=["floor", "method", "foresight", "ceiling"])
            for p in ("floor", "method", "foresight", "ceiling"):
                z = ex.serve(al[p], False); q = ex.serve(al[p], True)
                rec = dict(K=K, fold=fold, M=M, policy=p, zoned=100 * z["served"] / N, pooled=100 * q["served"] / N,
                           cross_share=100 * q["cross"] / max(q["served"], 1), tour_zoned=z["minutes"] / max(z["served"], 1),
                           tour_pooled=q["minutes"] / max(q["served"], 1),
                           extra_min_per_cross=(q["minutes"] - q["nearest_minutes"]) / max(q["cross"], 1))
                if p in ("floor", "method") and (K in (30, 38, 45, 70, 120)):
                    for dl in (1, 2, 5):
                        rec[f"pooled_delta{dl}"] = 100 * ex.serve(al[p], True, delta=dl)["served"] / N
                    for rr in (1.5, 2.0):
                        rec[f"pooled_r{rr}"] = 100 * ex.serve(al[p], True, radius=rr)["served"] / N
                    rec["pooled_two_nearest"] = 100 * ex.serve(al[p], True, two_nearest=True)["served"] / N
                rows.append(rec)
            if K in (15, 30, 38, 45, 70, 120) and M in (200, 300):
                ub = ex.bound(M, R.CAP)
                brow.append(dict(K=K, fold=fold, M=M, coverage=100 * pl.reach[ev] / N, zoned_ceiling=100 * R.served_slots(pl, al["ceiling"], ev).sum() / N,
                                 pooled_best_executed=max(r["pooled"] for r in rows if r["K"] == K and r["fold"] == fold and r["M"] == M),
                                 pooled_upper_bound=100 * ub / N))
        print(f"K={K} {fold} done {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(R.OUT / "e12_pooling.csv", index=False); pd.DataFrame(brow).to_csv(R.OUT / "e12_bound.csv", index=False)
print("ALL_DONE")
