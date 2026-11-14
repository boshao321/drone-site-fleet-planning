"""E1/E2 (R1.1, R1.2, R1.3, R2.3): is the aggregate aircraft-minute rule physically
attainable, and what does repositioning cost?

Every allocation of the paper is replayed with individually tracked aircraft
(revlib.simulate_aircraft): one sortie per aircraft at a time, sorties carried
across slot boundaries, and the fleet physically rearranged between slots.

Outputs
  e01_K_sweep.csv   service by number of sites, fleet and policy under four
                    evaluation rules (aggregate A4, aircraft level, aircraft
                    level with repositioning flights, aircraft level with no
                    sortie allowed past the slot end)
  e01_ref_ci.csv    paired block-bootstrap intervals at the reference plan
  e01_budget.csv    service when at most B aircraft may change site per slot
  e01_equiv.csv     worth in aircraft under the aggregate and the aircraft
                    level rule
"""
import sys, time
import numpy as np, pandas as pd
import revlib as R
P = R.P
REPO_V, REPO_H = 60.0, 1.0            # repositioning flight at cruise speed plus one minute of handling
POL = ["floor", "point", "method", "foresight"]
t0 = time.time()
st = P.Study()

def modes(pl, x, ev, books):
    agg = R.served_slots(pl, x, ev)
    veh = R.simulate_aircraft(st, pl, x, ev, books=books, detail=True)
    rep = R.simulate_aircraft(st, pl, x, ev, books=books, repo_speed=REPO_V, repo_handle=REPO_H, detail=True)
    strict = R.simulate_aircraft(st, pl, x, ev, books=books, overrun=False)
    return agg, veh, rep, strict

which = sys.argv[1:] or ["sweep", "ref", "budget", "equiv"]

if "sweep" in which:
    rows = []
    for K in R.KS:
        pl = st.plan("proposed", K)
        for cal, ev in (("val", "test"), ("test", "val")):
            N = st.district[ev]; books = R.hv_books(pl, ev)
            for M in (200, 300):
                al = R.policy_allocs(pl, M, R.CAP, cal, ev, which=POL)
                for p in POL:
                    agg, veh, rep, strict = modes(pl, al[p], ev, books)
                    rows.append(dict(K=K, fold=f"{cal}->{ev}", M=M, usable=min(M, K * R.CAP), policy=p,
                                     aggregate=100 * agg.sum() / N, aircraft=100 * veh["served"].sum() / N,
                                     aircraft_repo=100 * rep["served"].sum() / N, aircraft_strict=100 * strict.sum() / N,
                                     moves_per_slot=rep["moves"][1:].mean(), km_per_move=rep["move_km"].sum() / max(rep["moves"].sum(), 1),
                                     repo_min_per_slot=(rep["move_km"].sum() / REPO_V * 60 + REPO_H * rep["moves"].sum()) / len(agg)))
        print(f"K={K} done {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(R.OUT / "e01_K_sweep.csv", index=False)

pl = st.plan("proposed", 45)

if "ref" in which:
    rows = []
    for cal, ev in (("val", "test"), ("test", "val")):
        N = st.district[ev]; books = R.hv_books(pl, ev)
        for M in (200, 300):
            base = R.policy_allocs(pl, M, R.CAP, cal, ev, which=["floor", "point", "foresight"])
            res = {p: modes(pl, base[p], ev, books) for p in base}
            ms = [modes(pl, R.policy_allocs(pl, M, R.CAP, cal, ev, seed=s, which=["method"])["method"], ev, books) for s in range(5)]
            def per(i, r):   # per-slot served under evaluation rule i
                v = r[i]; return v["served"] if isinstance(v, dict) else v
            for i, mode in enumerate(["aggregate", "aircraft", "aircraft_repo", "aircraft_strict"]):
                meth = np.mean([per(i, m) for m in ms], axis=0)
                lev = {p: per(i, res[p]) for p in res}; lev["method"] = meth
                for a, b in (("method", "floor"), ("point", "floor"), ("foresight", "floor"), ("method", "point")):
                    d = lev[a] - lev[b]; (lo, hi), _ = R.boot_ci(d)
                    rows.append(dict(fold=f"{cal}->{ev}", M=M, mode=mode, comparison=f"{a}-{b}", diff_pp=100 * d.sum() / N,
                                     lo=100 * lo / N, hi=100 * hi / N, a_level=100 * lev[a].sum() / N, b_level=100 * lev[b].sum() / N))
    pd.DataFrame(rows).to_csv(R.OUT / "e01_ref_ci.csv", index=False)
    print("ref done", round(time.time() - t0), flush=True)

if "budget" in which:
    rows = []
    for cal, ev in (("val", "test"), ("test", "val")):
        N = st.district[ev]; books = R.hv_books(pl, ev)
        for M in (200, 300):
            for p in POL:
                g, e = R.marginal_tables(pl, p, M, R.CAP, cal, ev)
                x_free = R.policy_allocs(pl, M, R.CAP, cal, ev, which=[p])[p]
                # static deployment: one allocation for the whole day, from the day's mean estimate
                x_static = np.repeat(P.allocate_point(e.mean(axis=0, keepdims=True), M, R.CAP), len(e), axis=0)
                for B, x in [("free", x_free)] + [(b, R.allocate_move_budget(g, e, x_free, b)) for b in (60, 30, 15, 10, 5, 2)] + [("0 (static)", x_static)]:
                    rep = R.simulate_aircraft(st, pl, x, ev, books=books, repo_speed=REPO_V, repo_handle=REPO_H, detail=True)
                    rows.append(dict(fold=f"{cal}->{ev}", M=M, policy=p, budget=B, aggregate=100 * R.served_slots(pl, x, ev).sum() / N,
                                     aircraft_repo=100 * rep["served"].sum() / N, moves_per_slot=rep["moves"][1:].mean(),
                                     max_moves=int(rep["moves"][1:].max())))
            print(f"budget {ev} M={M} done {time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(R.OUT / "e01_budget.csv", index=False)

if "equiv" in which:
    rows = []
    for cal, ev in (("val", "test"), ("test", "val")):
        N = st.district[ev]; books = R.hv_books(pl, ev)
        fleets = np.arange(5, 605, 5)
        curve = {m: [] for m in ("aggregate", "aircraft_repo")}
        for m in fleets:
            x = P.allocate_point(pl.static[ev], int(m), R.CAP)
            curve["aggregate"].append(100 * R.served_slots(pl, x, ev).sum() / N)
            curve["aircraft_repo"].append(100 * R.simulate_aircraft(st, pl, x, ev, books=books, repo_speed=REPO_V, repo_handle=REPO_H).sum() / N)
        pd.DataFrame({"M": fleets, **curve}).to_csv(R.OUT / f"e01_floor_curve_{ev}.csv", index=False)
        for M in (200, 300):
            al = R.policy_allocs(pl, M, R.CAP, cal, ev, which=["point", "method", "foresight"])
            for mode in ("aggregate", "aircraft_repo"):
                phi = np.maximum.accumulate(np.array(curve[mode]))
                for p in ("point", "method", "foresight"):
                    sv = 100 * (R.served_slots(pl, al[p], ev).sum() if mode == "aggregate" else
                                R.simulate_aircraft(st, pl, al[p], ev, books=books, repo_speed=REPO_V, repo_handle=REPO_H).sum()) / N
                    base = np.interp(M, fleets, phi)
                    if sv > phi.max():
                        E = np.inf
                    else:
                        i = int(np.argmax(phi >= sv)); E = (fleets[i] if i == 0 else np.interp(sv, phi[i-1:i+1], fleets[i-1:i+1])) - M
                    rows.append(dict(fold=f"{cal}->{ev}", M=M, mode=mode, policy=p, floor=base, service=sv, gain_pp=sv - base, worth_aircraft=E))
    pd.DataFrame(rows).to_csv(R.OUT / "e01_equiv.csv", index=False)
    print("equiv done", round(time.time() - t0), flush=True)
