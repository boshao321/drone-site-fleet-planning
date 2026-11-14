"""Planning analysis on the grid: one currency for pads, aircraft and dispatch.

Reads outputs/grid.csv and outputs/floor_curve.csv and writes

    best_plans.csv        the best (rule, K) for every fleet size under each
                          dispatch rule, and the service it delivers
    equivalent_aircraft.csv
                          every policy at every plan expressed as the number of
                          extra aircraft the time-of-day floor of the SAME plan
                          would need to deliver the same service
    iso_service.csv       the fleet each plan needs to reach a service target,
                          under each dispatch rule
    cost_optimal.csv      the cheapest plan (K, M) for each service target and
                          pad-to-aircraft cost ratio, under each dispatch rule
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import core_pd as P

POLICIES = ["floor", "unc_static", "point", "blend", "method", "foresight", "ceiling"]
TARGETS = [40, 50, 60, 70, 75, 80, 85, 90]
RATIOS = [0.25, 0.5, 1, 2, 5, 10, 20]


def load():
    g = pd.read_csv(P.OUT / "grid.csv")
    c = pd.read_csv(P.OUT / "floor_curve.csv")
    return g, c


def best_plans(g):
    rows = []
    for M, gm in g.groupby("M"):
        for p in POLICIES:
            i = gm[p].idxmax()
            r = gm.loc[i]
            rows.append(dict(M=M, policy=p, rule=r.rule, K=int(r.K), service=r[p],
                             proposed38=float(gm[(gm.rule == "proposed") & (gm.K == 38)][p].iloc[0])))
    return pd.DataFrame(rows)


def equiv(curve_M, curve_S, target):
    """Fleet at which the floor curve reaches `target` (linear interpolation);
    NaN when the floor never reaches it inside the curve."""
    S = np.maximum.accumulate(curve_S)        # the floor is non-decreasing in M
    if target > S[-1] + 1e-9:
        return np.nan
    if target <= S[0]:
        return float(curve_M[0])
    return float(np.interp(target, S, curve_M))


def equivalent_aircraft(g, c):
    rows = []
    for (rule, K), gk in g.groupby(["rule", "K"]):
        ck = c[(c.rule == rule) & (c.K == K)].sort_values("M")
        cm, cs = ck.M.to_numpy(float), ck.floor.to_numpy(float)
        for _, r in gk.iterrows():
            rec = dict(rule=rule, K=K, M=r.M)
            for p in POLICIES[1:]:
                m_eq = equiv(cm, cs, r[p])
                rec[f"{p}_eq_M"] = m_eq
                rec[f"{p}_extra_aircraft"] = m_eq - r.M if np.isfinite(m_eq) else np.nan
                rec[f"{p}_extra_pct"] = 100 * (m_eq - r.M) / r.M if np.isfinite(m_eq) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def fleet_needed(g, c, rule, K, policy, target):
    """Smallest fleet at which `policy` on plan (rule, K) reaches `target`.

    The floor uses its fine curve. Other policies are known on the coarse fleet
    grid; between grid points their service is interpolated linearly, which the
    concavity of service in fleet makes slightly conservative."""
    if policy == "floor":
        ck = c[(c.rule == rule) & (c.K == K)].sort_values("M")
        return equiv(ck.M.to_numpy(float), ck.floor.to_numpy(float), target)
    gk = g[(g.rule == rule) & (g.K == K)].sort_values("M")
    M, S = gk.M.to_numpy(float), np.maximum.accumulate(gk[policy].to_numpy(float))
    if target > S[-1] + 1e-9:
        return np.nan
    if target <= S[0]:
        # below the first grid point, scale by the floor curve's shape
        return float(M[0] * target / S[0])
    return float(np.interp(target, S, M))


def iso_service(g, c):
    rows = []
    for (rule, K), _ in g.groupby(["rule", "K"]):
        for p in ("floor", "method", "ceiling", "point", "unc_static"):
            for t in TARGETS:
                rows.append(dict(rule=rule, K=K, policy=p, target=t,
                                 M_needed=fleet_needed(g, c, rule, K, p, t)))
    return pd.DataFrame(rows)


def cost_optimal(iso):
    rows = []
    for (p, t), d in iso.groupby(["policy", "target"]):
        d = d.dropna(subset=["M_needed"])
        if d.empty:
            continue
        for r in RATIOS:
            cost = d.K * r + d.M_needed
            i = cost.idxmin()
            rows.append(dict(policy=p, target=t, ratio=r, rule=d.loc[i, "rule"],
                             K=int(d.loc[i, "K"]), M=float(d.loc[i, "M_needed"]),
                             cost_in_aircraft=float(cost.loc[i])))
    return pd.DataFrame(rows)


def main():
    g, c = load()
    bp = best_plans(g)
    ea = equivalent_aircraft(g, c)
    iso = iso_service(g, c)
    co = cost_optimal(iso)
    bp.to_csv(P.OUT / "best_plans.csv", index=False)
    ea.to_csv(P.OUT / "equivalent_aircraft.csv", index=False)
    iso.to_csv(P.OUT / "iso_service.csv", index=False)
    co.to_csv(P.OUT / "cost_optimal.csv", index=False)
    pd.set_option("display.width", 220)
    print("Best plan by fleet and dispatch rule")
    print(bp.pivot(index="M", columns="policy", values="K").to_string())
    print(bp.pivot(index="M", columns="policy", values="rule").to_string())
    print(bp.pivot(index="M", columns="policy", values="service").round(2).to_string())
    print("\nCost-optimal plans, ratio 5")
    print(co[co.ratio == 5].pivot(index="target", columns="policy",
                                   values="cost_in_aircraft").round(0).to_string())


if __name__ == "__main__":
    main()
