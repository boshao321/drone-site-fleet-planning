"""Joint planning under pad limits.

Reads outputs/cap_grid.csv (and the floor curves) and answers the planner's
question for every pad limit c: which pad count K and fleet M deliver a
service target most cheaply, when a pad costs r aircraft, and how that answer
and its cost change with the dispatch rule the operator will use.

Every policy is read off the same 14-point fleet grid with the same linear
interpolation, so no rule is favoured by a finer curve.

Writes
    cap_best_plans.csv    best (rule, K) for each cap, fleet and dispatch rule
    cap_iso.csv           fleet needed by each plan to reach each target
    cap_cost_optimal.csv  cheapest (rule, K, M) per cap, target, ratio, rule
    cap_savings.csv       what each dispatch rule saves against the floor at
                          the cost-optimal design, in aircraft equivalents
    cap_currency.csv      the proposed rule and perfect forecast at the floor's
                          cost-optimal plan, priced on that plan's fine curve
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import core_pd as P
from analyze_planning import equiv

POL = ["floor", "unc_static", "point", "method", "foresight", "ceiling"]
TARGETS = [60, 70, 80, 85, 90]
RATIOS = [1, 2, 5, 10, 20]
SUFFIX = "_swap" if "--swap" in sys.argv else ""


def load():
    g = pd.read_csv(P.OUT / f"cap_grid{SUFFIX}.csv")
    c = pd.read_csv(P.OUT / f"cap_floor_curve{SUFFIX}.csv")
    g["cap"] = g["cap"].astype(str)
    c["cap"] = c["cap"].astype(str)
    return g, c


def usable(M, K, cap):
    """Aircraft a plan can actually stand: beyond K * cap the rest are grounded,
    so a planner would never buy them."""
    M = np.asarray(M, float)
    return M if cap == "none" else np.minimum(M, K * float(cap))


def curve_on_usable(M, Sv, K, cap):
    u = usable(M, K, cap)
    df = pd.DataFrame({"u": u, "s": np.asarray(Sv, float)}).groupby("u", sort=True).s.max()
    return df.index.to_numpy(float), np.maximum.accumulate(df.to_numpy(float))


def fleet_needed(M, S, target):
    S = np.maximum.accumulate(np.asarray(S, float))
    M = np.asarray(M, float)
    if target > S[-1] + 1e-9:
        return np.nan
    if target <= S[0]:
        return float(M[0] * target / S[0])
    return float(np.interp(target, S, M))


def worth(cm, cs, M, s1, plateau_tol=0.02):
    """Extra aircraft the floor curve (cm, cs) needs to reach s1 from fleet M.
    Returns inf when s1 lies above what the floor reaches before its curve goes
    flat, i.e. no fleet on this plan buys that service."""
    cs = np.maximum.accumulate(cs)
    if s1 > cs[-1] - plateau_tol and s1 > cs[np.searchsorted(cm, M, side="right") - 1] + 1e-9:
        flat_from = cm[np.argmax(cs >= cs[-1] - plateau_tol)]
        if s1 > cs[-1] + 1e-9 or flat_from <= M:
            return np.inf
    m = equiv(cm, cs, s1)
    return (m - M) if np.isfinite(m) else np.inf


def main():
    g, c = load()
    caps = [x for x in ("5", "10", "20", "none") if x in set(g.cap)]

    best = []
    for (cap, M), d in g.groupby(["cap", "M"]):
        for p in POL:
            i = d[p].idxmax()
            best.append(dict(cap=cap, M=M, policy=p, rule=d.loc[i, "rule"], K=int(d.loc[i, "K"]),
                             service=d.loc[i, p]))
    best = pd.DataFrame(best)

    iso = []
    for (rule, K, cap), d in g.groupby(["rule", "K", "cap"]):
        d = d.sort_values("M")
        for p in POL:
            u, sv_ = curve_on_usable(d.M, d[p], K, cap)
            for t in TARGETS:
                iso.append(dict(rule=rule, K=K, cap=cap, policy=p, target=t,
                                M_needed=fleet_needed(u, sv_, t)))
    iso = pd.DataFrame(iso)

    co = []
    for (cap, p, t), d in iso.groupby(["cap", "policy", "target"]):
        d = d.dropna(subset=["M_needed"])
        if d.empty:
            continue
        for r in RATIOS:
            cost = r * d.K + d.M_needed
            i = cost.idxmin()
            co.append(dict(cap=cap, policy=p, target=t, ratio=r, rule=d.loc[i, "rule"],
                           K=int(d.loc[i, "K"]), M=float(d.loc[i, "M_needed"]),
                           cost=float(cost.loc[i])))
    co = pd.DataFrame(co)

    sv = []
    for (cap, t, r), d in co.groupby(["cap", "target", "ratio"]):
        d = d.set_index("policy")
        if "floor" not in d.index:
            continue
        base = d.loc["floor", "cost"]
        for p in POL[1:]:
            if p in d.index:
                sv.append(dict(cap=cap, target=t, ratio=r, policy=p,
                               floor_plan=f"{d.loc['floor','rule']} K={d.loc['floor','K']} M={d.loc['floor','M']:.0f}",
                               plan=f"{d.loc[p,'rule']} K={d.loc[p,'K']} M={d.loc[p,'M']:.0f}",
                               saving_aircraft=base - d.loc[p, "cost"],
                               saving_pct=100 * (base - d.loc[p, "cost"]) / base))
    sv = pd.DataFrame(sv)

    # at each cap and fleet, the floor's best plan, and what the dispatch rules
    # are worth there on that plan's own fine floor curve
    cur = []
    for (cap, M), d in g.groupby(["cap", "M"]):
        i = d.floor.idxmax()
        rule, K = d.loc[i, "rule"], int(d.loc[i, "K"])
        ck = c[(c.rule == rule) & (c.K == K) & (c.cap == cap)].sort_values("M")
        cm, cs = curve_on_usable(ck.M, ck.floor, K, cap)
        Mu = float(usable(M, K, cap))
        for p in ("unc_static", "point", "method", "foresight", "ceiling"):
            cur.append(dict(cap=cap, M=M, usable=Mu, rule=rule, K=K, policy=p,
                            service_pp=d.loc[i, p] - d.loc[i, "floor"],
                            aircraft=worth(cm, cs, Mu, d.loc[i, p])))
    cur = pd.DataFrame(cur)

    for nm, df in (("cap_best_plans", best), ("cap_iso", iso), ("cap_cost_optimal", co),
                   ("cap_savings", sv), ("cap_currency", cur)):
        df.to_csv(P.OUT / f"{nm}{SUFFIX}.csv", index=False)

    pd.set_option("display.width", 250)
    for cap in caps:
        b = best[best.cap == cap]
        print(f"\n=== cap {cap}: best K by fleet and dispatch rule (service)")
        tab = b.assign(v=b.rule.str[:2] + b.K.astype(str) + ":" + b.service.round(1).astype(str))
        print(tab.pivot(index="M", columns="policy", values="v")[POL].to_string())
    print("\n=== cost-optimal designs, ratio 5")
    d = co[co.ratio == 5]
    print(d.assign(v=d.rule.str[:2] + d.K.astype(str) + "/" + d.M.round(0).astype(int).astype(str)
                   + " (" + d.cost.round(0).astype(int).astype(str) + ")")
          .pivot_table(index=["cap", "target"], columns="policy", values="v", aggfunc="first")[POL]
          .to_string())
    print("\n=== saving against the floor at the cost-optimal design (aircraft), ratio 5")
    d = sv[sv.ratio == 5]
    print(d.pivot_table(index=["cap", "target"], columns="policy", values="saving_aircraft")
          .round(1).to_string())
    print("\n=== proposed rule at the floor's best plan, worth in aircraft")
    d = cur[cur.policy.isin(["point", "method", "foresight"])]
    print(d.pivot_table(index=["cap", "M"], columns="policy", values="aircraft").round(1).to_string())


if __name__ == "__main__":
    main()
