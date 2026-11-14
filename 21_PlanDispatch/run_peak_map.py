"""Where the proposed rule places aircraft, and where its gain comes from.

Reference plan: 45 demand-ranked sites, 300 aircraft, at most ten aircraft per
site, fold 1 (calibrate on day 6, score day 7), AMSGCN forecasts.

Writes outputs/peak_map_45_300.json with
  * the 11:45 slot of the test day (the slot of Figure 1, sampled exactly as
    export_data.fig1_study does, so the four example sites keep 10/7/3/1):
    realized requirement and the aircraft placed by the floor, the proposed
    rule and the perfect forecast at every site, and the high-value orders each
    serves in that slot;
  * the whole test day: high-value orders served at every site under the three
    rules, run exactly as Plan.policy does (seed 0), so the totals reproduce
    the service shares in cap_grid.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core_pd as P  # noqa: E402

M, K, CAP, SLOT_HOUR = 300, 45, 10, 12.0


def per_site(pl, x, day="test"):
    out = np.zeros(pl.S)
    for (t, j), arr in pl.sc[day].order_book.items():
        cap = x[t, j] * P.SLOT
        if cap <= 0:
            continue
        k = int(np.searchsorted(np.cumsum(arr[:, 0]), cap, side="right"))
        out[j] += (arr[:k, 1] >= P.A.HIGH_VALUE_THRESHOLD).sum()
    return out


def main():
    st = P.Study(positioning=True)
    pl = st.plan("proposed", K)
    ec, ee = pl.forecast("AMSGCN", "val"), pl.forecast("AMSGCN", "test")
    lam = pl.choose_lambda(ec, M, "val", CAP)
    bc = (1 - lam) * pl.static["val"] + lam * ec
    be = (1 - lam) * pl.static["test"] + lam * ee
    slots = st.slots["test"]
    T = len(slots)

    # one slot, sampled as in Figure 1
    t = int(np.argmin(np.abs((slots % 96 + 0.5) / 4 - SLOT_HOUR)))
    smp1 = P.residual_sampler(bc, pl.truth["val"], seed=0)(be[t:t + 1])[0]
    xs = {"floor": P.allocate_point(pl.static["test"][t:t + 1], M, CAP)[0],
          "proposed": P.allocate_expected(smp1[None], M, CAP)[0],
          "perfect": P.allocate_point(pl.truth["test"][t:t + 1], M, CAP)[0]}
    R = pl.truth["test"][t]

    def served_slot(x):
        X = np.zeros((T, pl.S))
        X[t] = x
        return float(pl.served(X, "test")[t])

    slot = {
        "hour": float((slots[t] % 96) / 4),
        "need_minutes": [round(float(v), 2) for v in R],
        "x": {k: [int(v) for v in x] for k, x in xs.items()},
        "served": {k: served_slot(x) for k, x in xs.items()},
        "idle_minutes": {k: round(float(np.clip(15 * x - R, 0, None).sum()), 1) for k, x in xs.items()},
        "need_aircraft_total": round(float(R.sum() / 15), 1),
        "sites_need_above_cap": int((R / 15 > CAP).sum()),
    }

    # whole day, as Plan.policy runs it
    smp = P.residual_sampler(bc, pl.truth["val"], seed=0)
    X = {"floor": P.allocate_point(pl.static["test"], M, CAP),
         "proposed": P.allocate_expected(smp(be), M, CAP),
         "perfect": P.allocate_point(pl.truth["test"], M, CAP)}
    day = {k: per_site(pl, x) for k, x in X.items()}
    N = st.district["test"]
    need_day = pl.truth["test"].sum(axis=0) / 15
    busy = np.argsort(need_day)[::-1]
    same_slots = {int(j): float((X["floor"][:, j] == X["proposed"][:, j]).mean()) for j in range(pl.S)}

    out = {
        "plan": {"rule": "proposed", "K": K, "M": M, "cap": CAP, "lambda": lam, "fold": 1},
        "sites": [{"s": int(j), "node": int(n), "lat": round(float(st.coords[n][0]), 6),
                   "lng": round(float(st.coords[n][1]), 6)} for j, n in enumerate(pl.site_idx)],
        "slot": slot,
        "day": {
            "served_by_site": {k: [int(v) for v in d] for k, d in day.items()},
            "share_pct": {k: float(100 * d.sum() / N) for k, d in day.items()},
            "need_aircraft_by_site": [round(float(v), 1) for v in need_day],
            "busiest_order": [int(j) for j in busy],
            "share_slots_same_x": same_slots,
            "district_orders": int(N),
        },
    }
    path = HERE / "outputs" / "peak_map_45_300.json"
    path.write_text(json.dumps(out, indent=1))
    d = day["proposed"] - day["floor"]
    print(f"wrote {path.name}: slot {slot['hour']} served {slot['served']}; day shares {out['day']['share_pct']}; "
          f"gain {d.sum():.0f}, gainers {(d > 0).sum()}, losers {(d < 0).sum()}")


if __name__ == "__main__":
    main()
