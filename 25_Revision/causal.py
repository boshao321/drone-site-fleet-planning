"""Causal replay (R1 follow-up): orders enter the queue of their site only at their
payment time, and a sortie can start only after its order has been released.

timed_books(st, pl, ev) rebuilds the high value orders of each (slot, site) with
their release minute; simulate_causal() replays an allocation with individually
tracked aircraft, sorties carried across slots and charged repositioning flights,
where an aircraft that becomes free takes the highest value order that has been
released and has waited at most `wait` minutes, or waits for the next release.
"""
import heapq
import numpy as np
import revlib as R
P, A = R.P, R.A
SLOT = R.SLOT


def timed_books(st, pl, ev):
    """(slot, site) -> array of (tau, price, release minute within the slot), high value only."""
    o = st.orders
    slots = list(st.slots[ev]); pos = {s: i for i, s in enumerate(slots)}
    gi = st.sp.index
    node = np.array([gi.get(k, -1) for k in zip(o["p_lat_snap"], o["p_lng_snap"])])
    keep = node >= 0
    o = o[keep].copy(); o["node"] = node[keep]
    o["site"] = pl.g2s[o["node"].to_numpy()]
    o = o[(o["site"] >= 0) & (o["od_km"] <= A.MAX_SORTIE_RANGE_KM) & o["slot"].isin(pos)
          & (o["package_price"] >= A.HIGH_VALUE_THRESHOLD)]
    xy = st.coords[np.asarray(pl.site_idx)][o["site"].to_numpy()]
    d_sp = A.haversine_km(xy[:, 0], xy[:, 1], o["pickup_latitude"].to_numpy(), o["pickup_longitude"].to_numpy())
    d_ds = A.haversine_km(o["dropoff_latitude"].to_numpy(), o["dropoff_longitude"].to_numpy(), xy[:, 0], xy[:, 1])
    tau = A.tour_minutes(d_sp, o["od_km"].to_numpy(), d_ds, R.ENV["v"], R.ENV["ts"])
    rel = (o["pay_min"].to_numpy() % SLOT).astype(float)
    out = {}
    for (sl, si), idx in o.groupby(["slot", "site"]).indices.items():
        arr = np.stack([tau[idx], o["package_price"].to_numpy()[idx], rel[idx]], axis=1)
        out[(pos[int(sl)], int(si))] = arr[np.argsort(-arr[:, 1], kind="stable")]
    return out


def simulate_causal(st, pl, x, ev, tb, wait=SLOT, own_slot=False, repo_speed=60.0, repo_handle=1.0):
    """Served high value orders per slot (credited to the slot of the order).

    wait: minutes an order may wait after release before it is lost.
    own_slot: if True an order must start inside the slot in which it was released."""
    x = np.asarray(x, dtype=int); T, S = x.shape
    D = R.site_distance(st, pl)
    M = int(x.sum(axis=1).max())
    site = np.full(M, -1, dtype=int); free = np.zeros(M)      # absolute minutes of the day
    served = np.zeros(T)
    pend = {j: [] for j in range(S)}                           # [release, -price, tau, slot, done]
    for t in range(T):
        t0, t1 = t * SLOT, (t + 1) * SLOT
        for j in range(S):
            arr = tb.get((t, j))
            if arr is not None:
                pend[j].extend([[t0 + r, -p, tau, t, False] for tau, p, r in arr])
        # ---- rearrange the fleet to x[t] (as revlib.simulate_aircraft, with absolute clocks)
        cur = np.bincount(site[site >= 0], minlength=S) if (site >= 0).any() else np.zeros(S, dtype=int)
        need = x[t] - cur
        givers = []
        for j in np.where(need < 0)[0]:
            idx = np.where(site == j)[0]
            givers.extend(idx[np.argsort(free[idx], kind="stable")][: -need[j]].tolist())
        res = np.where(site == -1)[0]
        takers = np.repeat(np.where(need > 0)[0], need[need > 0])
        if len(takers):
            pool = givers + res.tolist()
            cost = np.zeros((len(pool), len(takers)))
            for r, a in enumerate(pool):
                cost[r] = D[site[a], takers] + 1e-3 * max(free[a] - t0, 0.0) if site[a] >= 0 else 1e3
            rr, cc = R.linear_sum_assignment(cost); used = set()
            for r, c in zip(rr, cc):
                a = pool[r]; j2 = takers[c]
                if site[a] >= 0 and repo_speed:
                    free[a] = max(free[a], t0) + D[site[a], j2] / repo_speed * 60.0 + repo_handle
                site[a] = j2; used.add(a)
            for a in givers:
                if a not in used: site[a] = -1
        else:
            for a in givers: site[a] = -1
        # ---- serve in continuous time inside the slot
        for j in np.where(x[t] > 0)[0]:
            q = [o for o in pend[j] if not o[4] and (o[0] >= t0 - wait)]
            if own_slot:
                q = [o for o in q if o[3] == t]
            if not q:
                continue
            h = [(max(free[a], t0), a) for a in np.where(site == j)[0]]
            heapq.heapify(h)
            while h:
                f, a = heapq.heappop(h)
                if f >= t1:
                    break
                avail = [o for o in q if not o[4] and o[0] <= f and f - o[0] <= wait and (not own_slot or f < (o[3] + 1) * SLOT)]
                if avail:
                    o = min(avail, key=lambda z: (z[1], z[0]))       # highest value, then earliest
                    o[4] = True; served[o[3]] += 1
                    free[a] = f + o[2]; heapq.heappush(h, (free[a], a))
                else:
                    nxt = [o[0] for o in q if not o[4] and o[0] > f]
                    if nxt and min(nxt) < t1:
                        heapq.heappush(h, (min(nxt), a))
            pend[j] = [o for o in pend[j] if not o[4] and o[0] >= t1 - wait]
    return served
