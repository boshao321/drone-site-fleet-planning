"""Standard facility-location procedures as siting baselines.

The lexicographic rule in Stage 2 is compared against the two procedures a
referee will name first. Both are given exactly the same candidate set, the same
K, and the same evaluation, so the comparison isolates the selection rule.

  MCLP        maximum covering location problem. Choose K sites to maximise the
              high-value demand lying within the service radius of some chosen
              site. Solved by the standard greedy algorithm, which for a
              monotone submodular coverage objective is guaranteed within
              1 - 1/e of the optimum; the realised optimality gap is reported
              against a Lagrangian-style upper bound.
  p-median    choose K sites to minimise demand-weighted distance from each
              demand grid to its nearest site. Solved by alternating assignment
              and re-selection from the candidate set, restarted from several
              seeds and reported at the best objective found.
  k-means     the clustering device common in this literature, run on
              demand-weighted grid coordinates, with each centroid snapped to
              its nearest candidate grid.

Each is scored on the same three quantities the manuscript reports for the
proposed rule: coverage of high-value order endpoints, coverage of the hundred
busiest OD links, and agreement with the operator-built stations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "09_Forecasting_Rebuilt"))
sys.path.insert(0, str(ROOT / "10_Allocation" / "scripts"))

import allocation as A            # noqa: E402
from core import data as D        # noqa: E402

OUT = ROOT / "13_SitingBaselines" / "outputs"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
T_ST = ROOT / "07_NetworkBenefit_KStar_NoWeights" / "outputs" / "tables"

KSTAR = 38
SERVICE_RADIUS_KM = 3.0
CORE_LINKS = 100
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
PMEDIAN_SEEDS = list(range(60))


def load_candidates():
    cand = pd.read_csv(T_ST / "candidate_network_indicators_no_weights.csv")
    coords = cand[["lat", "lng"]].to_numpy()
    hv = cand["high_value_orders"].to_numpy(dtype=float)
    return cand, coords, hv


def distance_matrix(coords):
    la, lo = coords[:, 0], coords[:, 1]
    return A.haversine_km(la[:, None], lo[:, None], la[None, :], lo[None, :])


def real_station_grids(step=0.010):
    import importlib.util
    spec_path = ROOT / "07_NetworkBenefit_KStar_NoWeights" / "scripts" / "run_network_benefit_kstar.py"
    spec = importlib.util.spec_from_file_location("kstar", spec_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["kstar"] = m
    spec.loader.exec_module(m)
    real = m.read_real_station_grids()
    return set(zip(real["validation_grid_lat"], real["validation_grid_lng"]))


# ------------------------------------------------------------------ methods


def greedy_mclp(dist, hv, k, radius):
    """Greedy maximum-coverage. Each demand grid is covered if some chosen site
    is within the service radius; a covered grid contributes its high-value
    volume once."""
    within = dist <= radius
    chosen, covered = [], np.zeros(len(hv), dtype=bool)
    for _ in range(k):
        gain = ((within & ~covered) * hv).sum(axis=1)
        gain[chosen] = -1
        j = int(gain.argmax())
        chosen.append(j)
        covered |= within[j]
    return chosen


def mclp_upper_bound(dist, hv, k, radius):
    """A simple upper bound on the covered volume: the k largest single-site
    coverages summed, which cannot be beaten by any feasible selection."""
    within = dist <= radius
    single = (within * hv).sum(axis=1)
    return float(np.sort(single)[::-1][:k].sum())


def p_median_objective(dist, hv, chosen):
    return float((dist[list(chosen)].min(axis=0) * hv).sum())


def p_median(dist, hv, k, seeds=PMEDIAN_SEEDS):
    """Teitz-Bart vertex substitution, restarted, run to a local optimum.

    The procedure repeatedly asks whether replacing one chosen site by one
    unchosen candidate lowers demand-weighted distance, and stops only when no
    single swap does. An earlier version alternated assignment and re-selection
    instead, which is Lloyd's algorithm on a discrete candidate set and stalls
    well short of that: it terminated at a solution worse than demand-weighted
    k-means, which made a comparison table claim that each procedure was best on
    the objective it optimises when p-median was not.

    Greedy construction seeds one restart, since the greedy solution is a strong
    starting point and makes the result reproducible rather than seed-dependent.
    """
    n = len(hv)
    best, best_obj = None, np.inf

    for s in range(-1, len(seeds)):
        if s < 0:
            cur = []
            for _ in range(k):
                cand = [j for j in range(n) if j not in cur]
                objs = [p_median_objective(dist, hv, cur + [j]) for j in cand]
                cur.append(cand[int(np.argmin(objs))])
        else:
            rng = np.random.default_rng(seeds[s])
            cur = list(rng.choice(n, k, replace=False))

        cur = list(cur)
        obj = p_median_objective(dist, hv, cur)
        improved = True
        while improved:
            improved = False
            for pos in range(k):
                base = [c for i, c in enumerate(cur) if i != pos]
                sub = dist[base].min(axis=0)             # cost without this site
                for cand in range(n):
                    if cand in cur:
                        continue
                    trial = float((np.minimum(sub, dist[cand]) * hv).sum())
                    if trial < obj - 1e-12:
                        cur[pos], obj, improved = cand, trial, True
                        break
                if improved:
                    break
        if obj < best_obj:
            best_obj, best = obj, list(cur)
    return best, best_obj


def weighted_kmeans(coords, hv, k, seeds=SEEDS):
    """Demand-weighted k-means on grid coordinates, centroids snapped to the
    nearest candidate grid."""
    best, best_inertia = None, np.inf
    for s in seeds:
        rng = np.random.default_rng(s)
        idx = rng.choice(len(coords), k, replace=False, p=hv / hv.sum())
        cent = coords[idx].copy()
        for _ in range(100):
            d = A.haversine_km(coords[:, 0][:, None], coords[:, 1][:, None],
                               cent[:, 0][None, :], cent[:, 1][None, :])
            lab = d.argmin(axis=1)
            new = cent.copy()
            for c in range(k):
                m = lab == c
                if m.sum() and hv[m].sum() > 0:
                    new[c] = (coords[m] * hv[m, None]).sum(0) / hv[m].sum()
            if np.allclose(new, cent):
                cent = new
                break
            cent = new
        d = A.haversine_km(coords[:, 0][:, None], coords[:, 1][:, None],
                           cent[:, 0][None, :], cent[:, 1][None, :])
        inertia = float((d.min(axis=1) * hv).sum())
        if inertia < best_inertia:
            dc = A.haversine_km(cent[:, 0][:, None], cent[:, 1][:, None],
                                coords[:, 0][None, :], coords[:, 1][None, :])
            snapped = list(dict.fromkeys(dc.argmin(axis=1).tolist()))
            while len(snapped) < k:
                for j in np.argsort(-hv):
                    if j not in snapped:
                        snapped.append(int(j)); break
            best_inertia, best = inertia, snapped[:k]
    return best, best_inertia


# ----------------------------------------------------------------- scoring


def score(name, chosen, cand, coords, hv, dist, edges, real_grids):
    sel = set(map(int, chosen))
    hv_cov = 100.0 * hv[list(sel)].sum() / hv.sum()

    core = edges.sort_values("order_flow", ascending=False).head(CORE_LINKS)
    key = {(round(float(r.lat), 6), round(float(r.lng), 6)) for _, r in
           cand.iloc[list(sel)].iterrows()}
    link_cov = 100.0 * sum(
        1 for r in core.itertuples(index=False)
        if (round(float(r.p_lat), 6), round(float(r.p_lng), 6)) in key
        or (round(float(r.d_lat), 6), round(float(r.d_lng), 6)) in key) / len(core)

    matched = len(key & real_grids)
    within = dist <= SERVICE_RADIUS_KM
    demand_within = 100.0 * hv[within[list(sel)].any(axis=0)].sum() / hv.sum()
    mean_dist = float((dist[list(sel)].min(axis=0) * hv).sum() / hv.sum())

    return {
        "method": name,
        "K": len(sel),
        "high_value_endpoint_coverage_pct": hv_cov,
        "high_value_within_3km_pct": demand_within,
        "key_od_link_coverage_pct": link_cov,
        "demand_weighted_mean_distance_km": mean_dist,
        "matched_operator_grids": matched,
        "recall_vs_operator_pct": 100.0 * matched / len(real_grids),
    }


def main():
    cand, coords, hv = load_candidates()
    dist = distance_matrix(coords)
    edges = pd.read_csv(T_ST / "operational_grid_od_edges.csv")
    real_grids = real_station_grids()
    print(f"Candidates {len(cand)}, K = {KSTAR}, operator grids {len(real_grids)}")

    proposed = pd.read_csv(T_ST / "selected_kstar_grids_no_weights.csv")
    key_to_idx = {(round(float(a), 6), round(float(b), 6)): i
                  for i, (a, b) in enumerate(coords)}
    proposed_idx = [key_to_idx[(round(float(a), 6), round(float(b), 6))]
                    for a, b in zip(proposed["lat"], proposed["lng"])
                    if (round(float(a), 6), round(float(b), 6)) in key_to_idx]

    rows = [score("Proposed (lexicographic)", proposed_idx, cand, coords, hv, dist, edges, real_grids)]

    m = greedy_mclp(dist, hv, KSTAR, SERVICE_RADIUS_KM)
    rows.append(score("MCLP (greedy)", m, cand, coords, hv, dist, edges, real_grids))

    p, pobj = p_median(dist, hv, KSTAR)
    rows.append(score("p-median", p, cand, coords, hv, dist, edges, real_grids))
    print(f"p-median objective {pobj:,.1f} demand-weighted km "
          f"({pobj / hv.sum():.4f} km per high-value endpoint order)")

    kmi, _ = weighted_kmeans(coords, hv, KSTAR)
    rows.append(score("k-means (demand-weighted)", kmi, cand, coords, hv, dist, edges, real_grids))

    vol = list(np.argsort(-cand["endpoint_orders"].to_numpy())[:KSTAR])
    rows.append(score("Endpoint volume only", vol, cand, coords, hv, dist, edges, real_grids))

    out = pd.DataFrame(rows)

    within = dist <= SERVICE_RADIUS_KM
    covered_by_mclp = hv[within[m].any(axis=0)].sum()
    ub = mclp_upper_bound(dist, hv, KSTAR, SERVICE_RADIUS_KM)
    print(f"\nGreedy MCLP covered high-value volume {covered_by_mclp:,.0f}; "
          f"loose upper bound {min(ub, hv.sum()):,.0f}; "
          f"greedy attains {100*covered_by_mclp/min(ub, hv.sum()):.1f}% of it "
          f"(the 1-1/e guarantee is 63.2%)")

    # the manuscript states that each procedure wins on the objective it
    # optimises; that is a claim about these numbers, so it is asserted here
    # rather than left to the reader
    own = {"MCLP (greedy)": ("high_value_within_3km_pct", "max"),
           "p-median": ("demand_weighted_mean_distance_km", "min"),
           "k-means (demand-weighted)": ("demand_weighted_mean_distance_km", "min")}
    print("\nIs each procedure best on the objective it optimises?")
    for method, (col, sense) in own.items():
        v = out.loc[out.method == method, col].iloc[0]
        best = out[col].max() if sense == "max" else out[col].min()
        winner = out.loc[out[col].idxmax() if sense == "max" else out[col].idxmin(), "method"]
        ok = abs(v - best) < 1e-9
        print(f"  {method:<28} {col} = {v:.4f}; best is {best:.4f} ({winner})"
              f"  ->  {'yes' if ok else 'NO'}")
    assert abs(out.loc[out.method == "p-median", "demand_weighted_mean_distance_km"].iloc[0]
               - out["demand_weighted_mean_distance_km"].min()) < 1e-9, \
        "p-median is not best on demand-weighted distance; the solver stopped short"

    out.to_csv(OUT / "tables" / "siting_baseline_comparison.csv", index=False,
               encoding="utf-8-sig")
    print("\n" + out.round(2).to_string(index=False))

    overlap = []
    sets = {"Proposed": set(proposed_idx), "MCLP": set(m), "p-median": set(p),
            "k-means": set(kmi), "Volume": set(vol)}
    for a in sets:
        overlap.append({"method": a, **{b: len(sets[a] & sets[b]) for b in sets}})
    ov = pd.DataFrame(overlap).set_index("method")
    ov.to_csv(OUT / "tables" / "siting_method_overlap.csv", encoding="utf-8-sig")
    print("\nNumber of shared sites between methods (out of 38)")
    print(ov.to_string())


if __name__ == "__main__":
    main()
