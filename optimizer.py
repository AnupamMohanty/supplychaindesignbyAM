"""
Core greenfield network optimization logic.
Kept separate from the Streamlit UI so it can be tested or reused (e.g. in a
FastAPI backend later) without dragging in UI code.
"""

import numpy as np
import pandas as pd
import geopandas as gpd


def generate_candidate_sites(demand_df: pd.DataFrame, grid_step_deg: float = 0.03,
                              buffer_deg: float = 0.02) -> pd.DataFrame:
    """Generate candidate warehouse sites on a grid covering the demand region."""
    min_lat, max_lat = demand_df["lat"].min() - buffer_deg, demand_df["lat"].max() + buffer_deg
    min_lon, max_lon = demand_df["lon"].min() - buffer_deg, demand_df["lon"].max() + buffer_deg

    lat_points = np.arange(min_lat, max_lat, grid_step_deg)
    lon_points = np.arange(min_lon, max_lon, grid_step_deg)

    candidates = []
    cid = 1
    for lat in lat_points:
        for lon in lon_points:
            candidates.append({"site_id": f"S{cid:03d}", "lat": lat, "lon": lon})
            cid += 1
    cand_df = pd.DataFrame(candidates)

    # Simple radial lease-cost proxy — replace with real cost data when available
    center_lat, center_lon = demand_df["lat"].mean(), demand_df["lon"].mean()
    cand_df["dist_from_center_km"] = np.sqrt(
        (cand_df["lat"] - center_lat) ** 2 + (cand_df["lon"] - center_lon) ** 2
    ) * 111
    cand_df["monthly_lease_cost"] = (80000 - cand_df["dist_from_center_km"] * 3000).clip(lower=20000).round(0)

    return cand_df


def estimate_utm_epsg(lon: float, lat: float) -> int:
    """Pick an appropriate UTM zone EPSG code from a longitude/latitude, so
    distance calculations are accurate in meters rather than degrees."""
    zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def score_candidates(demand_df: pd.DataFrame, cand_df: pd.DataFrame,
                      service_radius_km: float = 8.0,
                      coverage_weight: float = 0.7) -> pd.DataFrame:
    """Score each candidate site on demand coverage vs. cost, using proper
    geodesic distance (reprojected to a local UTM zone, not raw lat/lon degrees)."""
    demand_gdf = gpd.GeoDataFrame(
        demand_df, geometry=gpd.points_from_xy(demand_df.lon, demand_df.lat), crs="EPSG:4326"
    )
    cand_gdf = gpd.GeoDataFrame(
        cand_df, geometry=gpd.points_from_xy(cand_df.lon, cand_df.lat), crs="EPSG:4326"
    )

    epsg = estimate_utm_epsg(demand_df["lon"].mean(), demand_df["lat"].mean())
    demand_m = demand_gdf.to_crs(epsg=epsg)
    cand_m = cand_gdf.to_crs(epsg=epsg)

    results = []
    for _, site in cand_m.iterrows():
        dists = demand_m.geometry.distance(site.geometry) / 1000
        within = dists <= service_radius_km
        covered_orders = demand_df.loc[within.values, "daily_orders"].sum()
        results.append({
            "site_id": site["site_id"],
            "covered_demand_points": int(within.sum()),
            "covered_daily_orders": covered_orders,
            "avg_dist_to_covered_km": round(dists[within].mean(), 2) if within.sum() > 0 else 0,
        })

    score_df = pd.DataFrame(results).merge(cand_df, on="site_id")

    max_orders = score_df["covered_daily_orders"].max()
    score_df["coverage_norm"] = score_df["covered_daily_orders"] / max_orders if max_orders > 0 else 0

    cost_range = score_df["monthly_lease_cost"].max() - score_df["monthly_lease_cost"].min()
    if cost_range > 0:
        score_df["cost_norm"] = 1 - (score_df["monthly_lease_cost"] - score_df["monthly_lease_cost"].min()) / cost_range
    else:
        score_df["cost_norm"] = 1.0

    score_df["feasibility_score"] = (
        coverage_weight * score_df["coverage_norm"] + (1 - coverage_weight) * score_df["cost_norm"]
    )

    return score_df.sort_values("feasibility_score", ascending=False).reset_index(drop=True)


def _xy_meters(df: pd.DataFrame, epsg: int) -> np.ndarray:
    """Project lat/lon points to (x, y) meters in a local UTM zone."""
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
    ).to_crs(epsg=epsg)
    return np.column_stack([gdf.geometry.x.values, gdf.geometry.y.values])


def compute_coverage_matrix(source_df: pd.DataFrame, demand_df: pd.DataFrame,
                             service_radius_km: float, epsg: int) -> np.ndarray:
    """Boolean matrix (n_sources x n_demand): True where a source point is
    within service_radius_km of a demand point, using projected distance."""
    n_demand = len(demand_df)
    if source_df is None or len(source_df) == 0:
        return np.zeros((0, n_demand), dtype=bool)
    src_xy = _xy_meters(source_df, epsg)
    dem_xy = _xy_meters(demand_df, epsg)
    dists = np.sqrt(((src_xy[:, None, :] - dem_xy[None, :, :]) ** 2).sum(axis=2))
    return dists <= service_radius_km * 1000


def greedy_select_sites(demand_df: pd.DataFrame, cand_df: pd.DataFrame,
                         existing_df: pd.DataFrame | None,
                         service_radius_km: float = 8.0,
                         mode: str = "num_sites",
                         num_sites: int = 5,
                         target_pct: float = 80.0,
                         max_sites: int = 25) -> tuple[pd.DataFrame, dict]:
    """
    Greedy maximal-coverage facility location: iteratively picks the candidate
    site that covers the most currently-uncovered demand, accounting for
    demand already covered by existing facilities.

    mode="num_sites": open exactly `num_sites` new sites, maximize coverage.
    mode="service_target": open as many sites as needed (up to `max_sites`)
      to reach `target_pct` of total demand covered, stopping early once hit.

    This is a well-known heuristic for the "maximal covering location
    problem" — not guaranteed globally optimal, but close in practice and
    far better than scoring each site independently, since independent
    scoring double-counts demand that multiple nearby candidates would
    all claim to cover.
    """
    epsg = estimate_utm_epsg(demand_df["lon"].mean(), demand_df["lat"].mean())
    orders = demand_df["daily_orders"].values
    total_orders = orders.sum()
    n_demand = len(demand_df)

    existing_cov = compute_coverage_matrix(existing_df, demand_df, service_radius_km, epsg)
    covered = existing_cov.any(axis=0) if len(existing_cov) > 0 else np.zeros(n_demand, dtype=bool)
    baseline_covered_orders = orders[covered].sum()

    cand_cov = compute_coverage_matrix(cand_df, demand_df, service_radius_km, epsg)

    selected_rows = []
    selected_idx = []
    remaining_idx = list(range(len(cand_df)))
    cap = num_sites if mode == "num_sites" else max_sites

    while len(selected_idx) < cap and remaining_idx:
        best_gain, best_i = -1.0, None
        for i in remaining_idx:
            new_covered = covered | cand_cov[i]
            gain = orders[new_covered].sum() - orders[covered].sum()
            if gain > best_gain:
                best_gain, best_i = gain, i

        if best_i is None or best_gain <= 0:
            break  # no remaining candidate adds any new coverage

        covered = covered | cand_cov[best_i]
        cum_covered_orders = orders[covered].sum()
        row = cand_df.iloc[best_i].to_dict()
        row["incremental_orders_covered"] = round(best_gain, 1)
        row["cumulative_covered_orders"] = round(cum_covered_orders, 1)
        row["cumulative_coverage_pct"] = round(cum_covered_orders / total_orders * 100, 1) if total_orders > 0 else 0
        selected_rows.append(row)
        selected_idx.append(best_i)
        remaining_idx.remove(best_i)

        if mode == "service_target" and row["cumulative_coverage_pct"] >= target_pct:
            break

    selected_df = pd.DataFrame(selected_rows)
    final_covered_orders = orders[covered].sum()
    summary = {
        "total_daily_orders": round(total_orders, 1),
        "baseline_covered_orders": round(baseline_covered_orders, 1),
        "baseline_coverage_pct": round(baseline_covered_orders / total_orders * 100, 1) if total_orders > 0 else 0,
        "final_covered_orders": round(final_covered_orders, 1),
        "final_coverage_pct": round(final_covered_orders / total_orders * 100, 1) if total_orders > 0 else 0,
        "sites_selected": len(selected_idx),
        "target_met": (mode == "service_target" and
                        (final_covered_orders / total_orders * 100 if total_orders > 0 else 0) >= target_pct),
    }
    return selected_df, summary


def validate_existing_df(df: pd.DataFrame) -> tuple[bool, str]:
    """Existing facilities table is optional — empty is valid."""
    if df is None or len(df) == 0:
        return True, ""
    required_cols = {"facility_id", "lat", "lon"}
    missing = required_cols - set(df.columns)
    if missing:
        return False, f"Existing facilities table missing columns: {', '.join(sorted(missing))}"
    if df[["lat", "lon"]].isnull().any().any():
        return False, "Existing facilities table has empty lat/lon values."
    if not df["lat"].between(-90, 90).all() or not df["lon"].between(-180, 180).all():
        return False, "Existing facilities table has lat/lon values out of valid range."
    return True, ""


def validate_products_df(df: pd.DataFrame) -> tuple[bool, str]:
    """Products table is optional/reference-only for now — empty is valid."""
    if df is None or len(df) == 0:
        return True, ""
    required_cols = {"product_id", "product_name"}
    missing = required_cols - set(df.columns)
    if missing:
        return False, f"Products table missing columns: {', '.join(sorted(missing))}"
    return True, ""


def validate_demand_df(df: pd.DataFrame) -> tuple[bool, str]:
    """Check an uploaded demand CSV has the columns and value ranges we need."""
    required_cols = {"lat", "lon", "daily_orders"}
    missing = required_cols - set(df.columns)
    if missing:
        return False, f"Missing required columns: {', '.join(sorted(missing))}"
    if df[["lat", "lon", "daily_orders"]].isnull().any().any():
        return False, "Found empty values in lat, lon, or daily_orders columns."
    if not df["lat"].between(-90, 90).all():
        return False, "Latitude values must be between -90 and 90."
    if not df["lon"].between(-180, 180).all():
        return False, "Longitude values must be between -180 and 180."
    if (df["daily_orders"] < 0).any():
        return False, "daily_orders cannot be negative."
    return True, ""
