"""
Core greenfield network optimization logic.
Kept separate from the Streamlit UI so it can be tested or reused (e.g. in a
FastAPI backend later) without dragging in UI code.

Internal demand metric is generic "demand_value" — it can represent orders,
quantity, weight, or volume depending on how the Products table defines the
unit of measure for a given product. The optimizer itself is unit-agnostic:
it just sums and compares demand_value, so whatever unit the business cares
about flows straight through.
"""

import numpy as np
import pandas as pd
import geopandas as gpd


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def auto_grid_step_deg(service_radius_km: float, fraction: float = 0.35,
                        min_deg: float = 0.005, max_deg: float = 0.1) -> float:
    """Pick a sensible candidate-grid spacing automatically from the service
    radius, so this no longer needs to be a user-facing control. Roughly
    1/3 of the service radius is fine enough to find good sites without
    generating far more candidates than the radius could ever distinguish
    between."""
    step = (service_radius_km / 111.0) * fraction
    return float(np.clip(step, min_deg, max_deg))


def generate_candidate_sites(demand_df: pd.DataFrame, grid_step_deg: float = 0.03,
                              neighborhood: int = 1, max_candidates: int = 4000) -> pd.DataFrame:
    """Generate candidate warehouse sites on a grid, anchored around actual
    demand locations rather than a uniform grid over the full bounding box.

    This matters for geographically spread demand (e.g. nationwide data):
    a uniform grid over the whole bounding box scales with area, which can
    reach into the millions of candidates for a continent-spanning dataset
    and crash the app. Anchoring on demand points instead makes candidate
    count scale with the number of demand locations, staying manageable
    regardless of how spread out they are, while still generating a fine
    local grid within each demand cluster.

    `neighborhood` controls how many grid cells out from each demand point
    are included (1 = a 3x3 neighborhood around each point).
    """
    grid_cells = set()
    offsets = range(-neighborhood, neighborhood + 1)
    for lat, lon in zip(demand_df["lat"], demand_df["lon"]):
        base_lat = round(lat / grid_step_deg) * grid_step_deg
        base_lon = round(lon / grid_step_deg) * grid_step_deg
        for dlat in offsets:
            for dlon in offsets:
                grid_cells.add((round(base_lat + dlat * grid_step_deg, 6),
                                 round(base_lon + dlon * grid_step_deg, 6)))

    grid_cells = list(grid_cells)
    if len(grid_cells) > max_candidates:
        # Safety valve for very large demand datasets — randomly thin down
        # rather than crash. Documented as a known limitation for huge inputs.
        rng = np.random.default_rng(42)
        idx = rng.choice(len(grid_cells), size=max_candidates, replace=False)
        grid_cells = [grid_cells[i] for i in idx]

    cand_df = pd.DataFrame(
        [{"site_id": f"S{i+1:04d}", "lat": lat, "lon": lon} for i, (lat, lon) in enumerate(sorted(grid_cells))]
    )

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


# ---------------------------------------------------------------------------
# Geocoding — for users who only have city/country, not lat/lon
# ---------------------------------------------------------------------------

def needs_geocoding(df: pd.DataFrame) -> bool:
    """True if any row is missing lat or lon."""
    if df is None or len(df) == 0:
        return False
    if "lat" not in df.columns or "lon" not in df.columns:
        return True
    return df["lat"].isnull().any() or df["lon"].isnull().any()


def geocode_locations(df: pd.DataFrame, city_col: str = "city", country_col: str = "country",
                       lat_col: str = "lat", lon_col: str = "lon") -> tuple[pd.DataFrame, list]:
    """Fill in missing lat/lon by geocoding city + country with OpenStreetMap's
    Nominatim service (free, no API key, rate-limited to be a polite citizen
    of a shared public service). Returns the updated dataframe and a list of
    row indices that could not be geocoded."""
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    df = df.copy()
    if lat_col not in df.columns:
        df[lat_col] = np.nan
    if lon_col not in df.columns:
        df[lon_col] = np.nan

    geolocator = Nominatim(user_agent="supply_chain_design_by_am")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=1)

    failed_rows = []
    for idx, row in df.iterrows():
        lat_missing = pd.isna(row.get(lat_col))
        lon_missing = pd.isna(row.get(lon_col))
        if not (lat_missing or lon_missing):
            continue

        city = str(row.get(city_col, "") or "").strip()
        country = str(row.get(country_col, "") or "").strip()
        query = ", ".join(part for part in [city, country] if part)
        if not query:
            failed_rows.append(idx)
            continue

        try:
            location = geocode(query)
            if location:
                df.at[idx, lat_col] = location.latitude
                df.at[idx, lon_col] = location.longitude
            else:
                failed_rows.append(idx)
        except Exception:
            failed_rows.append(idx)

    return df, failed_rows


# ---------------------------------------------------------------------------
# Coverage math
# ---------------------------------------------------------------------------

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
    demand already covered by existing facilities (pass existing_df=None or
    an empty dataframe to run a pure greenfield analysis with no baseline).

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
    demand_vals = demand_df["demand_value"].values
    total_demand = demand_vals.sum()
    n_demand = len(demand_df)

    existing_cov = compute_coverage_matrix(existing_df, demand_df, service_radius_km, epsg)
    covered = existing_cov.any(axis=0) if len(existing_cov) > 0 else np.zeros(n_demand, dtype=bool)
    baseline_covered_demand = demand_vals[covered].sum()

    cand_cov = compute_coverage_matrix(cand_df, demand_df, service_radius_km, epsg)

    selected_rows = []
    selected_idx = []
    remaining_idx = list(range(len(cand_df)))
    cap = num_sites if mode == "num_sites" else max_sites

    while len(selected_idx) < cap and remaining_idx:
        best_gain, best_i = -1.0, None
        for i in remaining_idx:
            new_covered = covered | cand_cov[i]
            gain = demand_vals[new_covered].sum() - demand_vals[covered].sum()
            if gain > best_gain:
                best_gain, best_i = gain, i

        if best_i is None or best_gain <= 0:
            break  # no remaining candidate adds any new coverage

        covered = covered | cand_cov[best_i]
        cum_covered_demand = demand_vals[covered].sum()
        row = cand_df.iloc[best_i].to_dict()
        row["incremental_demand_covered"] = round(best_gain, 1)
        row["cumulative_covered_demand"] = round(cum_covered_demand, 1)
        row["cumulative_coverage_pct"] = round(cum_covered_demand / total_demand * 100, 1) if total_demand > 0 else 0
        selected_rows.append(row)
        selected_idx.append(best_i)
        remaining_idx.remove(best_i)

        if mode == "service_target" and row["cumulative_coverage_pct"] >= target_pct:
            break

    selected_df = pd.DataFrame(selected_rows)
    final_covered_demand = demand_vals[covered].sum()
    summary = {
        "total_demand": round(total_demand, 1),
        "baseline_covered_demand": round(baseline_covered_demand, 1),
        "baseline_coverage_pct": round(baseline_covered_demand / total_demand * 100, 1) if total_demand > 0 else 0,
        "final_covered_demand": round(final_covered_demand, 1),
        "final_coverage_pct": round(final_covered_demand / total_demand * 100, 1) if total_demand > 0 else 0,
        "sites_selected": len(selected_idx),
        "target_met": (mode == "service_target" and
                        (final_covered_demand / total_demand * 100 if total_demand > 0 else 0) >= target_pct),
    }
    return selected_df, summary


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_demand_df(df: pd.DataFrame) -> tuple[bool, str]:
    """Business-data validation — does NOT require lat/lon (those can come
    from geocoding). Call coordinates_ready() separately before running."""
    if df is None or len(df) == 0:
        return False, "Customer Demand table is empty — add at least one row."
    required_cols = {"city", "country", "product_id", "demand_value"}
    missing = required_cols - set(df.columns)
    if missing:
        return False, f"Missing required columns: {', '.join(sorted(missing))}"
    if df[["city", "country", "product_id"]].isnull().any().any():
        return False, "Found empty values in city, country, or product_id — every row needs all three."

    # Coerce demand_value to numeric rather than assuming it already is —
    # manual edits in the table can leave stray text, which would otherwise
    # crash the numeric comparison below with a raw Python error.
    numeric_demand = pd.to_numeric(df["demand_value"], errors="coerce")
    if numeric_demand.isnull().any():
        return False, "demand_value has empty or non-numeric entries — every row needs a numeric value."
    if (numeric_demand < 0).any():
        return False, "demand_value cannot be negative."
    return True, ""


def validate_products_df(df: pd.DataFrame) -> tuple[bool, str]:
    if df is None or len(df) == 0:
        return False, "Products table is empty — add at least one product."
    required_cols = {"product_id", "product_name", "unit_of_measure"}
    missing = required_cols - set(df.columns)
    if missing:
        return False, f"Products table missing columns: {', '.join(sorted(missing))}"
    if df[["product_id", "product_name", "unit_of_measure"]].isnull().any().any():
        return False, "Found empty values in product_id, product_name, or unit_of_measure."
    return True, ""


def validate_existing_df(df: pd.DataFrame) -> tuple[bool, str]:
    """Existing facilities table is optional — empty is valid."""
    if df is None or len(df) == 0:
        return True, ""
    required_cols = {"facility_id", "city", "country"}
    missing = required_cols - set(df.columns)
    if missing:
        return False, f"Existing facilities table missing columns: {', '.join(sorted(missing))}"
    return True, ""


def coordinates_ready(df: pd.DataFrame) -> tuple[bool, str]:
    """Check a table has valid, complete lat/lon — call this after geocoding,
    right before the table is used in the optimizer."""
    if df is None or len(df) == 0:
        return True, ""  # empty is a separate concern, handled elsewhere
    if "lat" not in df.columns or "lon" not in df.columns:
        return False, "No coordinates found — use 'Geocode missing locations' first."
    lat_numeric = pd.to_numeric(df["lat"], errors="coerce")
    lon_numeric = pd.to_numeric(df["lon"], errors="coerce")
    if lat_numeric.isnull().any() or lon_numeric.isnull().any():
        return False, "Some rows are missing or have non-numeric coordinates — click 'Geocode missing locations' or fill lat/lon in manually."
    if not lat_numeric.between(-90, 90).all() or not lon_numeric.between(-180, 180).all():
        return False, "Latitude/longitude values are out of valid range."
    return True, ""
