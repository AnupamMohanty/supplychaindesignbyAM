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
                       lat_col: str = "lat", lon_col: str = "lon",
                       progress_callback=None) -> tuple[pd.DataFrame, list]:
    """Fill in missing lat/lon by geocoding city + country with OpenStreetMap's
    Nominatim service (free, no API key, rate-limited to be a polite citizen
    of a shared public service — 1 request/second is the hard floor per
    lookup, by Nominatim's own usage policy, so this can't go faster than
    that per unique location).

    Speed optimization: geocodes each UNIQUE (city, country) pair only ONCE,
    then applies the result to every matching row — for real-world data with
    many rows sharing a city (the common case), this cuts total network
    calls (and thus total wait time) dramatically versus geocoding every
    row independently.

    `progress_callback`, if given, is called as
    progress_callback(done, total, elapsed_seconds) after each unique lookup,
    so the caller can render a live progress bar + elapsed-time indicator.

    Returns the updated dataframe and a list of row indices that could not
    be geocoded."""
    import time as _time
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    df = df.copy()
    if lat_col not in df.columns:
        df[lat_col] = np.nan
    if lon_col not in df.columns:
        df[lon_col] = np.nan

    geolocator = Nominatim(user_agent="supply_chain_design_by_am", timeout=5)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=1, error_wait_seconds=1.0)

    needs_geo_mask = df[lat_col].isna() | df[lon_col].isna()
    rows_needing = df[needs_geo_mask]

    # Group by the exact query string so identical city/country pairs share
    # a single lookup, regardless of which rows they belong to.
    queries_to_rows: dict[str, list] = {}
    failed_rows = []
    for idx, row in rows_needing.iterrows():
        city = str(row.get(city_col, "") or "").strip()
        country = str(row.get(country_col, "") or "").strip()
        query = ", ".join(part for part in [city, country] if part)
        if not query:
            failed_rows.append(idx)
            continue
        queries_to_rows.setdefault(query, []).append(idx)

    unique_queries = list(queries_to_rows.keys())
    start_time = _time.time()
    for i, query in enumerate(unique_queries):
        try:
            location = geocode(query)
            if location:
                for idx in queries_to_rows[query]:
                    df.at[idx, lat_col] = location.latitude
                    df.at[idx, lon_col] = location.longitude
            else:
                failed_rows.extend(queries_to_rows[query])
        except Exception:
            failed_rows.extend(queries_to_rows[query])

        if progress_callback is not None:
            progress_callback(i + 1, len(unique_queries), _time.time() - start_time)

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


def compute_service_radius(service_time_value: float, service_time_unit: str,
                            miles_per_day: float) -> tuple[float, float]:
    """Convert a service-time target + a last-mile daily travel capacity
    assumption into an effective service radius.

    e.g. "I want to serve customers within 1 day, and a truck can cover
    400 miles/day" -> effective radius = 400 miles = 643.7 km.

    Returns (radius_km, radius_miles).
    """
    days = service_time_value if service_time_unit == "Days" else service_time_value / 24.0
    radius_miles = days * miles_per_day
    radius_km = radius_miles * 1.60934
    return radius_km, radius_miles


def compute_weighted_avg_distance_km(demand_df: pd.DataFrame, facility_lat_lon_df: pd.DataFrame,
                                      epsg: int) -> float | None:
    """Demand-weighted average distance (km) from each demand point to its
    NEAREST open facility (existing + newly opened combined) — the classic
    "weighted average service distance" logistics KPI:
        sum(distance_to_nearest_facility * demand) / sum(demand)
    Unlike coverage %, this measures actual service quality across every
    demand point, not just whether it falls inside the radius or not.
    Returns None if there are no facilities to measure against.
    """
    if facility_lat_lon_df is None or len(facility_lat_lon_df) == 0:
        return None
    fac_xy = _xy_meters(facility_lat_lon_df, epsg)
    dem_xy = _xy_meters(demand_df, epsg)
    dists_m = np.sqrt(((dem_xy[:, None, :] - fac_xy[None, :, :]) ** 2).sum(axis=2))
    nearest_km = dists_m.min(axis=1) / 1000
    weights = demand_df["demand_value"].values
    total_weight = weights.sum()
    if total_weight <= 0:
        return None
    return float((nearest_km * weights).sum() / total_weight)


def reverse_geocode_names(sites_df: pd.DataFrame) -> list:
    """Reverse-geocode each site's lat/lon into a human-readable name
    (e.g. "Ashburn, Virginia") using OpenStreetMap's free Nominatim
    service. Falls back to the site_id if a lookup fails, so naming never
    blocks the rest of the app. See reverse_geocode_details() for a version
    that also returns the state separately."""
    return [d["name"] for d in reverse_geocode_details(sites_df)]


def reverse_geocode_details(sites_df: pd.DataFrame) -> list:
    """Like reverse_geocode_names, but returns a dict per site with the
    display name AND the state separately (needed to look up a real
    logistics-hub-city recommendation for that state)."""
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    geolocator = Nominatim(user_agent="supply_chain_design_by_am", timeout=5)
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1, max_retries=1, error_wait_seconds=1.0)

    results = []
    for _, row in sites_df.iterrows():
        fallback = str(row.get("site_id", "DC"))
        try:
            location = reverse((row["lat"], row["lon"]), exactly_one=True, zoom=10)
            if location and location.raw.get("address"):
                addr = location.raw["address"]
                place = addr.get("city") or addr.get("town") or addr.get("village") or \
                    addr.get("county") or addr.get("suburb")
                state = addr.get("state") or addr.get("region")
                country = addr.get("country")
                parts = [p for p in [place, state] if p]
                results.append({"name": ", ".join(parts) if parts else fallback,
                                 "state": state, "country": country})
            else:
                results.append({"name": fallback, "state": None, "country": None})
        except Exception:
            results.append({"name": fallback, "state": None, "country": None})
    return results


def assign_customers_to_facilities(demand_df: pd.DataFrame, facilities_df: pd.DataFrame,
                                    epsg: int, service_radius_km: float | None = None) -> pd.DataFrame:
    """For each demand row, find its NEAREST open facility (existing + new
    combined) and attach that facility's identity + distance.

    If `service_radius_km` is given, the radius is enforced as a REAL
    constraint: a customer beyond that distance from every facility is
    marked "Unserved (beyond service radius)" rather than silently assigned
    to its nearest facility regardless of how far away that is. This keeps
    the map, the per-DC customer counts, and the service-distance metric all
    consistent with the same radius the optimizer actually used.
    `facilities_df` must have columns: facility_name, lat, lon (at minimum).
    """
    result = demand_df.copy().reset_index(drop=True)
    if facilities_df is None or len(facilities_df) == 0:
        result["assigned_facility_name"] = "Unserved (no facilities)"
        result["distance_to_facility_km"] = None
        return result

    fac_xy = _xy_meters(facilities_df, epsg)
    dem_xy = _xy_meters(demand_df, epsg)
    dists_m = np.sqrt(((dem_xy[:, None, :] - fac_xy[None, :, :]) ** 2).sum(axis=2))
    nearest_idx = dists_m.argmin(axis=1)
    nearest_km = dists_m.min(axis=1) / 1000

    facilities_reset = facilities_df.reset_index(drop=True)
    assigned_names = facilities_reset.loc[nearest_idx, "facility_name"].values.astype(object)
    if service_radius_km is not None:
        out_of_range = nearest_km > service_radius_km
        assigned_names[out_of_range] = "Unserved (beyond service radius)"

    result["assigned_facility_name"] = assigned_names
    result["distance_to_facility_km"] = np.round(nearest_km, 2)
    return result


def compute_current_network_health(demand_df: pd.DataFrame, existing_df: pd.DataFrame,
                                    service_radius_km: float) -> dict:
    """Diagnose how well an existing ('current') network serves current demand,
    with ZERO new sites opened — a pure baseline health check. Unlike
    greedy_select_sites, this never touches candidate generation or the
    greedy loop; it's just coverage math against the facilities already
    provided. Returns a summary dict in the same shape as greedy_select_sites'
    summary (sites_selected=0), so it can be saved as a scenario and compared
    against future what-if runs on equal footing."""
    epsg = estimate_utm_epsg(demand_df["lon"].mean(), demand_df["lat"].mean())
    demand_vals = demand_df["demand_value"].values
    total_demand = demand_vals.sum()
    n_demand = len(demand_df)

    existing_cov = compute_coverage_matrix(existing_df, demand_df, service_radius_km, epsg)
    covered = existing_cov.any(axis=0) if len(existing_cov) > 0 else np.zeros(n_demand, dtype=bool)
    covered_demand = demand_vals[covered].sum()
    unserved_demand = demand_vals[~covered].sum()

    # Weighted average distance for a CURRENT-NETWORK health check is
    # deliberately computed over ALL demand — not just the subset within
    # the configured service radius. This is a diagnostic ("how far, on
    # average, is my demand from its nearest existing facility today?"),
    # not a constrained optimization — silently dropping unserved demand
    # from the average would hide exactly the gap this check exists to
    # surface, and would make the number not reflect reality. This is the
    # genuine Σ(distance from nearest current facility × demand) / Σ(demand)
    # across every demand point.
    if existing_df is not None and len(existing_df) > 0:
        weighted_avg_distance_km = compute_weighted_avg_distance_km(demand_df, existing_df, epsg)
    else:
        weighted_avg_distance_km = None

    return {
        "total_demand": round(total_demand, 1),
        "baseline_covered_demand": round(covered_demand, 1),
        "baseline_coverage_pct": round(covered_demand / total_demand * 100, 1) if total_demand > 0 else 0,
        "final_covered_demand": round(covered_demand, 1),
        "final_coverage_pct": round(covered_demand / total_demand * 100, 1) if total_demand > 0 else 0,
        "unserved_demand": round(unserved_demand, 1),
        "unserved_demand_pct": round(unserved_demand / total_demand * 100, 1) if total_demand > 0 else 0,
        "sites_selected": 0,
        "target_met": False,
        "weighted_avg_distance_km": round(weighted_avg_distance_km, 2) if weighted_avg_distance_km is not None else None,
    }


# ---------------------------------------------------------------------------
# Site scoring — weighted multi-criteria evaluation of opened locations
# ---------------------------------------------------------------------------

DEFAULT_SCORING_CRITERIA = [
    {"key": "logistics_infra", "label": "Logistics Infrastructure", "default_weight": 20},
    {"key": "labor_availability", "label": "Labor Availability", "default_weight": 20},
    {"key": "warehouse_rental_cost", "label": "Warehouse Rental Cost (lower=better)", "default_weight": 15},
    {"key": "highway_proximity", "label": "Highway Proximity", "default_weight": 15},
    {"key": "airport_proximity", "label": "Airport Proximity", "default_weight": 10},
    {"key": "seaport_proximity", "label": "Seaport Proximity", "default_weight": 10},
    {"key": "power_reliability", "label": "Power/Utility Reliability", "default_weight": 5},
    {"key": "tax_incentives", "label": "Tax & Regulatory Incentives", "default_weight": 5},
]


# Real research (Sept 2026): warehouse rental rates, highway/airport/labor
# context for a few major US logistics hub regions, condensed to a 0-10 scale
# per criterion. Used to pre-fill the scoring table when a selected site's
# reverse-geocoded name matches one of these regions — saves the user from
# starting with a totally blank table, while being upfront that this is a
# starting reference, not a substitute for actual site diligence.
REFERENCE_SITE_SCORES = {
    "ashburn|loudoun|sterling|virginia|dulles": {
        "logistics_infra": 9, "labor_availability": 8, "warehouse_rental_cost": 3,
        "highway_proximity": 9, "airport_proximity": 10, "seaport_proximity": 3,
        "power_reliability": 7, "tax_incentives": 7,
        "_note": "Ashburn/Loudoun VA: ~$20/SF/yr industrial rent (LoopNet, Sep 2026) — "
                 "well above the ~$9.54/SF national average, reflecting Data Center Alley demand. "
                 "Adjacent to Dulles Intl Airport; Port of Virginia (Norfolk) is ~3hrs away.",
    },
    "dallas|fort worth|arlington.*texas|texas": {
        "logistics_infra": 9, "labor_availability": 8, "warehouse_rental_cost": 7,
        "highway_proximity": 9, "airport_proximity": 9, "seaport_proximity": 2,
        "power_reliability": 6, "tax_incentives": 8,
        "_note": "Dallas-Fort Worth: ~$9-12/SF/yr industrial rent (JLL/CommercialCafe, Q2 2026), "
                 "1.12B SF total inventory, one of the largest US industrial markets. DFW Airport is a "
                 "major air cargo hub; nearest seaport (Houston) is ~4hrs away.",
    },
    "columbus|new albany|hilliard|ohio": {
        "logistics_infra": 8, "labor_availability": 7, "warehouse_rental_cost": 7,
        "highway_proximity": 9, "airport_proximity": 8, "seaport_proximity": 2,
        "power_reliability": 7, "tax_incentives": 7,
        "_note": "Columbus OH: ~$10.45/SF/yr industrial rent (CommercialCafe/CityFeet, Aug 2026). "
                 "Sits at the I-70/I-71 junction; Rickenbacker Intl is a major dedicated air-cargo airport. "
                 "No direct seaport access.",
    },
}


def lookup_reference_scores(dc_name: str) -> dict | None:
    """Best-effort match of a reverse-geocoded DC name against the researched
    reference regions above. Returns None if no match — the user then fills
    the scores in manually."""
    import re
    if not dc_name:
        return None
    name_lower = dc_name.lower()
    for pattern, scores in REFERENCE_SITE_SCORES.items():
        if re.search(pattern, name_lower):
            return {k: v for k, v in scores.items() if k != "_note"}, scores.get("_note", "")
    return None



DATA_SOURCES = [
    {"region": "Ashburn / Loudoun County, VA", "criterion": "Warehouse Rental Cost",
     "value": "~$20/SF/yr (range $16-32)", "source": "LoopNet, CityFeet",
     "url": "https://www.loopnet.com/search/industrial-space/ashburn-va/for-lease/", "accessed": "Sep 2026"},
    {"region": "National (US) benchmark", "criterion": "Warehouse Rental Cost",
     "value": "$9.54/SF/yr NNN average", "source": "WarehousingCosts.com",
     "url": "https://warehousingcosts.com/guides/warehouse-lease-rates", "accessed": "Jun 2026"},
    {"region": "Ashburn / Loudoun County, VA", "criterion": "Airport / Highway Proximity",
     "value": "Adjacent to Dulles Intl Airport; near I-95/Rte 28/Dulles Greenway",
     "source": "IndustrialSpaces.net", "url": "https://industrialspaces.net/ashburn-va/warehouses-for-rent/",
     "accessed": "Sep 2026"},
    {"region": "Dallas-Fort Worth, TX", "criterion": "Warehouse Rental Cost",
     "value": "~$9-12/SF/yr NNN average", "source": "JLL Research, CommercialCafe",
     "url": "https://www.jll.com/en-us/insights/market-dynamics/dallas-fort-worth-industrial", "accessed": "Jul 2026"},
    {"region": "Dallas-Fort Worth, TX", "criterion": "Logistics Infrastructure",
     "value": "1.12B SF total industrial inventory; largest US market", "source": "LEE & Associates Dallas",
     "url": "https://leedallas.com/news/dallas-commercial-real-estate-industrial-market-report-spring-2026/",
     "accessed": "Apr 2026"},
    {"region": "Dallas-Fort Worth, TX", "criterion": "Labor Availability / Cost",
     "value": "Entry warehouse wage ~$17.30-18.50/hr", "source": "find3PLs.com",
     "url": "https://find3pls.com/blog/dfw-warehousing-costs-2026", "accessed": "Sep 2026"},
    {"region": "Columbus, OH", "criterion": "Warehouse Rental Cost",
     "value": "~$10.45/SF/yr average", "source": "CommercialCafe, CityFeet",
     "url": "https://www.commercialcafe.com/industrial/us/oh/columbus/", "accessed": "Aug 2026"},
    {"region": "Columbus, OH", "criterion": "Highway / Airport Proximity",
     "value": "I-70/I-71 junction; Rickenbacker Intl (dedicated air-cargo hub)",
     "source": "CityFeet, Columbus Warehouse Space", "url": "https://www.cityfeet.com/cont/columbus-oh/industrial-space-for-lease",
     "accessed": "Sep 2026"},
]


def build_data_sources_workbook() -> bytes:
    """Build an in-memory Excel workbook listing every external data source
    this app's reference scores are drawn from, so 'where did this number
    come from' is always one click away rather than buried in code comments."""
    import io as _io
    buf = _io.BytesIO()
    sources_df = pd.DataFrame(DATA_SOURCES)
    criteria_df = pd.DataFrame(DEFAULT_SCORING_CRITERIA)[["label", "default_weight"]]
    criteria_df.columns = ["Criterion", "Default Weight (%)"]
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        sources_df.to_excel(writer, sheet_name="Data Sources", index=False)
        criteria_df.to_excel(writer, sheet_name="Scoring Criteria", index=False)
    buf.seek(0)
    return buf.getvalue()


# Real-world knowledge: for each US state, the metro/city most established as
# a logistics/distribution hub (intermodal rail, highway junctions, major
# air-cargo capacity), used to give a concrete named-city recommendation
# instead of just flagging that a location scores low. Deliberately limited
# to states where there's a clearly dominant, well-known logistics hub.
STATE_LOGISTICS_HUBS = {
    "Illinois": {"city": "Joliet / Elwood, IL", "rationale": "BNSF and Union Pacific intermodal terminals "
                 "(CenterPoint Intermodal Center) just southwest of Chicago — one of the largest inland port "
                 "complexes in North America."},
    "Texas": {"city": "Dallas-Fort Worth, TX", "rationale": "I-35/I-20/I-30 junction, DFW Airport air-cargo hub, "
              "1B+ SF of industrial inventory."},
    "Ohio": {"city": "Columbus / Rickenbacker, OH", "rationale": "I-70/I-71 junction, Rickenbacker Intl "
             "dedicated air-cargo airport, within a day's drive of 50%+ of the US/Canada population."},
    "Georgia": {"city": "Atlanta, GA", "rationale": "Hartsfield-Jackson (busiest air-cargo airport in the "
                "Southeast), I-20/I-75/I-85 junction, close to the Port of Savannah."},
    "California": {"city": "Ontario / Inland Empire, CA", "rationale": "Adjacent to the Ports of LA/Long Beach, "
                   "major rail intermodal yards, the largest industrial submarket in the US."},
    "New Jersey": {"city": "Exit 8A Corridor (Cranbury/Monroe Twp), NJ", "rationale": "Central NJ Turnpike "
                   "corridor — the primary distribution hub for the NY/NJ metro and Port of NY/NJ."},
    "Pennsylvania": {"city": "Lehigh Valley, PA", "rationale": "I-78/I-81 junction, major East Coast distribution "
                     "hub within a day's drive of NYC, Philadelphia, and Baltimore."},
    "Tennessee": {"city": "Memphis, TN", "rationale": "FedEx global air hub, I-40/I-55 junction, Mississippi "
                  "River barge access."},
    "Indiana": {"city": "Indianapolis, IN", "rationale": "Crossroads of America — more interstate highways "
                "converge here than any other US city; FedEx's 2nd-largest air hub."},
    "Arizona": {"city": "Phoenix, AZ", "rationale": "I-10/I-17 junction, growing distribution hub for Southwest "
                "US/Mexico cross-border trade."},
    "Nevada": {"city": "Reno, NV", "rationale": "I-80 corridor, no state income tax, major West Coast "
               "distribution alternative to CA with lower costs."},
    "Washington": {"city": "Seattle-Tacoma, WA", "rationale": "Port of Seattle/Tacoma, primary Pacific Northwest "
                   "gateway for Asia-Pacific trade."},
    "Kentucky": {"city": "Louisville, KY", "rationale": "UPS Worldport global air hub, I-64/I-65/I-71 junction."},
    "Missouri": {"city": "Kansas City, MO", "rationale": "Largest rail freight hub in the US by tonnage, "
                 "central US location for national distribution."},
    "Virginia": {"city": "Ashburn / Loudoun County, VA", "rationale": "Dulles Intl Airport, dense fiber/data "
                 "infrastructure, I-95/Rte 28 access."},
    "North Carolina": {"city": "Charlotte, NC", "rationale": "I-77/I-85 junction, growing Southeast distribution "
                       "hub with strong intermodal rail."},
}


# Broader fallback: the best-known major metro in every US state, used when
# STATE_LOGISTICS_HUBS above has no specialized entry. Ensures a real named
# city is always recommended — never a vague "consider alternatives."
FALLBACK_STATE_MAJOR_CITY = {
    "Alabama": "Birmingham, AL", "Alaska": "Anchorage, AK", "Arkansas": "Little Rock, AR",
    "California": "Los Angeles / Inland Empire, CA", "Colorado": "Denver, CO", "Connecticut": "Hartford, CT",
    "Delaware": "Wilmington, DE", "Florida": "Jacksonville, FL", "Hawaii": "Honolulu, HI",
    "Idaho": "Boise, ID", "Iowa": "Des Moines, IA", "Kansas": "Wichita, KS",
    "Louisiana": "New Orleans, LA", "Maine": "Portland, ME", "Maryland": "Baltimore, MD",
    "Massachusetts": "Boston, MA", "Michigan": "Detroit, MI", "Minnesota": "Minneapolis, MN",
    "Mississippi": "Jackson, MS", "Montana": "Billings, MT", "Nebraska": "Omaha, NE",
    "New Hampshire": "Manchester, NH", "New Mexico": "Albuquerque, NM", "New York": "New York City, NY",
    "North Dakota": "Fargo, ND", "Oklahoma": "Oklahoma City, OK", "Oregon": "Portland, OR",
    "Rhode Island": "Providence, RI", "South Carolina": "Charleston, SC", "South Dakota": "Sioux Falls, SD",
    "Utah": "Salt Lake City, UT", "Vermont": "Burlington, VT", "West Virginia": "Charleston, WV",
    "Wisconsin": "Milwaukee, WI", "Wyoming": "Cheyenne, WY",
}

# Non-US fallback: the dominant logistics/trade hub for each major country,
# used when a recommended site falls outside the US (where the two dicts
# above have no coverage at all). Real, well-established hubs — major sea
# ports, free trade zones, or primary distribution/industrial centers —
# not live web research, but a genuine geography-based knowledge base so
# recommendations are never just silent for non-US locations.
COUNTRY_LOGISTICS_HUBS = {
    "India": {"city": "Mumbai / Navi Mumbai (JNPT), India", "rationale": "Jawaharlal Nehru Port (JNPT) handles "
              "over half of India's containerized cargo; strong road/rail/air connectivity to the rest of the country."},
    "United Kingdom": {"city": "Felixstowe / London Gateway, UK", "rationale": "Felixstowe is the UK's busiest "
                        "container port; London Gateway offers a modern deep-water alternative with direct motorway access."},
    "Germany": {"city": "Hamburg / Duisburg, Germany", "rationale": "Hamburg is Germany's largest port and a "
                "major EU gateway; Duisburg is the world's largest inland port, key for European rail freight."},
    "Netherlands": {"city": "Rotterdam, Netherlands", "rationale": "Europe's largest seaport, with extensive "
                    "rail/barge/road links deep into the EU — the primary gateway for continental distribution."},
    "France": {"city": "Le Havre / Paris region, France", "rationale": "Le Havre is France's leading container "
               "port; the Paris region offers the largest consumer market and logistics real estate concentration."},
    "China": {"city": "Shanghai / Shenzhen, China", "rationale": "Shanghai is the world's busiest container port; "
              "Shenzhen anchors the Pearl River Delta manufacturing and export corridor."},
    "Japan": {"city": "Tokyo / Yokohama, Japan", "rationale": "The Tokyo-Yokohama port complex is Japan's largest, "
              "with dense rail and highway links across the Kanto region."},
    "South Korea": {"city": "Busan, South Korea", "rationale": "Busan is one of the world's top-10 container "
                    "ports and South Korea's primary trade gateway."},
    "Singapore": {"city": "Singapore", "rationale": "One of the world's busiest transshipment hubs, with "
                  "world-class port infrastructure and free-trade-zone advantages."},
    "United Arab Emirates": {"city": "Dubai (Jebel Ali), UAE", "rationale": "Jebel Ali is the largest man-made "
                              "harbor and a major transshipment hub for the Middle East/Africa/South Asia corridor."},
    "Australia": {"city": "Melbourne / Sydney, Australia", "rationale": "Melbourne is Australia's busiest "
                  "container port; combined with Sydney, these two cover most of the domestic population."},
    "Canada": {"city": "Vancouver / Toronto, Canada", "rationale": "Vancouver is Canada's largest port (key "
               "Pacific gateway); Toronto anchors the largest domestic consumer market and rail network."},
    "Mexico": {"city": "Manzanillo / Mexico City, Mexico", "rationale": "Manzanillo is Mexico's busiest Pacific "
               "port; Mexico City region offers the largest population and cross-border rail/road access to the US."},
    "Brazil": {"city": "Santos, Brazil", "rationale": "Latin America's busiest port, serving the São Paulo "
               "industrial region — the dominant gateway for Brazilian trade."},
    "South Africa": {"city": "Durban, South Africa", "rationale": "Sub-Saharan Africa's busiest container port "
                      "and the primary gateway for southern African trade."},
    "Italy": {"city": "Genoa / Milan, Italy", "rationale": "Genoa is Italy's leading seaport; Milan anchors the "
              "country's largest industrial and consumer market in the north."},
    "Spain": {"city": "Valencia / Algeciras, Spain", "rationale": "Valencia is Spain's busiest container port; "
              "Algeciras is a major Mediterranean transshipment hub near the Strait of Gibraltar."},
    "Vietnam": {"city": "Ho Chi Minh City, Vietnam", "rationale": "Vietnam's largest commercial hub and the "
                "center of its rapidly growing manufacturing/export sector."},
    "Indonesia": {"city": "Jakarta (Tanjung Priok), Indonesia", "rationale": "Indonesia's busiest port, serving "
                  "the country's largest population center and industrial base."},
    "Thailand": {"city": "Bangkok / Laem Chabang, Thailand", "rationale": "Laem Chabang is Thailand's main deep-sea "
                 "port; Bangkok anchors the country's largest consumer and manufacturing base."},
    "Turkey": {"city": "Istanbul, Turkey", "rationale": "Turkey's largest city and trade hub, straddling Europe "
               "and Asia with major port and air cargo infrastructure."},
    "Saudi Arabia": {"city": "Jeddah, Saudi Arabia", "rationale": "Saudi Arabia's primary Red Sea port and "
                      "commercial gateway for the western region."},
    "Poland": {"city": "Warsaw / Gdańsk, Poland", "rationale": "Gdańsk is Poland's largest Baltic port; Warsaw "
               "anchors the country's largest logistics and distribution market."},
}


def suggest_best_next_location(dc_name: str, scores_row: pd.Series, weights: dict,
                                cand_df: pd.DataFrame, selected_df: pd.DataFrame,
                                service_radius_km: float, state: str | None = None,
                                country: str | None = None) -> str:
    """For a site scoring below the quality bar, produce a concrete,
    data-grounded suggestion — never a vague "consider alternatives": lists
    EVERY criterion scoring below 5 (not just the single weakest one), and
    always recommends a real named city (a specialized US-state logistics
    hub where known, else the US state's best-known major metro, else —
    for non-US locations — the country's dominant logistics/trade hub),
    with a holistic explanation of why it's the better fit across all the
    flagged gaps — plus, for a rental-cost weakness, checks the run's own
    candidate pool for a genuinely cheaper nearby alternative."""
    criterion_labels = {c["key"]: c["label"] for c in DEFAULT_SCORING_CRITERIA}
    crit_scores = {k: scores_row.get(k, 5) for k in weights if k in criterion_labels}
    if not crit_scores:
        return f"**{dc_name}**: not enough scoring data to generate a suggestion."

    # Every criterion scoring below 5 — not just the single weakest — sorted worst-first
    weak_criteria = sorted(
        [(criterion_labels.get(k, k), v) for k, v in crit_scores.items() if v < 5],
        key=lambda x: x[1]
    )
    if not weak_criteria:
        # Nothing below 5 individually, but the weighted average still fell below 7 —
        # fall back to naming just the single lowest-scoring criterion.
        weakest_key = min(crit_scores, key=crit_scores.get)
        weak_criteria = [(criterion_labels.get(weakest_key, weakest_key), crit_scores[weakest_key])]

    weak_list_str = ", ".join(f"**{label}** ({val}/10)" for label, val in weak_criteria)
    if len(weak_criteria) == 1:
        suggestion = f"**{dc_name}** scores low on {weak_list_str}. "
    else:
        suggestion = f"**{dc_name}** falls short on {len(weak_criteria)} criteria: {weak_list_str}. "

    hub = STATE_LOGISTICS_HUBS.get(state) if state else None
    weak_labels_lower = [label.lower() for label, _ in weak_criteria]
    rec_city, rec_rationale = None, None
    if hub and hub["city"].split(",")[0].split(" / ")[0].strip().lower() not in dc_name.lower():
        rec_city, rec_rationale = hub["city"], hub["rationale"]
    elif state and state in FALLBACK_STATE_MAJOR_CITY and \
            FALLBACK_STATE_MAJOR_CITY[state].split(",")[0].split(" / ")[0].strip().lower() not in dc_name.lower():
        rec_city = FALLBACK_STATE_MAJOR_CITY[state]
        rec_rationale = "the state's largest metro, generally offering more developed transport and logistics infrastructure"
    elif country and country in COUNTRY_LOGISTICS_HUBS:
        country_hub = COUNTRY_LOGISTICS_HUBS[country]
        if country_hub["city"].split(",")[0].split(" / ")[0].strip().lower() not in dc_name.lower():
            rec_city, rec_rationale = country_hub["city"], country_hub["rationale"]

    if rec_city:
        rationale_clean = rec_rationale.rstrip(".")
        suggestion += (f"Considering all {len(crit_scores)} scoring criteria together — not just one weak spot — "
                       f"**{rec_city}** is the recommended alternative: {rationale_clean}. This addresses the "
                       f"{', '.join(l for l, _ in weak_criteria)} gap{'s' if len(weak_criteria) > 1 else ''} "
                       f"flagged above better than {dc_name}. ")

    if "warehouse rental cost" in weak_labels_lower and cand_df is not None and len(cand_df) > 0 \
            and selected_df is not None and "lat" in selected_df.columns:
        this_site = selected_df[selected_df["dc_name"] == dc_name]
        if len(this_site) > 0:
            site_lat, site_lon = this_site.iloc[0]["lat"], this_site.iloc[0]["lon"]
            this_cost = this_site.iloc[0].get("monthly_lease_cost", None)
            nearby = cand_df.copy()
            nearby["_dist_deg"] = np.sqrt((nearby["lat"] - site_lat) ** 2 + (nearby["lon"] - site_lon) ** 2)
            nearby_radius_deg = (service_radius_km * 1.5) / 111.0
            nearby = nearby[nearby["_dist_deg"] <= nearby_radius_deg]
            if this_cost is not None and len(nearby) > 0 and nearby["monthly_lease_cost"].min() < this_cost:
                cheaper = nearby.loc[nearby["monthly_lease_cost"].idxmin()]
                savings_pct = (1 - cheaper["monthly_lease_cost"] / this_cost) * 100
                suggestion += (f"Within the current candidate pool, site {cheaper['site_id']} "
                               f"(~{cheaper['dist_from_center_km']:.0f} km from cluster center) has an estimated "
                               f"lease cost ~{savings_pct:.0f}% lower — worth evaluating as a nearer-term alternative.")

    return suggestion


# ---------------------------------------------------------------------------
# GenAI copilot — basefile transformation + scenario comparison assistant
#
# Both features require the user's OWN Anthropic API key (entered in the
# sidebar, kept in session only, never written to disk). This app has no
# bundled key — that would mean shipping Anthropic credentials inside code
# handed to users, which is never appropriate. Without a key, these features
# show a clear message rather than silently failing.
# ---------------------------------------------------------------------------

def call_claude_api(api_key: str, system_prompt: str, user_message: str,
                     max_tokens: int = 1500, model: str = "claude-sonnet-4-5-20250929") -> tuple[str | None, str | None]:
    """Call the Anthropic Messages API directly over HTTPS (no SDK dependency).
    Returns (response_text, error_message) — exactly one will be None."""
    import requests
    if not api_key or not api_key.strip():
        return None, "No API key provided."
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key.strip(),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_message}],
            },
            timeout=30,
        )
        if resp.status_code == 401:
            return None, "API key was rejected (401) — check it's correct and active."
        if resp.status_code != 200:
            return None, f"API error (HTTP {resp.status_code}): {resp.text[:300]}"
        data = resp.json()
        text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
        return "\n".join(text_blocks), None
    except requests.exceptions.Timeout:
        return None, "Request timed out — try again."
    except Exception as e:
        return None, f"Request failed: {e}"


TARGET_SCHEMAS = {
    "Customer Demand": ["city", "country", "company_code", "product_id", "demand_value", "lat", "lon"],
    "Products": ["product_id", "product_name"],
    "Existing Facilities": ["facility_id", "facility_name", "city", "country", "lat", "lon"],
}


def build_mapping_prompt(target_table: str, raw_columns: list, sample_rows: str,
                          user_instruction: str) -> tuple[str, str]:
    """Build the system + user prompt asking Claude to propose a column
    mapping from a raw basefile (shipments/transactions/forecast) onto one
    of this app's required table schemas. Returns (system_prompt, user_msg)."""
    target_cols = TARGET_SCHEMAS.get(target_table, [])
    system_prompt = (
        "You are a data-mapping assistant for a supply-chain network-design tool. "
        "Given a raw file's column names, a few sample rows, and the user's instructions, "
        "propose a mapping from the raw columns onto a required target schema. "
        "Respond with ONLY a JSON object, no other text, no markdown fences, in this exact shape:\n"
        '{"mapping": {"<target_column>": "<raw_column_or_expression>", ...}, '
        '"notes": "<one or two sentences on any assumptions made>", '
        '"unmapped_target_columns": ["<target col with no good source>", ...]}\n'
        "For a target column, the value can be a raw column name to copy directly, or a simple "
        "pandas-eval-safe expression using raw column names (e.g. \"weight_kg * 2.20462\" to convert "
        "to pounds). If a target column has no reasonable source, list it in unmapped_target_columns "
        "and omit it from mapping."
    )
    user_msg = (
        f"Target table: {target_table}\n"
        f"Required target columns: {target_cols}\n"
        f"Raw file columns: {raw_columns}\n"
        f"Sample rows (as text):\n{sample_rows}\n"
        f"User's instructions: {user_instruction or '(none given — infer the best mapping from column names)'}"
    )
    return system_prompt, user_msg


def parse_mapping_response(response_text: str) -> tuple[dict | None, str | None]:
    """Parse the JSON mapping Claude returns. Returns (mapping_dict, error)."""
    import json
    import re
    if not response_text:
        return None, "Empty response from the model."
    cleaned = response_text.strip()
    cleaned = re.sub(r"^```json\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
        if "mapping" not in parsed:
            return None, "Response was valid JSON but missing the 'mapping' key."
        return parsed, None
    except json.JSONDecodeError as e:
        return None, f"Could not parse the model's response as JSON: {e}"


def apply_column_mapping(raw_df: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame | None, str | None]:
    """Apply a {target_col: raw_col_or_expression} mapping to produce the
    transformed target table. Expressions are evaluated with pandas.eval
    against the raw dataframe's columns only (no arbitrary code execution)."""
    result = pd.DataFrame(index=raw_df.index)
    for target_col, source in mapping.items():
        try:
            if source in raw_df.columns:
                result[target_col] = raw_df[source]
            else:
                result[target_col] = raw_df.eval(source)
        except Exception as e:
            return None, f"Could not compute column '{target_col}' from '{source}': {e}"
    return result, None


COMPARISON_INTENTS = ["coverage_comparison", "distance_comparison", "sites_comparison", "summary_table"]


def classify_comparison_intent(api_key: str, scenario_names: list, user_question: str) -> tuple[str | None, str | None]:
    """Use Claude to classify a free-text comparison question into one of a
    small, SAFE set of supported chart/table types, which the app then
    renders itself with its own plotting code — never executing arbitrary
    model-generated code."""
    system_prompt = (
        "You classify a user's question about comparing supply-chain network scenarios into exactly one "
        f"of these categories: {COMPARISON_INTENTS}. "
        "coverage_comparison = they want to compare demand coverage % across scenarios. "
        "distance_comparison = they want to compare weighted average service distance across scenarios. "
        "sites_comparison = they want to compare number of sites opened / cost across scenarios. "
        "summary_table = anything else, or a general overview request. "
        "Respond with ONLY the category name, nothing else."
    )
    user_msg = f"Saved scenarios: {scenario_names}\nQuestion: {user_question}"
    response, error = call_claude_api(api_key, system_prompt, user_msg, max_tokens=20)
    if error:
        return None, error
    intent = response.strip().lower()
    if intent not in COMPARISON_INTENTS:
        intent = "summary_table"
    return intent, None


def compute_weighted_scores(scores_df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Combine per-criterion scores (0-10 scale, already entered by the user
    or pre-filled from research) with user-set weights (must sum to 100)
    into a single weighted score per site, 0-10 scale.

    `scores_df` must have a 'dc_name' column plus one numeric column per
    criterion key in `weights`. Missing/non-numeric cells are treated as 0.
    """
    result = scores_df.copy()
    total_weight = sum(weights.values())
    if total_weight <= 0:
        result["weighted_score"] = 0.0
        return result

    weighted_sum = pd.Series(0.0, index=result.index)
    for key, weight in weights.items():
        if key in result.columns:
            col_numeric = pd.to_numeric(result[key], errors="coerce").fillna(0)
            weighted_sum += col_numeric * (weight / total_weight)
    result["weighted_score"] = weighted_sum.round(2)
    return result.sort_values("weighted_score", ascending=False).reset_index(drop=True)


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

    expected_cols = list(cand_df.columns) + ["incremental_demand_covered", "cumulative_covered_demand", "cumulative_coverage_pct"]
    selected_df = pd.DataFrame(selected_rows, columns=expected_cols) if selected_rows else pd.DataFrame(columns=expected_cols)
    final_covered_demand = demand_vals[covered].sum()

    # Weighted average service distance — measured ONLY over demand actually
    # served within the service radius. The radius is a real constraint:
    # a demand point beyond every facility's radius is "unserved," and
    # including its distance in this average would misrepresent service
    # quality for the network as actually configured.
    facility_frames = []
    if existing_df is not None and len(existing_df) > 0:
        facility_frames.append(existing_df[["lat", "lon"]])
    if len(selected_df) > 0:
        facility_frames.append(selected_df[["lat", "lon"]])
    combined_facilities = pd.concat(facility_frames, ignore_index=True) if facility_frames else None

    if combined_facilities is not None and covered.any():
        served_demand_df = demand_df[covered].reset_index(drop=True)
        weighted_avg_distance_km = compute_weighted_avg_distance_km(served_demand_df, combined_facilities, epsg)
    else:
        weighted_avg_distance_km = None

    unserved_demand = demand_vals[~covered].sum()

    summary = {
        "total_demand": round(total_demand, 1),
        "baseline_covered_demand": round(baseline_covered_demand, 1),
        "baseline_coverage_pct": round(baseline_covered_demand / total_demand * 100, 1) if total_demand > 0 else 0,
        "final_covered_demand": round(final_covered_demand, 1),
        "final_coverage_pct": round(final_covered_demand / total_demand * 100, 1) if total_demand > 0 else 0,
        "unserved_demand": round(unserved_demand, 1),
        "unserved_demand_pct": round(unserved_demand / total_demand * 100, 1) if total_demand > 0 else 0,
        "sites_selected": len(selected_idx),
        "target_met": (mode == "service_target" and
                        (final_covered_demand / total_demand * 100 if total_demand > 0 else 0) >= target_pct),
        "weighted_avg_distance_km": round(weighted_avg_distance_km, 2) if weighted_avg_distance_km is not None else None,
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
    required_cols = {"product_id", "product_name"}
    missing = required_cols - set(df.columns)
    if missing:
        return False, f"Products table missing columns: {', '.join(sorted(missing))}"
    if df[["product_id", "product_name"]].isnull().any().any():
        return False, "Found empty values in product_id or product_name."
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
