import io
import time

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from optimizer import (
    compute_service_radius,
    coordinates_ready,
    generate_candidate_sites,
    geocode_locations,
    greedy_select_sites,
    needs_geocoding,
    validate_demand_df,
    validate_existing_df,
    validate_products_df,
)

st.set_page_config(page_title="Supply Chain Design by AM", layout="wide", page_icon="🚚")

# ---------- Theme ----------
st.markdown("""
<style>
.main-header {
    background-color: #0B3D91;
    color: #F5C518;
    padding: 18px 24px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.main-header h1 { margin: 0; font-size: 28px; }
.main-header p { margin: 4px 0 0 0; color: #EAF2FB; font-size: 14px; }
div.stButton > button[kind="primary"] {
    background-color: #F5C518;
    color: #0B3D91;
    border: none;
    font-weight: 600;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #d9ac00;
    color: #FFFFFF;
}
</style>
<div class="main-header">
    <h1>Supply Chain Design by AM</h1>
    <p>Greenfield facility location optimizer — customer demand, products, and existing network</p>
</div>
""", unsafe_allow_html=True)

# ---------- Empty-table schemas (no forced sample data) ----------
EMPTY_PRODUCTS = pd.DataFrame(columns=["product_id", "product_name"])
EMPTY_DEMAND = pd.DataFrame(columns=["city", "country", "company_code", "product_id", "demand_value", "lat", "lon"])
EMPTY_EXISTING = pd.DataFrame(columns=["facility_id", "facility_name", "city", "country", "lat", "lon"])

UOM_OPTIONS = ["Orders", "Quantity (units)", "Weight (kg)", "Weight (lbs)",
               "Volume (m3)", "Volume (ft3)", "Pallets", "Containers", "Other (specify below)"]

# ---------- Session state ----------
defaults = {
    "view": "input",
    "demand_df": EMPTY_DEMAND.copy(),
    "products_df": EMPTY_PRODUCTS.copy(),
    "existing_df": EMPTY_EXISTING.copy(),
    "include_existing": True,
    "model_uom": "Orders",
    "model_uom_custom": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _sample_demand():
    rng = np.random.default_rng(42)
    cluster_centers = [
        (13.0827, 80.2707), (13.0067, 80.2206), (13.1500, 80.2101),
        (12.9698, 80.2200), (13.0500, 80.2900),
    ]
    product_ids = ["P001", "P002", "P003"]
    company_codes = ["CC-100", "CC-200"]
    rows, rid = [], 1
    for clat, clon in cluster_centers:
        lats = rng.normal(clat, 0.04, 8)
        lons = rng.normal(clon, 0.04, 8)
        vols = rng.gamma(2.0, 15, 8)
        for lat, lon, vol in zip(lats, lons, vols):
            rows.append({
                "city": "Chennai", "country": "India",
                "company_code": company_codes[rid % len(company_codes)],
                "product_id": product_ids[rid % len(product_ids)],
                "demand_value": round(float(vol), 1),
                "lat": round(float(lat), 5), "lon": round(float(lon), 5),
            })
            rid += 1
    return pd.DataFrame(rows)


# ---------- Sidebar: model-wide settings + optimization controls ----------
with st.sidebar:
    st.header("Model settings")

    uom_choice = st.selectbox("Unit of measure for this model", UOM_OPTIONS,
                               index=UOM_OPTIONS.index(st.session_state.model_uom)
                               if st.session_state.model_uom in UOM_OPTIONS else 0,
                               help="Every demand_value in this model is measured in this single unit — "
                                    "orders, weight, volume, etc. Mixing units breaks the math, so there's "
                                    "one unit for the whole model, not one per product.")
    st.session_state.model_uom = uom_choice
    if uom_choice == "Other (specify below)":
        st.session_state.model_uom_custom = st.text_input("Custom unit name", value=st.session_state.model_uom_custom)
    active_uom = st.session_state.model_uom_custom if uom_choice == "Other (specify below)" and st.session_state.model_uom_custom else uom_choice

    st.divider()
    st.header("Service coverage assumptions")

    col_a, col_b = st.columns(2)
    with col_a:
        service_time_value = st.number_input("Desired service time", min_value=0.1, value=1.0, step=0.5)
    with col_b:
        service_time_unit = st.selectbox("Unit", ["Days", "Hours"])

    miles_per_day = st.number_input("Last-mile daily travel capacity (miles/day)", min_value=50, max_value=1000,
                                     value=400, step=50,
                                     help="Assumption: how far a delivery truck can realistically travel in one day. "
                                          "This, combined with your desired service time, determines the effective "
                                          "service radius.")

    service_radius_km, service_radius_miles = compute_service_radius(service_time_value, service_time_unit, miles_per_day)
    st.caption(f"→ Effective service radius: **{service_radius_km:,.0f} km** ({service_radius_miles:,.0f} miles)")

    st.session_state.include_existing = st.checkbox(
        "Include existing sites in this run?", value=st.session_state.include_existing,
        help="Uncheck to run a pure greenfield analysis, ignoring the Existing Facilities table entirely."
    )

    st.subheader("How much demand to serve")
    opt_mode = st.radio("Choose optimization mode",
                         options=["Number of new sites", "Service coverage target (%)"],
                         label_visibility="collapsed")

    if opt_mode == "Number of new sites":
        num_sites = st.number_input("Number of new sites to open", min_value=1, max_value=50, value=5, step=1)
        target_pct, max_sites_cap, mode_key = None, None, "num_sites"
    else:
        target_pct = st.number_input("Target % of demand to serve within the service time above", min_value=1.0,
                                      max_value=100.0, value=80.0, step=1.0)
        max_sites_cap = st.number_input("Max new sites allowed (safety cap)", min_value=1, max_value=50, value=15, step=1)
        num_sites, mode_key = None, "service_target"

# =========================================================================
# INPUT VIEW
# =========================================================================
if st.session_state.view == "input":

    st.subheader("1. Products")
    st.caption(f"Every product's demand is measured in **{active_uom}** (set in the sidebar). "
               "This is a **required** table.")
    p_col1, p_col2 = st.columns([3, 1])
    with p_col1:
        products_upload = st.file_uploader("Upload products CSV", type=["csv"], key="products_upload")
    with p_col2:
        st.write("")
        st.write("")
        if st.button("Load sample", key="load_sample_products"):
            st.session_state.products_df = pd.DataFrame({
                "product_id": ["P001", "P002", "P003"],
                "product_name": ["Standard Parcel", "Bulk Pallet", "Cold Chain Item"],
            })
            st.rerun()

    if products_upload is not None and st.session_state.get("_products_upload_id") != products_upload.file_id:
        st.session_state.products_df = pd.read_csv(products_upload)
        st.session_state["_products_upload_id"] = products_upload.file_id

    st.session_state.products_df = st.data_editor(
        st.session_state.products_df, num_rows="dynamic", width="stretch", key="products_editor"
    )
    valid_p, msg_p = validate_products_df(st.session_state.products_df)

    st.divider()

    st.subheader("2. Customer Demand")
    st.caption(f"One row per customer / city. `demand_value` is measured in **{active_uom}**. "
               "This is a **required** table.")
    d_col1, d_col2 = st.columns([3, 1])
    with d_col1:
        demand_upload = st.file_uploader("Upload customer demand CSV", type=["csv"], key="demand_upload")
    with d_col2:
        st.write("")
        st.write("")
        if st.button("Load sample", key="load_sample_demand"):
            st.session_state.demand_df = _sample_demand()
            if st.session_state.products_df.empty:
                st.session_state.products_df = pd.DataFrame({
                    "product_id": ["P001", "P002", "P003"],
                    "product_name": ["Standard Parcel", "Bulk Pallet", "Cold Chain Item"],
                })
            st.rerun()

    if demand_upload is not None and st.session_state.get("_demand_upload_id") != demand_upload.file_id:
        st.session_state.demand_df = pd.read_csv(demand_upload)
        st.session_state["_demand_upload_id"] = demand_upload.file_id

    product_options = st.session_state.products_df["product_id"].dropna().unique().tolist() \
        if "product_id" in st.session_state.products_df.columns else []
    column_config = {
        "demand_value": st.column_config.NumberColumn("demand_value", min_value=0.0, format="%.1f"),
        "lat": st.column_config.NumberColumn("lat", min_value=-90.0, max_value=90.0, format="%.5f"),
        "lon": st.column_config.NumberColumn("lon", min_value=-180.0, max_value=180.0, format="%.5f"),
    }
    if product_options:
        column_config["product_id"] = st.column_config.SelectboxColumn("product_id", options=product_options)

    st.session_state.demand_df = st.data_editor(
        st.session_state.demand_df, num_rows="dynamic", width="stretch",
        key="demand_editor", column_config=column_config,
    )
    valid_d, msg_d = validate_demand_df(st.session_state.demand_df)

    st.divider()

    st.subheader("3. Existing Facilities")
    include_note = "included in" if st.session_state.include_existing else "**excluded from** (toggle in sidebar to include)"
    st.caption(f"Facilities already open. Currently {include_note} this optimization run. Optional table.")
    e_col1, e_col2 = st.columns([3, 1])
    with e_col1:
        existing_upload = st.file_uploader("Upload existing facilities CSV", type=["csv"], key="existing_upload")
    with e_col2:
        st.write("")
        st.write("")
        if st.button("Load sample", key="load_sample_existing"):
            st.session_state.existing_df = pd.DataFrame({
                "facility_id": ["EX-001"], "facility_name": ["Existing DC - City Center"],
                "city": ["Chennai"], "country": ["India"], "lat": [13.0827], "lon": [80.2707],
            })
            st.rerun()

    if existing_upload is not None and st.session_state.get("_existing_upload_id") != existing_upload.file_id:
        st.session_state.existing_df = pd.read_csv(existing_upload)
        st.session_state["_existing_upload_id"] = existing_upload.file_id

    st.session_state.existing_df = st.data_editor(
        st.session_state.existing_df, num_rows="dynamic", width="stretch", key="existing_editor",
        column_config={
            "lat": st.column_config.NumberColumn("lat", min_value=-90.0, max_value=90.0, format="%.5f"),
            "lon": st.column_config.NumberColumn("lon", min_value=-180.0, max_value=180.0, format="%.5f"),
        },
    )
    valid_e, msg_e = validate_existing_df(st.session_state.existing_df)

    st.divider()

    # ---------- Geocoding ----------
    demand_needs_geo = needs_geocoding(st.session_state.demand_df)
    existing_needs_geo = st.session_state.include_existing and needs_geocoding(st.session_state.existing_df)

    if demand_needs_geo or existing_needs_geo:
        st.info("Some rows are missing lat/lon coordinates. If you don't know them, geocode automatically from City + Country below.")
        if st.button("🌍 Geocode missing locations", key="geocode_btn"):
            with st.spinner("Looking up coordinates from City + Country (this can take a moment)..."):
                if demand_needs_geo:
                    st.session_state.demand_df, failed_d = geocode_locations(st.session_state.demand_df)
                else:
                    failed_d = []
                if existing_needs_geo:
                    st.session_state.existing_df, failed_e = geocode_locations(st.session_state.existing_df)
                else:
                    failed_e = []
            n_failed = len(failed_d) + len(failed_e)
            if n_failed == 0:
                st.success("Geocoding complete — all rows resolved.")
            else:
                st.warning(f"Geocoding complete, but {n_failed} row(s) could not be resolved. "
                           f"Check the city/country spelling for those rows, or fill in lat/lon manually.")
            st.rerun()

    # ---------- Validation summary before Run ----------
    coords_d_ok, coords_d_msg = coordinates_ready(st.session_state.demand_df)
    coords_e_ok, coords_e_msg = (True, "") if not st.session_state.include_existing else coordinates_ready(st.session_state.existing_df)

    problems = []
    if not valid_p:
        problems.append(f"**Products:** {msg_p}")
    if not valid_d:
        problems.append(f"**Customer Demand:** {msg_d}")
    if not coords_d_ok:
        problems.append(f"**Customer Demand coordinates:** {coords_d_msg}")
    if not valid_e:
        problems.append(f"**Existing Facilities:** {msg_e}")
    if not coords_e_ok:
        problems.append(f"**Existing Facilities coordinates:** {coords_e_msg}")

    ready_to_run = len(problems) == 0

    if problems:
        st.error("Fix the following before you can run the optimizer:\n\n" + "\n".join(f"- {p}" for p in problems))

    run_button = st.button("▶ Run optimization", type="primary", disabled=not ready_to_run, width="stretch")

    if run_button and ready_to_run:
        placeholder = st.empty()
        placeholder.info("⏳ Model is running...")

        start_t = time.perf_counter()

        demand_df = st.session_state.demand_df.copy()
        demand_df["demand_value"] = pd.to_numeric(demand_df["demand_value"], errors="coerce")
        existing_df = st.session_state.existing_df.copy() if st.session_state.include_existing else pd.DataFrame(
            columns=["facility_id", "city", "country", "lat", "lon"]
        )

        from optimizer import auto_grid_step_deg
        grid_step = auto_grid_step_deg(service_radius_km)
        cand_df = generate_candidate_sites(demand_df, grid_step_deg=grid_step)
        selected_df, summary = greedy_select_sites(
            demand_df, cand_df, existing_df,
            service_radius_km=service_radius_km,
            mode=mode_key,
            num_sites=num_sites if num_sites else 5,
            target_pct=target_pct if target_pct else 80.0,
            max_sites=max_sites_cap if max_sites_cap else 25,
        )

        elapsed = time.perf_counter() - start_t
        placeholder.empty()

        st.session_state["selected_df"] = selected_df
        st.session_state["opt_summary"] = summary
        st.session_state["run_demand_df"] = demand_df
        st.session_state["run_existing_df"] = existing_df
        st.session_state["run_service_radius_km"] = service_radius_km
        st.session_state["run_service_radius_miles"] = service_radius_miles
        st.session_state["run_time"] = elapsed
        st.session_state["run_mode_key"] = mode_key
        st.session_state["run_target_pct"] = target_pct
        st.session_state["run_max_sites_cap"] = max_sites_cap
        st.session_state["run_uom"] = active_uom
        st.session_state["view"] = "results"
        st.rerun()

# =========================================================================
# RESULTS VIEW
# =========================================================================
else:
    if st.button("⬅ Back to Input"):
        st.session_state["view"] = "input"
        st.rerun()

    selected_df = st.session_state["selected_df"]
    summary = st.session_state["opt_summary"]
    run_demand_df = st.session_state["run_demand_df"]
    run_existing_df = st.session_state["run_existing_df"]
    run_service_radius_km = st.session_state["run_service_radius_km"]
    run_service_radius_miles = st.session_state["run_service_radius_miles"]
    run_mode_key = st.session_state["run_mode_key"]
    run_target_pct = st.session_state["run_target_pct"]
    run_max_sites_cap = st.session_state["run_max_sites_cap"]
    run_uom = st.session_state["run_uom"]

    st.success(f"✅ Model run complete — run time: {st.session_state['run_time']:.2f} secs")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Sites opened", summary["sites_selected"])
    m2.metric("Baseline coverage", f"{summary['baseline_coverage_pct']}%")
    m3.metric("Final coverage", f"{summary['final_coverage_pct']}%")
    m4.metric(f"Total demand ({run_uom})", f"{summary['total_demand']:.0f}")
    wavg = summary.get("weighted_avg_distance_km")
    if wavg is not None:
        wavg_miles = wavg * 0.621371
        m5.metric("Weighted avg service distance", f"{wavg:.0f} km", help=f"≈ {wavg_miles:.0f} miles. "
                  "Demand-weighted average distance from each customer to its nearest open facility "
                  "(Σ distance × demand share) — lower is better service.")
    else:
        m5.metric("Weighted avg service distance", "—")

    if run_mode_key == "service_target":
        if summary.get("target_met"):
            st.success(f"Target of {run_target_pct}% coverage reached with {summary['sites_selected']} new site(s), "
                       f"within a {run_service_radius_km:,.0f} km ({run_service_radius_miles:,.0f} mi) service radius.")
        else:
            st.warning(f"Target of {run_target_pct}% coverage NOT reached within the {run_max_sites_cap}-site cap — "
                       f"reached {summary['final_coverage_pct']}%. Go back and try raising the site cap, the service "
                       f"time, or the daily travel capacity assumption.")

    if summary["sites_selected"] == 0:
        st.info("No new sites were needed — existing facilities (or the current network) already meet the "
                "coverage target within this service radius. Try a smaller service radius/time, a higher "
                "coverage target, or unchecking 'Include existing sites' to see a pure greenfield result.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Map — demand bubbles sized by {run_uom}")
        center_lat, center_lon = run_demand_df["lat"].mean(), run_demand_df["lon"].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="OpenStreetMap")

        max_demand = run_demand_df["demand_value"].max()
        max_demand = max_demand if max_demand and max_demand > 0 else 1

        demand_layer = folium.FeatureGroup(name="Demand (bubble size = volume)")
        for _, row in run_demand_df.iterrows():
            bubble_radius = 4 + (row["demand_value"] / max_demand) * 20
            folium.CircleMarker(
                [row["lat"], row["lon"]], radius=bubble_radius, color="#0B36C0",
                fill=True, fill_color="#0B36C0", fill_opacity=0.45, weight=1,
                popup=f"{row.get('city', '')}: {row['demand_value']:.0f} {run_uom} ({row.get('product_id', '')})",
            ).add_to(demand_layer)
        demand_layer.add_to(m)

        if run_existing_df is not None and len(run_existing_df) > 0:
            existing_layer = folium.FeatureGroup(name="Existing facilities")
            for _, row in run_existing_df.iterrows():
                folium.Marker(
                    [row["lat"], row["lon"]],
                    icon=folium.Icon(color="blue", icon="industry", prefix="fa"),
                    popup=str(row.get("facility_name", row.get("facility_id", "Existing facility"))),
                ).add_to(existing_layer)
                folium.Circle(
                    [row["lat"], row["lon"]], radius=run_service_radius_km * 1000,
                    color="#0B3D91", fill=False, weight=1, dash_array="5",
                ).add_to(existing_layer)
            existing_layer.add_to(m)

        new_layer = folium.FeatureGroup(name="New sites (recommended)")
        for i, row in selected_df.iterrows():
            folium.Marker(
                [row["lat"], row["lon"]],
                icon=folium.Icon(color="orange", icon="warehouse", prefix="fa"),
                popup=(f"{row['site_id']} (opened #{i+1})<br>"
                       f"Incremental demand covered: {row['incremental_demand_covered']:.0f} {run_uom}<br>"
                       f"Cumulative coverage: {row['cumulative_coverage_pct']}%<br>"
                       f"Lease: {row['monthly_lease_cost']:,.0f}/mo"),
            ).add_to(new_layer)
            folium.Circle(
                [row["lat"], row["lon"]], radius=run_service_radius_km * 1000,
                color="#F5C518", fill=False, weight=2,
            ).add_to(new_layer)
        new_layer.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, width=None, height=550, returned_objects=[])

    with col2:
        st.subheader("Sites opened, in order")
        if len(selected_df) > 0:
            st.dataframe(
                selected_df[["site_id", "incremental_demand_covered", "cumulative_coverage_pct", "monthly_lease_cost"]].round(2),
                hide_index=True,
            )
            csv_buffer = io.StringIO()
            selected_df.to_csv(csv_buffer, index=False)
            st.download_button("Download results (CSV)", csv_buffer.getvalue(),
                                file_name="selected_sites.csv", mime="text/csv")
        else:
            st.caption("No new sites opened in this run.")
