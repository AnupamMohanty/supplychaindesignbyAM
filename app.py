import io

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from optimizer import (
    generate_candidate_sites,
    greedy_select_sites,
    validate_demand_df,
    validate_existing_df,
    validate_products_df,
)

st.set_page_config(page_title="Supply Chain Design by AM", layout="wide", page_icon="🚚")

# ---------- Theme accents beyond what config.toml covers ----------
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

# ---------- Sample data ----------
SAMPLE_DEMAND_COLS = ["lat", "lon", "daily_orders"]
SAMPLE_PRODUCTS = pd.DataFrame({
    "product_id": ["P001", "P002", "P003"],
    "product_name": ["Standard Parcel", "Bulk Pallet", "Cold Chain Item"],
    "unit_weight_kg": [2.5, 40.0, 5.0],
})
SAMPLE_EXISTING = pd.DataFrame({
    "facility_id": ["EX-001"],
    "facility_name": ["Existing DC - City Center"],
    "lat": [13.0827],
    "lon": [80.2707],
})

# ---------- Session state init ----------
if "demand_df" not in st.session_state:
    st.session_state.demand_df = pd.read_csv("sample_demand.csv")[["lat", "lon", "daily_orders"]]
if "products_df" not in st.session_state:
    st.session_state.products_df = SAMPLE_PRODUCTS.copy()
if "existing_df" not in st.session_state:
    st.session_state.existing_df = SAMPLE_EXISTING.copy()

# ---------- Tabs ----------
tab_data, tab_results = st.tabs(["📋 Data Input", "🗺️ Map & Results"])

# ================= DATA INPUT TAB =================
with tab_data:
    st.subheader("Customer Demand")
    st.caption("One row per customer / demand location. Required columns: lat, lon, daily_orders.")
    demand_upload = st.file_uploader("Upload customer demand CSV", type=["csv"], key="demand_upload")
    if demand_upload is not None:
        st.session_state.demand_df = pd.read_csv(demand_upload)
    st.session_state.demand_df = st.data_editor(
        st.session_state.demand_df, num_rows="dynamic", width='stretch', key="demand_editor"
    )
    valid_d, msg_d = validate_demand_df(st.session_state.demand_df)
    if not valid_d:
        st.error(f"Customer demand: {msg_d}")

    st.divider()

    st.subheader("Products")
    st.caption("Reference table for product mix. (Product-level capacity constraints are a planned future enhancement — not yet used in the optimization.)")
    products_upload = st.file_uploader("Upload products CSV", type=["csv"], key="products_upload")
    if products_upload is not None:
        st.session_state.products_df = pd.read_csv(products_upload)
    st.session_state.products_df = st.data_editor(
        st.session_state.products_df, num_rows="dynamic", width='stretch', key="products_editor"
    )
    valid_p, msg_p = validate_products_df(st.session_state.products_df)
    if not valid_p:
        st.error(f"Products: {msg_p}")

    st.divider()

    st.subheader("Existing Facilities")
    st.caption("Facilities already open. Their coverage counts as a baseline before any new sites are added. Leave empty for a pure greenfield run.")
    existing_upload = st.file_uploader("Upload existing facilities CSV", type=["csv"], key="existing_upload")
    if existing_upload is not None:
        st.session_state.existing_df = pd.read_csv(existing_upload)
    st.session_state.existing_df = st.data_editor(
        st.session_state.existing_df, num_rows="dynamic", width='stretch', key="existing_editor"
    )
    valid_e, msg_e = validate_existing_df(st.session_state.existing_df)
    if not valid_e:
        st.error(f"Existing facilities: {msg_e}")

# ================= SIDEBAR: OPTIMIZATION CONTROLS =================
with st.sidebar:
    st.header("Optimization settings")

    grid_step = st.slider("Candidate grid spacing (degrees)", 0.01, 0.08, 0.03, 0.005,
                           help="Smaller = more candidate sites, slower to compute")
    service_radius = st.slider("Service radius (km)", 1, 30, 8,
                                help="Max distance a facility can economically serve a demand point")

    st.subheader("Optimize by")
    opt_mode = st.radio(
        "Choose optimization mode",
        options=["Number of new sites", "Service coverage target (%)"],
        label_visibility="collapsed",
    )

    if opt_mode == "Number of new sites":
        num_sites = st.number_input("Number of new sites to open", min_value=1, max_value=50, value=5, step=1)
        target_pct = None
        max_sites_cap = None
        mode_key = "num_sites"
    else:
        target_pct = st.number_input("Target % of demand served within radius", min_value=1.0, max_value=100.0,
                                      value=80.0, step=1.0)
        max_sites_cap = st.number_input("Max new sites allowed (safety cap)", min_value=1, max_value=50, value=15, step=1)
        num_sites = None
        mode_key = "service_target"

    run_button = st.button("Run optimization", type="primary", width='stretch')

# ================= RUN OPTIMIZATION =================
demand_df = st.session_state.demand_df.dropna(subset=["lat", "lon", "daily_orders"])
existing_df = st.session_state.existing_df.dropna(subset=["lat", "lon"]) if len(st.session_state.existing_df) else st.session_state.existing_df

if run_button:
    valid_d, msg_d = validate_demand_df(demand_df)
    if not valid_d:
        st.error(f"Cannot run — customer demand data problem: {msg_d}")
    else:
        with st.spinner("Generating candidates and optimizing..."):
            cand_df = generate_candidate_sites(demand_df, grid_step_deg=grid_step)
            selected_df, summary = greedy_select_sites(
                demand_df, cand_df, existing_df,
                service_radius_km=service_radius,
                mode=mode_key,
                num_sites=num_sites if num_sites else 5,
                target_pct=target_pct if target_pct else 80.0,
                max_sites=max_sites_cap if max_sites_cap else 25,
            )
        st.session_state["selected_df"] = selected_df
        st.session_state["opt_summary"] = summary
        st.session_state["run_demand_df"] = demand_df
        st.session_state["run_existing_df"] = existing_df
        st.session_state["run_service_radius"] = service_radius

# ================= RESULTS TAB =================
with tab_results:
    if "selected_df" not in st.session_state:
        st.info("Fill in your data on the 'Data Input' tab, set your optimization mode in the sidebar, and click 'Run optimization'.")
    else:
        selected_df = st.session_state["selected_df"]
        summary = st.session_state["opt_summary"]
        run_demand_df = st.session_state["run_demand_df"]
        run_existing_df = st.session_state["run_existing_df"]
        run_service_radius = st.session_state["run_service_radius"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Sites opened", summary["sites_selected"])
        m2.metric("Baseline coverage (existing only)", f"{summary['baseline_coverage_pct']}%")
        m3.metric("Final coverage (existing + new)", f"{summary['final_coverage_pct']}%")
        m4.metric("Total daily orders", f"{summary['total_daily_orders']:.0f}")

        if mode_key == "service_target":
            if summary.get("target_met"):
                st.success(f"Target of {target_pct}% coverage reached with {summary['sites_selected']} new site(s).")
            else:
                st.warning(f"Target of {target_pct}% coverage NOT reached within the {max_sites_cap}-site cap — "
                           f"reached {summary['final_coverage_pct']}%. Try raising the site cap or the service radius.")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Map")
            center_lat, center_lon = run_demand_df["lat"].mean(), run_demand_df["lon"].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")

            demand_layer = folium.FeatureGroup(name="Demand points")
            for _, row in run_demand_df.iterrows():
                folium.CircleMarker(
                    [row["lat"], row["lon"]], radius=3, color="#0B36C0",
                    fill=True, fill_opacity=0.5,
                    popup=f"{row['daily_orders']:.0f} orders/day",
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
                        [row["lat"], row["lon"]], radius=run_service_radius * 1000,
                        color="#0B3D91", fill=False, weight=1, dash_array="5",
                    ).add_to(existing_layer)
                existing_layer.add_to(m)

            new_layer = folium.FeatureGroup(name="New sites (recommended)")
            for i, row in selected_df.iterrows():
                folium.Marker(
                    [row["lat"], row["lon"]],
                    icon=folium.Icon(color="orange", icon="warehouse", prefix="fa"),
                    popup=(f"{row['site_id']} (opened #{i+1})<br>"
                           f"Incremental orders covered: {row['incremental_orders_covered']:.0f}<br>"
                           f"Cumulative coverage: {row['cumulative_coverage_pct']}%<br>"
                           f"Lease: {row['monthly_lease_cost']:,.0f}/mo"),
                ).add_to(new_layer)
                folium.Circle(
                    [row["lat"], row["lon"]], radius=run_service_radius * 1000,
                    color="#F5C518", fill=False, weight=2,
                ).add_to(new_layer)
            new_layer.add_to(m)

            folium.LayerControl(collapsed=False).add_to(m)
            st_folium(m, width=None, height=550, returned_objects=[])

        with col2:
            st.subheader("Sites opened, in order")
            st.dataframe(
                selected_df[["site_id", "incremental_orders_covered", "cumulative_coverage_pct", "monthly_lease_cost"]].round(2),
                hide_index=True,
            )
            csv_buffer = io.StringIO()
            selected_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download results (CSV)", csv_buffer.getvalue(),
                file_name="selected_sites.csv", mime="text/csv",
            )
