import io
import time

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from optimizer import (
    DEFAULT_SCORING_CRITERIA,
    TARGET_SCHEMAS,
    apply_column_mapping,
    assign_customers_to_facilities,
    auto_grid_step_deg,
    build_data_sources_workbook,
    build_mapping_prompt,
    call_claude_api,
    classify_comparison_intent,
    compute_service_radius,
    compute_weighted_scores,
    coordinates_ready,
    estimate_utm_epsg,
    generate_candidate_sites,
    geocode_locations,
    greedy_select_sites,
    lookup_reference_scores,
    needs_geocoding,
    parse_mapping_response,
    reverse_geocode_details,
    suggest_best_next_location,
    validate_demand_df,
    validate_existing_df,
    validate_products_df,
)

st.set_page_config(page_title="Supply Chain Design by AM", layout="wide", page_icon="🚚")

# ---------- Theme ----------
# IMPORTANT: no blank lines inside this <style> block. Streamlit's markdown
# renderer splits raw HTML on blank lines into separate "paragraphs", and any
# paragraph not starting with a recognized HTML tag gets escaped and shown
# as literal text on the page instead of being applied as CSS.
st.markdown("""<style>
html, body, [class*="css"] { font-family: -apple-system, 'Segoe UI', Roboto, Inter, sans-serif; }
.hero {
    background: linear-gradient(120deg, #0B3D91 0%, #123B7A 55%, #0A2A63 100%);
    color: #FFFFFF;
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(11,61,145,0.18);
}
.hero h1 { margin: 0; font-size: 30px; font-weight: 800; letter-spacing: -0.5px; }
.hero p { margin: 6px 0 0 0; color: #CFE0FF; font-size: 15px; font-weight: 500; }
.hero .badge {
    display: inline-block; background: #F5C518; color: #0B3D91;
    font-weight: 700; font-size: 12px; padding: 3px 10px; border-radius: 20px;
    margin-top: 10px;
}
.section-title {
    font-size: 17px; font-weight: 700; color: #0B3D91;
    margin-bottom: 2px; display: flex; align-items: center; gap: 8px;
}
.section-sub { color: #6B7A99; font-size: 13px; margin-bottom: 14px; }
div[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #E7ECF5; border-radius: 12px;
    padding: 14px 16px; box-shadow: 0 2px 8px rgba(11,61,145,0.05);
}
div[data-testid="stMetricLabel"] { font-weight: 600; color: #4A5A78; }
div.stButton > button[kind="primary"] {
    background: linear-gradient(120deg, #F5C518, #E8B400);
    color: #0B3D91; border: none; font-weight: 700;
    box-shadow: 0 4px 12px rgba(245,197,24,0.35);
}
div.stButton > button[kind="primary"]:hover { filter: brightness(1.05); }
[data-testid="stSidebar"] { background: #F7F9FC; border-right: 1px solid #E7ECF5; }
.rec-card {
    background: linear-gradient(135deg, #F7F9FC 0%, #EAF2FB 100%);
    border: 1px solid #D9E4F5; border-radius: 14px; padding: 20px 24px; margin-top: 10px;
}
.source-pill {
    display: inline-block; background: #EAF2FB; color: #0B3D91;
    font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px;
}
.step-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #FFFFFF; border: 1px solid #E7ECF5; border-radius: 20px;
    padding: 6px 14px; font-size: 13px; font-weight: 600; color: #4A5A78;
    margin-right: 8px; margin-bottom: 16px;
}
.step-pill.active { background: #0B3D91; color: #FFFFFF; border-color: #0B3D91; }
div[data-testid="stExpander"] { border: 1px solid #E7ECF5; border-radius: 12px; }
div[data-testid="stChatMessage"] { border-radius: 12px; }
.app-footer {
    text-align: center; color: #9AA7C0; font-size: 12px; margin-top: 32px;
    padding-top: 16px; border-top: 1px solid #E7ECF5;
}
</style>""", unsafe_allow_html=True)

st.markdown("""<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stToolbar"] {visibility: hidden;}
[data-testid="stDecoration"] {display: none;}
</style>""", unsafe_allow_html=True)

st.markdown("""<div class="hero">
    <h1>🚚 Supply Chain Design by AM</h1>
    <p>Greenfield facility location optimizer — demand, network, and site scoring in one flow</p>
    <span class="badge">GREENFIELD · MCLP ENGINE</span>
</div>""", unsafe_allow_html=True)


def _step_indicator(current: str):
    steps = [("data", "① Data & Settings"), ("compare", "📊 Compare Scenarios"), ("results", "② Results & Analysis")]
    pills = "".join(
        f'<span class="step-pill{" active" if key == current else ""}">{label}</span>'
        for key, label in steps
    )
    st.markdown(pills, unsafe_allow_html=True)

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
    "scoring_weights": {c["key"]: c["default_weight"] for c in DEFAULT_SCORING_CRITERIA},
    "api_key": "",
    "scenarios": {},
    "baseline_scenario": None,
    "chat_history": [],
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
    with st.expander("🤖 GenAI settings (optional)"):
        st.session_state.api_key = st.text_input(
            "Anthropic API key", value=st.session_state.api_key, type="password",
            help="Powers the basefile copilot and the natural-language scenario chat below. "
                 "Your key is kept only in this session, never saved to disk. Get one at console.anthropic.com. "
                 "NOT required for the 'Compare All Scenarios' button — that works free, with no key.",
        )
        if not st.session_state.api_key:
            st.caption("Optional — the free 'Compare All Scenarios' button below doesn't need this.")

    st.markdown("""<div style="background:#0B3D91; border-radius:12px; padding:12px 14px; margin-bottom:12px;">
        <span style="color:#F5C518; font-weight:800; font-size:15px;">💾 Saved Scenarios</span>
        </div>""", unsafe_allow_html=True)

    n_scenarios = len(st.session_state.scenarios)
    n_runnable = len([s for s in st.session_state.scenarios.values() if s.get("summary")])

    if n_scenarios == 0:
        st.caption("No scenarios saved yet. Run the optimizer, then use 'Save current scenario' below to start comparing.")
    else:
        for name, snap in list(st.session_state.scenarios.items()):
            label = f"⭐ **{name}**" if name == st.session_state.baseline_scenario else f"**{name}**"
            st.markdown(label)
            if snap.get("summary"):
                s = snap["summary"]
                wavg = s.get("weighted_avg_distance_km")
                wavg_txt = f" · {wavg*0.621371:.0f} mi avg" if wavg is not None else ""
                st.caption(f"{s['sites_selected']} sites · {s['final_coverage_pct']}% coverage{wavg_txt}")
            else:
                st.caption("Inputs only — not yet run")
            lc1, lc2 = st.columns(2)
            with lc1:
                if st.button("Load", key=f"load_{name}", width="stretch"):
                    st.session_state.demand_df = snap["demand_df"].copy()
                    st.session_state.products_df = snap["products_df"].copy()
                    st.session_state.existing_df = snap["existing_df"].copy()
                    st.session_state.include_existing = snap["include_existing"]
                    st.session_state["view"] = "input"
                    st.rerun()
            with lc2:
                if st.button("Delete", key=f"delete_{name}", width="stretch"):
                    del st.session_state.scenarios[name]
                    if st.session_state.baseline_scenario == name:
                        st.session_state.baseline_scenario = None
                    st.rerun()

        st.markdown("")
        if st.button("📊 Compare All Scenarios", type="primary", width="stretch",
                      disabled=n_runnable < 2,
                      help=None if n_runnable >= 2 else "Save at least 2 scenarios with a completed run to compare."):
            st.session_state["view"] = "compare"
            st.rerun()
        if n_runnable < 2:
            st.caption(f"{n_runnable}/2 scenarios with completed runs — free, no API key needed.")

    st.divider()
    st.markdown("### ⚙️ Model settings")

    uom_choice = st.selectbox("Unit of measure for this model", UOM_OPTIONS,
                               index=UOM_OPTIONS.index(st.session_state.model_uom)
                               if st.session_state.model_uom in UOM_OPTIONS else 0,
                               help="Every demand_value in this model is measured in this single unit. "
                                    "One unit for the whole model — mixing units breaks the math.")
    st.session_state.model_uom = uom_choice
    if uom_choice == "Other (specify below)":
        st.session_state.model_uom_custom = st.text_input("Custom unit name", value=st.session_state.model_uom_custom)
    active_uom = st.session_state.model_uom_custom if uom_choice == "Other (specify below)" and st.session_state.model_uom_custom else uom_choice

    st.divider()
    st.markdown("### 🎯 Optimize by")
    opt_mode = st.segmented_control(
        "Choose optimization mode", options=["Number of new sites", "Service coverage target (%)"],
        default="Number of new sites", label_visibility="collapsed",
    )
    opt_mode = opt_mode or "Number of new sites"

    if opt_mode == "Number of new sites":
        num_sites = st.number_input("Number of new sites to open", min_value=1, max_value=50, value=5, step=1)
        target_pct, max_sites_cap, mode_key = None, None, "num_sites"
    else:
        target_pct = st.number_input("Target % of demand to serve", min_value=1.0,
                                      max_value=100.0, value=80.0, step=1.0)
        max_sites_cap = st.number_input("Max new sites allowed (safety cap)", min_value=1, max_value=50, value=15, step=1)
        num_sites, mode_key = None, "service_target"

    st.divider()
    st.markdown("### ⏱️ Service coverage assumptions")

    col_a, col_b = st.columns(2)
    with col_a:
        service_time_value = st.number_input("Desired service time", min_value=0.1, value=1.0, step=0.5)
    with col_b:
        service_time_unit = st.selectbox("Unit", ["Days", "Hours"])

    miles_per_day = st.number_input("Last-mile daily travel capacity (miles/day)", min_value=50, max_value=1000,
                                     value=400, step=50,
                                     help="Assumption: how far a delivery truck can realistically travel in one day.")

    service_radius_km, service_radius_miles = compute_service_radius(service_time_value, service_time_unit, miles_per_day)
    st.caption(f"→ Effective service radius: **{service_radius_km:,.0f} km** ({service_radius_miles:,.0f} miles)")

    st.session_state.include_existing = st.checkbox(
        "Include existing sites in this run?", value=st.session_state.include_existing,
        help="Uncheck to run a pure greenfield analysis, ignoring the Existing Facilities table entirely."
    )

    st.divider()
    st.markdown("### 💾 Save current scenario")

    scenario_name = st.text_input("Scenario name", key="scenario_name_input", placeholder="e.g. Baseline 2026")
    is_baseline_checkbox = st.checkbox("Consider this scenario as baseline?", key="is_baseline_checkbox")

    sc_col1, sc_col2 = st.columns(2)
    with sc_col1:
        save_scenario_clicked = st.button("💾 Save", width="stretch")
    with sc_col2:
        clear_scenario_clicked = st.button("🗑️ Clear inputs", width="stretch")

    if save_scenario_clicked:
        if not scenario_name.strip():
            st.warning("Give the scenario a name before saving.")
        else:
            snapshot = {
                "products_df": st.session_state.products_df.copy(),
                "demand_df": st.session_state.demand_df.copy(),
                "existing_df": st.session_state.existing_df.copy(),
                "model_uom": active_uom,
                "include_existing": st.session_state.include_existing,
                "service_time_value": service_time_value,
                "service_time_unit": service_time_unit,
                "miles_per_day": miles_per_day,
                "service_radius_km": service_radius_km,
                "opt_mode": opt_mode,
                "num_sites": num_sites,
                "target_pct": target_pct,
                "max_sites_cap": max_sites_cap,
                "is_baseline": is_baseline_checkbox,
                "summary": st.session_state.get("opt_summary"),
                "selected_df": st.session_state.get("selected_df"),
                "run_demand_df": st.session_state.get("run_demand_df"),
            }
            st.session_state.scenarios[scenario_name.strip()] = snapshot
            if is_baseline_checkbox:
                st.session_state.baseline_scenario = scenario_name.strip()
            st.success(f"Scenario '{scenario_name.strip()}' saved" +
                       (" as baseline." if is_baseline_checkbox else "."))
            st.rerun()

    if clear_scenario_clicked:
        st.session_state.demand_df = EMPTY_DEMAND.copy()
        st.session_state.products_df = EMPTY_PRODUCTS.copy()
        st.session_state.existing_df = EMPTY_EXISTING.copy()
        for k in ["selected_df", "opt_summary", "run_demand_df"]:
            st.session_state.pop(k, None)
        st.session_state["view"] = "input"
        st.success("Inputs cleared — ready for a new scenario.")
        st.rerun()

# =========================================================================
# INPUT VIEW
# =========================================================================
if st.session_state.view == "input":
    _step_indicator("data")

    with st.container(border=True):
        st.markdown('<div class="section-title">🤖 Load a Basefile (AI-Assisted)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Have shipment history, transactional data, or a forecast instead of '
                     'a ready-made table? Upload it here and tell the copilot how to map it onto one of the three '
                     'tables below.</div>', unsafe_allow_html=True)

        basefile = st.file_uploader("Upload shipments / transactions / forecast (CSV or Excel)",
                                     type=["csv", "xlsx"], key="basefile_upload")

        if basefile is not None:
            try:
                raw_df = pd.read_csv(basefile) if basefile.name.endswith(".csv") else pd.read_excel(basefile)
                st.dataframe(raw_df.head(5), width="stretch")

                bc1, bc2 = st.columns([1, 2])
                with bc1:
                    target_table = st.selectbox("Map this file to", list(TARGET_SCHEMAS.keys()), key="basefile_target")
                with bc2:
                    user_instruction = st.text_input(
                        "Tell the copilot how to map it (optional)", key="basefile_instruction",
                        placeholder="e.g. origin_city and origin_state are the customer location, weight_kg is demand",
                    )

                if not st.session_state.api_key:
                    st.info("Enter an Anthropic API key in the sidebar (🤖 GenAI settings) to enable AI-assisted mapping.")
                transform_clicked = st.button("🪄 Transform with AI", disabled=not st.session_state.api_key)

                if transform_clicked:
                    with st.spinner("Copilot is mapping your file..."):
                        sys_p, user_p = build_mapping_prompt(
                            target_table, list(raw_df.columns), raw_df.head(5).to_string(), user_instruction
                        )
                        response, err = call_claude_api(st.session_state.api_key, sys_p, user_p)
                    if err:
                        st.error(f"Copilot error: {err}")
                    else:
                        parsed, perr = parse_mapping_response(response)
                        if perr:
                            st.error(f"Couldn't understand the copilot's response: {perr}")
                        else:
                            transformed, aerr = apply_column_mapping(raw_df, parsed["mapping"])
                            if aerr:
                                st.error(f"Couldn't apply the mapping: {aerr}")
                            else:
                                st.session_state["_pending_transform"] = {
                                    "target_table": target_table, "df": transformed, "notes": parsed.get("notes", ""),
                                    "unmapped": parsed.get("unmapped_target_columns", []),
                                }

                pending = st.session_state.get("_pending_transform")
                if pending:
                    st.success(f"Proposed mapping for **{pending['target_table']}**:")
                    if pending["notes"]:
                        st.caption(pending["notes"])
                    if pending["unmapped"]:
                        st.warning(f"Not mapped (fill in manually after applying): {', '.join(pending['unmapped'])}")
                    st.dataframe(pending["df"].head(10), width="stretch")
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        if st.button("✅ Apply to " + pending["target_table"], width="stretch"):
                            target_key = {"Customer Demand": "demand_df", "Products": "products_df",
                                          "Existing Facilities": "existing_df"}[pending["target_table"]]
                            st.session_state[target_key] = pending["df"]
                            del st.session_state["_pending_transform"]
                            st.rerun()
                    with pc2:
                        if st.button("Discard", width="stretch"):
                            del st.session_state["_pending_transform"]
                            st.rerun()
            except Exception as e:
                st.error(f"Couldn't read that file: {e}")

    with st.container(border=True):
        st.markdown('<div class="section-title">📦 1. Products</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">Every product\'s demand is measured in <b>{active_uom}</b> '
                     f'(set in the sidebar). Required table.</div>', unsafe_allow_html=True)
        p_col1, p_col2 = st.columns([3, 1])
        with p_col1:
            products_upload = st.file_uploader("Upload products CSV", type=["csv"], key="products_upload",
                                                 label_visibility="collapsed")
        with p_col2:
            if st.button("Load sample", key="load_sample_products", width="stretch"):
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

    with st.container(border=True):
        st.markdown('<div class="section-title">📍 2. Customer Demand</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">One row per customer / city. demand_value is measured in '
                     f'<b>{active_uom}</b>. Required table.</div>', unsafe_allow_html=True)
        d_col1, d_col2 = st.columns([3, 1])
        with d_col1:
            demand_upload = st.file_uploader("Upload customer demand CSV", type=["csv"], key="demand_upload",
                                               label_visibility="collapsed")
        with d_col2:
            if st.button("Load sample", key="load_sample_demand", width="stretch"):
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

    with st.container(border=True):
        st.markdown('<div class="section-title">🏭 3. Existing Facilities</div>', unsafe_allow_html=True)
        include_note = "included in" if st.session_state.include_existing else "excluded from (toggle in sidebar)"
        st.markdown(f'<div class="section-sub">Facilities already open. Currently <b>{include_note}</b> '
                     f'this run. Optional table.</div>', unsafe_allow_html=True)
        e_col1, e_col2 = st.columns([3, 1])
        with e_col1:
            existing_upload = st.file_uploader("Upload existing facilities CSV", type=["csv"], key="existing_upload",
                                                 label_visibility="collapsed")
        with e_col2:
            if st.button("Load sample", key="load_sample_existing", width="stretch"):
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

    # ---------- Geocoding ----------
    demand_needs_geo = needs_geocoding(st.session_state.demand_df)
    existing_needs_geo = st.session_state.include_existing and needs_geocoding(st.session_state.existing_df)

    if demand_needs_geo or existing_needs_geo:
        st.info("📍 Some rows are missing lat/lon coordinates. Geocode automatically from City + Country below.")
        if st.button("🌍 Geocode missing locations", key="geocode_btn"):
            with st.spinner("Looking up coordinates from City + Country..."):
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
                st.warning(f"Geocoding complete, but {n_failed} row(s) could not be resolved.")
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

    run_button = st.button("▶  Run optimization", type="primary", disabled=not ready_to_run, width="stretch")

    if run_button and ready_to_run:
        placeholder = st.empty()
        placeholder.info("⏳ Model is running...")

        start_t = time.perf_counter()

        demand_df = st.session_state.demand_df.copy()
        demand_df["demand_value"] = pd.to_numeric(demand_df["demand_value"], errors="coerce")
        existing_df = st.session_state.existing_df.copy() if st.session_state.include_existing else pd.DataFrame(
            columns=["facility_id", "facility_name", "city", "country", "lat", "lon"]
        )
        if "facility_name" not in existing_df.columns:
            existing_df["facility_name"] = existing_df.get("facility_id", "Existing DC")

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

        # DC naming via reverse geocoding
        if len(selected_df) > 0:
            with st.spinner("Naming new sites..."):
                details = reverse_geocode_details(selected_df)
                selected_df["dc_name"] = [d["name"] for d in details]
                selected_df["dc_state"] = [d["state"] for d in details]
                selected_df["facility_name"] = "New DC – " + selected_df["dc_name"]

        # Customer -> nearest facility assignment
        combined_facilities = []
        if len(existing_df) > 0:
            combined_facilities.append(existing_df[["facility_name", "lat", "lon"]])
        if len(selected_df) > 0:
            combined_facilities.append(selected_df[["facility_name", "lat", "lon"]])
        all_facilities_df = pd.concat(combined_facilities, ignore_index=True) if combined_facilities else None

        epsg = estimate_utm_epsg(demand_df["lon"].mean(), demand_df["lat"].mean())
        assigned_demand_df = assign_customers_to_facilities(demand_df, all_facilities_df, epsg)

        elapsed = time.perf_counter() - start_t
        placeholder.empty()

        st.session_state["selected_df"] = selected_df
        st.session_state["opt_summary"] = summary
        st.session_state["run_demand_df"] = assigned_demand_df
        st.session_state["run_existing_df"] = existing_df
        st.session_state["run_service_radius_km"] = service_radius_km
        st.session_state["run_service_radius_miles"] = service_radius_miles
        st.session_state["run_time"] = elapsed
        st.session_state["run_mode_key"] = mode_key
        st.session_state["run_target_pct"] = target_pct
        st.session_state["run_max_sites_cap"] = max_sites_cap
        st.session_state["run_uom"] = active_uom
        st.session_state["run_cand_df"] = cand_df
        st.session_state["view"] = "results"
        st.rerun()

elif st.session_state.view == "compare":
    _step_indicator("compare")
    if st.button("⬅  Back"):
        st.session_state["view"] = "input"
        st.rerun()

    st.markdown('<div class="section-title">📊 Compare All Scenarios</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Free comparison across every saved scenario with a completed run — '
                 'no API key needed. Baseline is marked ⭐.</div>', unsafe_allow_html=True)

    runnable = {name: snap for name, snap in st.session_state.scenarios.items() if snap.get("summary")}

    if len(runnable) < 2:
        st.info(f"You have {len(runnable)} scenario(s) with completed runs. Save at least 2 to compare "
                "(sidebar → 💾 Save current scenario, after running the optimizer).")
    else:
        rows = []
        for name, snap in runnable.items():
            s = snap["summary"]
            wavg = s.get("weighted_avg_distance_km")
            rows.append({
                "Scenario": name,
                "Baseline": "⭐" if name == st.session_state.baseline_scenario else "",
                "Sites opened": s["sites_selected"],
                "Baseline coverage %": s["baseline_coverage_pct"],
                "Final coverage %": s["final_coverage_pct"],
                "Weighted avg distance (mi)": round(wavg * 0.621371, 1) if wavg is not None else None,
                "Total demand": s["total_demand"],
            })
        comp_df = pd.DataFrame(rows).set_index("Scenario")

        st.dataframe(comp_df, width="stretch")

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Final coverage % by scenario**")
            st.bar_chart(comp_df[["Final coverage %"]])
        with cc2:
            st.markdown("**Weighted avg service distance (mi) by scenario**")
            st.bar_chart(comp_df[["Weighted avg distance (mi)"]])

        cc3, cc4 = st.columns(2)
        with cc3:
            st.markdown("**Sites opened by scenario**")
            st.bar_chart(comp_df[["Sites opened"]])
        with cc4:
            if st.session_state.baseline_scenario and st.session_state.baseline_scenario in runnable:
                st.markdown(f"**vs. baseline ({st.session_state.baseline_scenario})**")
                base_cov = comp_df.loc[st.session_state.baseline_scenario, "Final coverage %"]
                delta_df = (comp_df[["Final coverage %"]] - base_cov).rename(
                    columns={"Final coverage %": "Coverage % vs baseline"})
                st.bar_chart(delta_df)
            else:
                st.caption("Mark a scenario as baseline (when saving) to see delta comparisons here.")

        csv_buffer = io.StringIO()
        comp_df.to_csv(csv_buffer)
        st.download_button("⬇ Download comparison (CSV)", csv_buffer.getvalue(),
                            file_name="scenario_comparison.csv", mime="text/csv")

    st.divider()
    st.markdown('<div class="section-title">💬 Ask the AI assistant (optional)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">For natural-language questions about your scenarios. The comparisons '
                 'above are free and need no key — this is an optional extra for phrasing your own questions.'
                 '</div>', unsafe_allow_html=True)

    if len(runnable) < 2:
        st.caption("Save at least 2 scenarios with completed runs to use this too.")
    elif not st.session_state.api_key:
        st.caption("Enter an Anthropic API key in the sidebar (🤖 GenAI settings) to enable this — optional, "
                   "the comparisons above already work without it.")
    else:
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

        user_q = st.chat_input("e.g. 'Which scenario has the best coverage per site opened?'")
        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            intent, err = classify_comparison_intent(st.session_state.api_key, list(runnable.keys()), user_q)
            if err:
                reply = f"Couldn't reach the AI classifier ({err}) — see the comparison table and charts above."
            elif intent == "coverage_comparison":
                reply = "See the 'Final coverage % by scenario' chart above."
            elif intent == "distance_comparison":
                reply = "See the 'Weighted avg service distance' chart above."
            elif intent == "sites_comparison":
                reply = "See the 'Sites opened by scenario' chart above."
            else:
                reply = "See the full comparison table above for all metrics side by side."
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

        if st.session_state.chat_history and st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

# =========================================================================
# RESULTS VIEW
# =========================================================================
else:
    _step_indicator("results")
    if st.button("⬅  Back to Input"):
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
        m5.metric("⭐ Weighted avg service distance", f"{wavg_miles:.0f} mi ({wavg:.0f} km)",
                  help="Σ(distance to nearest facility × demand) / Σ(demand) — demand-weighted average "
                       "distance from every customer to its nearest open facility.")
    else:
        m5.metric("Weighted avg service distance", "—")

    if run_mode_key == "service_target":
        if summary.get("target_met"):
            st.success(f"Target of {run_target_pct}% coverage reached with {summary['sites_selected']} new site(s), "
                       f"within a {run_service_radius_km:,.0f} km ({run_service_radius_miles:,.0f} mi) service radius.")
        else:
            st.warning(f"Target of {run_target_pct}% coverage NOT reached within the {run_max_sites_cap}-site cap — "
                       f"reached {summary['final_coverage_pct']}%.")

    if summary["sites_selected"] == 0:
        st.info("No new sites were needed — existing facilities already meet the coverage target within this "
                "service radius. Try a smaller service radius/time, a higher coverage target, or unchecking "
                "'Include existing sites' to see a pure greenfield result.")

    st.divider()
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f'<div class="section-title">🗺️ Network Map — bubbles = demand ({run_uom}), dotted lines = customer→DC assignment</div>', unsafe_allow_html=True)
        show_lines = st.checkbox("Show customer → DC assignment lines", value=len(run_demand_df) <= 300,
                                  help="Auto-disabled by default for very large demand tables to keep the map responsive.")

        center_lat, center_lon = run_demand_df["lat"].mean(), run_demand_df["lon"].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="OpenStreetMap")

        max_demand = run_demand_df["demand_value"].max()
        max_demand = max_demand if max_demand and max_demand > 0 else 1

        # Facility name -> coords lookup, for drawing lines
        facility_coords = {}
        if run_existing_df is not None:
            for _, r in run_existing_df.iterrows():
                facility_coords[r.get("facility_name")] = (r["lat"], r["lon"])
        for _, r in selected_df.iterrows():
            facility_coords[r.get("facility_name")] = (r["lat"], r["lon"])

        if show_lines and "assigned_facility_name" in run_demand_df.columns:
            line_layer = folium.FeatureGroup(name="Customer → DC assignment")
            for _, row in run_demand_df.iterrows():
                fac_name = row.get("assigned_facility_name")
                if fac_name in facility_coords:
                    folium.PolyLine(
                        [(row["lat"], row["lon"]), facility_coords[fac_name]],
                        color="#8592AD", weight=1, opacity=0.6, dash_array="4,6",
                    ).add_to(line_layer)
            line_layer.add_to(m)

        demand_layer = folium.FeatureGroup(name="Demand (bubble size = volume)")
        for _, row in run_demand_df.iterrows():
            bubble_radius = 4 + (row["demand_value"] / max_demand) * 20
            served_by = row.get("assigned_facility_name", "")
            folium.CircleMarker(
                [row["lat"], row["lon"]], radius=bubble_radius, color="#D32F2F",
                fill=True, fill_color="#E53935", fill_opacity=0.45, weight=1,
                popup=f"{row.get('city', '')}: {row['demand_value']:.0f} {run_uom}<br>Served by: {served_by}",
            ).add_to(demand_layer)
        demand_layer.add_to(m)

        if run_existing_df is not None and len(run_existing_df) > 0:
            existing_layer = folium.FeatureGroup(name="Existing facilities")
            for _, row in run_existing_df.iterrows():
                folium.Marker(
                    [row["lat"], row["lon"]],
                    icon=folium.Icon(color="blue", icon="industry", prefix="fa"),
                    popup=str(row.get("facility_name", "Existing facility")),
                ).add_to(existing_layer)
                folium.Circle(
                    [row["lat"], row["lon"]], radius=run_service_radius_km * 1000,
                    color="#0B3D91", fill=False, weight=1, dash_array="5",
                ).add_to(existing_layer)
            existing_layer.add_to(m)

        new_layer = folium.FeatureGroup(name="New sites (recommended)")
        triangle_svg = (
            '<svg width="28" height="26" viewBox="0 0 28 26" xmlns="http://www.w3.org/2000/svg">'
            '<polygon points="14,1 27,25 1,25" fill="#0B6B2C" stroke="#053D18" stroke-width="1.5"/>'
            '</svg>'
        )
        for i, row in selected_df.iterrows():
            folium.Marker(
                [row["lat"], row["lon"]],
                icon=folium.DivIcon(html=triangle_svg, icon_size=(28, 26), icon_anchor=(14, 20)),
                popup=(f"<b>{row.get('dc_name', row['site_id'])}</b> (opened #{i+1})<br>"
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
        st_folium(m, width=None, height=560, returned_objects=[])

    with col2:
        st.markdown('<div class="section-title">🏭 Sites opened, in order</div>', unsafe_allow_html=True)
        if len(selected_df) > 0:
            display_cols = ["dc_name", "incremental_demand_covered", "cumulative_coverage_pct", "monthly_lease_cost"]
            display_cols = [c for c in display_cols if c in selected_df.columns]
            st.dataframe(selected_df[display_cols].round(2), hide_index=True, width="stretch")
        else:
            st.caption("No new sites opened in this run.")

        if "assigned_facility_name" in run_demand_df.columns and run_demand_df["assigned_facility_name"].notna().any():
            st.markdown("**Customers served per DC**")
            serve_counts = run_demand_df.groupby("assigned_facility_name")["demand_value"].agg(["count", "sum"])
            serve_counts.columns = ["# customers", f"Total {run_uom}"]
            st.dataframe(serve_counts.round(1), width="stretch")

            # Customer-level detail export: dc_name, customer, demand, distance in miles
            detail_export = run_demand_df.copy()
            detail_export["customer"] = detail_export.get("city", "")
            detail_export["distance_miles"] = (detail_export["distance_to_facility_km"] * 0.621371).round(2)
            detail_export = detail_export.rename(columns={"assigned_facility_name": "dc_name",
                                                            "demand_value": f"demand_{run_uom}"})
            detail_cols = ["dc_name", "customer", f"demand_{run_uom}", "distance_miles"]
            detail_cols = [c for c in detail_cols if c in detail_export.columns]
            csv_buffer = io.StringIO()
            detail_export[detail_cols].to_csv(csv_buffer, index=False)
            st.download_button("⬇ Download customer-level results (CSV)", csv_buffer.getvalue(),
                                file_name="customer_dc_assignments.csv", mime="text/csv", width="stretch")

    # =====================================================================
    # CAUSAL ANALYSIS
    # =====================================================================
    if len(selected_df) > 0:
        st.divider()
        ca_col1, ca_col2 = st.columns([5, 1])
        with ca_col1:
            st.markdown('<div class="section-title">🧭 Causal Analysis — weighted multi-criteria evaluation</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Set weights for each factor (must total 100%), then score each opened '
                         'DC 0-10 on that factor. Where a site matches a researched reference region, scores are '
                         'pre-filled — adjust freely based on your own diligence.</div>', unsafe_allow_html=True)
        with ca_col2:
            st.download_button("📊 Data Sources", build_data_sources_workbook(),
                                file_name="data_sources.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                width="stretch")

        weight_cols = st.columns(4)
        new_weights = {}
        for i, crit in enumerate(DEFAULT_SCORING_CRITERIA):
            with weight_cols[i % 4]:
                new_weights[crit["key"]] = st.number_input(
                    crit["label"], min_value=0, max_value=100,
                    value=st.session_state.scoring_weights.get(crit["key"], crit["default_weight"]),
                    step=5, key=f"weight_{crit['key']}",
                )
        st.session_state.scoring_weights = new_weights
        total_weight = sum(new_weights.values())

        if total_weight == 100:
            st.success(f"✅ Weights total {total_weight}%")
        else:
            st.warning(f"⚠️ Weights total {total_weight}% — should total 100% for the score to be meaningful. "
                       f"Scoring will still run, normalized to what you've entered.")

        # Build/refresh the scores table, pre-filled from research where matched
        if "site_scores_df" not in st.session_state or set(st.session_state.get("_scored_dc_names", [])) != set(selected_df["dc_name"]):
            rows = []
            research_notes = []
            for _, site in selected_df.iterrows():
                dc_name = site["dc_name"]
                match = lookup_reference_scores(dc_name)
                row = {"dc_name": dc_name}
                if match:
                    scores, note = match
                    row.update(scores)
                    research_notes.append(f"**{dc_name}** — {note}")
                else:
                    row.update({c["key"]: 5 for c in DEFAULT_SCORING_CRITERIA})
                rows.append(row)
            st.session_state.site_scores_df = pd.DataFrame(rows)
            st.session_state["_scored_dc_names"] = list(selected_df["dc_name"])
            st.session_state["_research_notes"] = research_notes

        if st.session_state.get("_research_notes"):
            with st.expander("📚 Research notes for matched locations"):
                for note in st.session_state["_research_notes"]:
                    st.markdown(note)
                st.caption("Full sourcing with URLs is in the 'Data Sources' download above.")

        st.markdown("**Score each site (0–10 per factor)**")
        score_column_config = {
            "dc_name": st.column_config.TextColumn("DC Location", width="medium"),
            **{c["key"]: st.column_config.NumberColumn(c["label"], min_value=0, max_value=10, step=1)
               for c in DEFAULT_SCORING_CRITERIA},
        }
        st.session_state.site_scores_df = st.data_editor(
            st.session_state.site_scores_df, width="stretch", key="scores_editor",
            column_config=score_column_config, hide_index=True,
        )

        scored = compute_weighted_scores(st.session_state.site_scores_df, new_weights)

        st.markdown("**Ranked site scores**")
        ranked_column_config = {
            "dc_name": st.column_config.TextColumn("DC Location", width="medium"),
            "weighted_score": st.column_config.ProgressColumn(
                "⭐ Weighted Score", min_value=0, max_value=10, format="%.2f"
            ),
            **{c["key"]: st.column_config.NumberColumn(c["label"], width="small")
               for c in DEFAULT_SCORING_CRITERIA},
        }
        st.dataframe(
            scored[["dc_name", "weighted_score"] + [c["key"] for c in DEFAULT_SCORING_CRITERIA]],
            hide_index=True, width="stretch", column_config=ranked_column_config,
        )

        # Best-next-location advisory for any site scoring below the bar
        low_scorers = scored[scored["weighted_score"] < 7]
        if len(low_scorers) > 0:
            st.markdown("**⚠️ Locations below score threshold (7.0) — suggested next steps**")
            for _, row in low_scorers.iterrows():
                site_state = None
                site_match = selected_df[selected_df["dc_name"] == row["dc_name"]]
                if len(site_match) > 0 and "dc_state" in site_match.columns:
                    site_state = site_match.iloc[0]["dc_state"]
                suggestion = suggest_best_next_location(
                    row["dc_name"], row, new_weights,
                    st.session_state.get("run_cand_df"), selected_df, run_service_radius_km,
                    state=site_state,
                )
                st.warning(suggestion)

        # =================================================================
        # RECOMMENDATION — synthesis of optimization + causal analysis
        # =================================================================
        st.divider()
        st.markdown('<div class="section-title">✅ Recommendation</div>', unsafe_allow_html=True)

        top_site = scored.iloc[0] if len(scored) > 0 else None
        avg_score = scored["weighted_score"].mean() if len(scored) > 0 else 0
        n_low = len(low_scorers)

        rec_lines = []
        rec_lines.append(f"This run opens **{summary['sites_selected']} new site(s)**, lifting coverage from "
                          f"**{summary['baseline_coverage_pct']}%** to **{summary['final_coverage_pct']}%** of total "
                          f"demand, with a demand-weighted average service distance of "
                          f"**{(wavg * 0.621371):.0f} miles ({wavg:.0f} km)**." if wavg is not None else
                          f"This run opens **{summary['sites_selected']} new site(s)**, lifting coverage from "
                          f"{summary['baseline_coverage_pct']}% to {summary['final_coverage_pct']}%.")

        if top_site is not None:
            rec_lines.append(f"On causal analysis, **{top_site['dc_name']}** ranks highest "
                              f"(weighted score **{top_site['weighted_score']:.1f}/10**), making it the strongest "
                              f"combination of network fit and site-quality factors among the sites opened.")

        rec_lines.append(f"Average causal score across opened sites: **{avg_score:.1f}/10**.")

        if n_low > 0:
            rec_lines.append(f"⚠️ **{n_low} site(s) scored below 7.0** — see the suggested next steps above before "
                              f"finalizing. Proceeding with these sites is reasonable if their network-coverage "
                              f"contribution is high enough to outweigh the causal-factor gap, but it should be a "
                              f"deliberate trade-off, not a default.")
        else:
            rec_lines.append("All opened sites scored at or above the 7.0 quality bar — no immediate causal-factor "
                              "concerns flagged.")

        rec_lines.append("**Overall:** combine both lenses — the optimizer picked sites that maximize network "
                          "coverage per site opened; the causal analysis checks whether those same locations are "
                          "actually good places to operate. Where they agree (high coverage contribution *and* high "
                          "causal score), proceed with confidence. Where they diverge, use the suggestions above to "
                          "decide whether to substitute a nearby alternative.")

        st.markdown(f'<div class="rec-card">{"<br><br>".join(rec_lines)}</div>', unsafe_allow_html=True)

    # =====================================================================
    # SCENARIO ASSISTANT — AI-assisted comparison across saved scenarios
    # =====================================================================
    st.divider()
    st.markdown('<div class="section-title">💬 Scenario Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Ask questions comparing your saved scenarios — coverage, service '
                 'distance, sites opened, cost. Save at least 2 scenarios (with a completed run) to compare.'
                 '</div>', unsafe_allow_html=True)

    runnable_scenarios = {name: snap for name, snap in st.session_state.scenarios.items() if snap.get("summary")}

    if len(runnable_scenarios) < 2:
        st.info(f"You have {len(runnable_scenarios)} scenario(s) with completed runs saved. Save at least 2 "
                "(use 'Save' in the sidebar after running each) to unlock comparisons.")
    elif not st.session_state.api_key:
        st.info("Enter an Anthropic API key in the sidebar (🤖 GenAI settings) to enable the comparison assistant.")
    else:
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg["content"])

        user_q = st.chat_input("e.g. 'Compare coverage across my scenarios'")
        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            intent, err = classify_comparison_intent(st.session_state.api_key, list(runnable_scenarios.keys()), user_q)

            comp_rows = []
            for name, snap in runnable_scenarios.items():
                s = snap["summary"]
                wavg = s.get("weighted_avg_distance_km")
                comp_rows.append({
                    "Scenario": ("⭐ " if name == st.session_state.baseline_scenario else "") + name,
                    "Sites opened": s["sites_selected"],
                    "Final coverage %": s["final_coverage_pct"],
                    "Weighted avg distance (mi)": round(wavg * 0.621371, 1) if wavg is not None else None,
                })
            comp_df = pd.DataFrame(comp_rows).set_index("Scenario")

            if err:
                reply = f"Couldn't reach the AI classifier ({err}) — showing the full comparison table instead."
                intent = "summary_table"
            elif intent == "coverage_comparison":
                reply = "Here's coverage % across your scenarios:"
            elif intent == "distance_comparison":
                reply = "Here's weighted average service distance across your scenarios:"
            elif intent == "sites_comparison":
                reply = "Here's sites opened across your scenarios:"
            else:
                reply = "Here's a full comparison across your scenarios:"

            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.session_state["_last_comparison_intent"] = intent
            st.session_state["_last_comparison_df"] = comp_df
            st.rerun()

        if "_last_comparison_df" in st.session_state:
            comp_df = st.session_state["_last_comparison_df"]
            intent = st.session_state.get("_last_comparison_intent", "summary_table")
            if intent == "coverage_comparison":
                st.bar_chart(comp_df[["Final coverage %"]])
            elif intent == "distance_comparison":
                st.bar_chart(comp_df[["Weighted avg distance (mi)"]])
            elif intent == "sites_comparison":
                st.bar_chart(comp_df[["Sites opened"]])
            else:
                st.dataframe(comp_df, width="stretch")

        if st.session_state.chat_history and st.button("Clear chat"):
            st.session_state.chat_history = []
            st.session_state.pop("_last_comparison_df", None)
            st.rerun()

st.markdown('<div class="app-footer">Supply Chain Design by AM · Greenfield MCLP engine · '
            'Reference research current as of Sept 2026</div>', unsafe_allow_html=True)
