# Supply Chain Design by AM

Greenfield facility-location optimizer with existing-network awareness.

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501

## What's new in this version
1. **Exact site count** — "Number of new sites" is now a typed number input, not a range slider.
2. **Renamed** to "Supply Chain Design by AM" with a yellow/white/blue theme (`.streamlit/config.toml` + custom CSS in `app.py`).
3. **Three input tables** (Data Input tab), each editable inline or uploadable as CSV:
   - **Customer Demand**: `lat`, `lon`, `daily_orders`
   - **Products**: `product_id`, `product_name`, `unit_weight_kg` (currently reference-only — see "Not yet built" below)
   - **Existing Facilities**: `facility_id`, `facility_name`, `lat`, `lon` — their coverage counts as a baseline before new sites are added
4. **Two optimization modes** (sidebar):
   - **Number of new sites** — you set an exact count, optimizer picks the best combination to maximize coverage
   - **Service coverage target (%)** — you set a target (e.g. 80% of demand within X km), optimizer opens the minimum number of sites (up to a safety cap) to hit it
5. **Optimizer upgraded** from independent per-site scoring to a **greedy maximal-coverage algorithm** — it picks sites based on *incremental* new coverage, not just standalone scores, so it no longer double-counts demand that overlapping candidates would both claim.

## Not yet built (flagging honestly)
- **Product-level capacity constraints**: the Products table is currently reference-only. A full Llamasoft-style model would tie product weight/volume to facility capacity and solve a constrained allocation problem (which facility serves which customer for which product) — that's a meaningfully bigger optimization (transportation/allocation problem, not just coverage) and is a good next milestone, not a quick add.
- **Existing facility closure/consolidation** — right now existing facilities are treated as fixed and always-open; the model doesn't yet consider closing or downsizing them.

## Next steps
- Try it with your own demand/existing-facility data
- Deploy to Streamlit Community Cloud for a shareable link
- If product-capacity constraints matter for your decision, that's the next real feature to scope
