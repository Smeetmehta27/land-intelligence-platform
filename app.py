from pathlib import Path
import json
import math as _math
import datetime as _dt

import geopandas as gpd
import pandas as pd
import numpy as np  # type: ignore
import streamlit as st  # type: ignore
import folium  # type: ignore
from streamlit_folium import st_folium  # type: ignore

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
SCORED_FILE = DATA / "parcels_scored_final.geojson"
DIAGNOSTICS_FILE = DATA / "member4_diagnostics.json"

# Numeric tolerance (+/- 0.0001 sqm) to prevent floating-point rounding noise near zero
# from incorrectly categorizing zero spare capacity as positive or negative.
CAPACITY_TOLERANCE = 1e-4

# Opportunity score threshold for screening candidates (0-100 scale).
# Parcels scoring >= 70.0 represent the upper tier of multi-criteria desirability
# (dataset median is 72.0, 75th percentile is 76.7).
HIGH_SCORE_THRESHOLD = 70.0

st.set_page_config(
    page_title="Land Intelligence | Member 4",
    page_icon="🏙️",
    layout="wide",
)

st.title("Land Intelligence Platform")
st.subheader("Member 4 — Regulatory Engine, Capacity Gap & Opportunity Screening")

# Prominent system-wide disclaimers and notices
st.warning(
    "**PROVISIONAL SCREENING RESULTS — TECHNICAL DEMONSTRATION ONLY**\n\n"
    "1. **Market Price Model Pending:** Member 1 automated valuation model (AVM) output (`predicted_price_sqft`) "
    "is unavailable. All parcels receive a neutral mid-scale default contribution (0.50), contributing a flat ~12.11 points "
    "out of 100. Current rankings are **not market-validated**.\n"
    "2. **Illustrative Zoning Rules:** FSI and TOD parameters are unverified research examples (`UNVERIFIED_EXAMPLE_VALUES`). "
    "They do not constitute legal zoning certifications.\n"
    "3. **Spatial Units:** Geometries represent OpenStreetMap building footprints, not legally verified cadastral parcels."
)

with st.expander("ℹ️ Conceptual Framework: Development Capacity vs. Investment Screening", expanded=False):
    st.markdown(
        """
        The platform distinguishes two fundamentally different analytical concepts:

        - **A. Estimated Development Capacity (Physical & Zoning Envelope):**
          Calculates the theoretical difference between permissible built area (based on footprint area, access road width, 
          and illustrative BDA/TOD FSI rules) and existing built area (footprint area × estimated floors).
          - **Positive estimated capacity (> 0 m²):** Potential spare building volume under example zoning.
          - **No estimated spare capacity (== 0 m²):** Existing structure matches permissible envelope.
          - **Negative estimated capacity — investigate (< 0 m²):** Existing built volume exceeds illustrative permissible FSI. 
            *Note:* This indicates a need for site-specific regulatory due diligence; it is **not** a confirmed finding of 
            illegal construction or code violation.

        - **B. Multi-Criteria Investment Screening (AHP Opportunity Score):**
          Aggregates multi-criteria spatial factors (metro accessibility, POI density, capacity gap, and environmental risk) 
          into a normalized 0–100 index.
          *Crucial Limitation:* A high AHP score reflects spatial desirability and accessibility, but is **NOT** a guarantee 
          of legal feasibility, clear property title, commercial profitability, or a positive redevelopment capacity gap.
        """
    )

if not SCORED_FILE.exists():
    st.error(f"Scored dataset not found: {SCORED_FILE}")
    st.info(
        "Run src/member4_regulatory_ahp.py first to generate "
        "parcels_scored_final.geojson."
    )
    st.stop()

@st.cache_data(show_spinner="Loading scored spatial data...")
def load_data(path, modified_time):
    return gpd.read_file(path)

gdf = load_data(str(SCORED_FILE), SCORED_FILE.stat().st_mtime)

if DIAGNOSTICS_FILE.exists():
    with DIAGNOSTICS_FILE.open("r", encoding="utf-8") as f:
        diagnostics = json.load(f)
else:
    diagnostics = {}

required = [
    "base_fsi", "tod_bonus_fsi", "total_permissible_fsi",
    "existing_built_sqm", "permissible_built_sqm",
    "capacity_gap_sqm", "land_opportunity_score", "opportunity_rank",
]
missing = [col for col in required if col not in gdf.columns]
if missing:
    st.error(f"Required columns missing: {missing}")
    st.stop()

# Convert numeric fields safely for filtering and summaries.
numeric_cols = [
    "base_fsi", "tod_bonus_fsi", "total_permissible_fsi",
    "existing_built_sqm", "permissible_built_sqm",
    "capacity_gap_sqm", "land_opportunity_score", "opportunity_rank",
    "network_dist_metro_m", "poi_count_1km",
    "environmental_risk_score", "predicted_price_sqft",
]
for col in numeric_cols:
    if col in gdf.columns:
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

if "in_ngtn_buffer" not in gdf.columns:
    gdf["in_ngtn_buffer"] = False
else:
    gdf["in_ngtn_buffer"] = gdf["in_ngtn_buffer"].map(
        lambda x: True if str(x).strip().lower() in ["true", "1", "yes"] else False
    )

# --- DERIVED FIELDS FOR PRESENTATION ---

# 1. Capacity Status Label
def compute_capacity_status(gap: float) -> str:
    if pd.isna(gap):
        return "Unknown capacity"
    if gap > CAPACITY_TOLERANCE:
        return "Positive estimated capacity"
    elif gap < -CAPACITY_TOLERANCE:
        return "Negative estimated capacity — investigate"
    else:
        return "No estimated spare capacity"

gdf["capacity_status"] = gdf["capacity_gap_sqm"].apply(compute_capacity_status)

# 2. Market Price Status Label
gdf["market_price_status"] = "Unavailable (AVM pending)"

# 3. Opportunity Classification
def compute_opportunity_classification(row) -> str:
    # Hard disqualifier: inside NGTN environmental buffer zone
    if bool(row.get("in_ngtn_buffer", False)):
        return "Environmental constraint"
    score = row.get("land_opportunity_score", 0.0)
    gap = row.get("capacity_gap_sqm", 0.0)
    if pd.isna(score) or pd.isna(gap):
        return "Further review required"
    if score >= HIGH_SCORE_THRESHOLD:
        if gap > CAPACITY_TOLERANCE:
            return "Screening candidate"
        elif gap < -CAPACITY_TOLERANCE:
            return "High score — negative capacity"
    return "Further review required"

gdf["opportunity_classification"] = gdf.apply(compute_opportunity_classification, axis=1)

st.caption(
    f"Dataset: {len(gdf):,} spatial features | CRS: {gdf.crs} | "
    f"Source: {SCORED_FILE.relative_to(ROOT)}"
)

# Sidebar filters
st.sidebar.header("Filter & Screening Controls")

score_min = float(gdf["land_opportunity_score"].min())
score_max = float(gdf["land_opportunity_score"].max())
score_range = st.sidebar.slider(
    "AHP Opportunity score range",
    min_value=score_min,
    max_value=score_max,
    value=(score_min, score_max),
)

gap_filter = st.sidebar.selectbox(
    "Development capacity status",
    [
        "All capacity statuses",
        "Positive estimated capacity (> 0 m²)",
        "No estimated spare capacity (== 0 m²)",
        "Negative estimated capacity — investigate (< 0 m²)",
    ],
)

classification_filter = st.sidebar.selectbox(
    "Opportunity classification",
    [
        "All classifications",
        "Screening candidate",
        "High score — negative capacity",
        "Environmental constraint",
        "Further review required",
    ],
)

show_excluded = st.sidebar.checkbox(
    "Include NGTN-buffer features in table/map",
    value=False,
    help="When unchecked, features within the National Green Tribunal buffer zone are excluded.",
)

map_style_choice = st.sidebar.radio(
    "Map visualization mode",
    [
        "Opportunity classification (Categorical)",
        "AHP opportunity score (Continuous gradient)",
    ],
)

# Apply filters
filtered = gdf[
    gdf["land_opportunity_score"].between(*score_range)
].copy()

if gap_filter == "Positive estimated capacity (> 0 m²)":
    filtered = filtered[filtered["capacity_gap_sqm"] > CAPACITY_TOLERANCE]
elif gap_filter == "No estimated spare capacity (== 0 m²)":
    filtered = filtered[filtered["capacity_gap_sqm"].abs() <= CAPACITY_TOLERANCE]
elif gap_filter == "Negative estimated capacity — investigate (< 0 m²)":
    filtered = filtered[filtered["capacity_gap_sqm"] < -CAPACITY_TOLERANCE]

if classification_filter != "All classifications":
    filtered = filtered[filtered["opportunity_classification"] == classification_filter]

if not show_excluded:
    filtered = filtered[~filtered["in_ngtn_buffer"]]

# --- STRUCTURED KPI METRIC CARDS ---
st.markdown("### Executive Screening Summary")

col_cap_box, col_scr_box = st.columns(2)

with col_cap_box:
    st.markdown("##### 📐 Estimated Development Capacity (Physical & Zoning)")
    c1, c2, c3 = st.columns(3)
    pos_count = int((filtered["capacity_gap_sqm"] > CAPACITY_TOLERANCE).sum())
    zero_count = int((filtered["capacity_gap_sqm"].abs() <= CAPACITY_TOLERANCE).sum())
    neg_count = int((filtered["capacity_gap_sqm"] < -CAPACITY_TOLERANCE).sum())
    c1.metric("Positive Capacity", f"{pos_count:,}", help="Permissible built area > existing built area")
    c2.metric("No Spare Capacity", f"{zero_count:,}", help="Permissible built area == existing built area")
    c3.metric("Negative Capacity", f"{neg_count:,}", help="Existing built area > illustrative permissible — investigate")

with col_scr_box:
    st.markdown("##### 🎯 Investment Screening (Multi-Criteria Heuristic)")
    s1, s2, s3 = st.columns(3)
    s1.metric("Features in View", f"{len(filtered):,}")
    s2.metric(
        "Mean AHP Score",
        f"{filtered['land_opportunity_score'].mean():.2f}" if not filtered.empty else "—",
    )
    cand_count = int((filtered["opportunity_classification"] == "Screening candidate").sum())
    s3.metric("Screening Candidates", f"{cand_count:,}", help="Score >= 70, positive capacity, outside NGTN buffer")

tab_overview, tab_map, tab_rank, tab_rules = st.tabs(
    ["Overview", "Opportunity Map", "Rankings & Export", "Rules & AHP"]
)

with tab_overview:
    st.markdown("### Capacity-Gap Analysis & Distribution")

    if filtered.empty:
        st.info("No features match the current filters.")
    else:
        ov_col1, ov_col2 = st.columns(2)
        with ov_col1:
            st.markdown("##### Development Capacity Status Breakdown")
            gap_summary = pd.DataFrame({
                "Status": [
                    "Positive estimated capacity",
                    "No estimated spare capacity",
                    "Negative estimated capacity — investigate",
                ],
                "Count": [
                    pos_count,
                    zero_count,
                    neg_count,
                ],
            }).set_index("Status")
            st.bar_chart(gap_summary)

        with ov_col2:
            st.markdown("##### Opportunity Classification Breakdown")
            class_counts = filtered["opportunity_classification"].value_counts()
            class_summary = pd.DataFrame({
                "Classification": class_counts.index,
                "Count": class_counts.values,
            }).set_index("Classification")
            st.bar_chart(class_summary)

        st.markdown("##### Capacity Gap Curve (Sorted Distribution in m²)")
        st.line_chart(
            filtered["capacity_gap_sqm"]
            .dropna()
            .sort_values()
            .reset_index(drop=True)
        )

    st.markdown("### Due Diligence & Analytical Findings")
    st.info(
        "**Why do top-scoring parcels often have negative capacity gaps?**\n\n"
        "In the Koramangala–HSR–Bellandur corridor, most existing buildings surveyed have 2 or more floors. "
        "Under the illustrative base FSI rules (1.50–1.75), an existing 2-storey building with 100% ground footprint coverage "
        "already utilizes an FSI of 2.0, creating a negative capacity gap (e.g., permissible built 150 m² vs existing 200 m² = -50 m²). "
        "Because 98.35% of all corridor features have negative capacity gaps, parcels with modest negative gaps (-30 to -50 m²) "
        "rank in the top percentiles of the capacity gap criterion. When combined with close metro distance and dense commercial POIs, "
        "these parcels receive high composite AHP scores. "
        "**Therefore, high AHP score must never be conflated with spare development potential.**"
    )

    st.markdown("### Data Quality & Model Limitations")
    price_missing = (
        "predicted_price_sqft" not in gdf.columns
        or gdf["predicted_price_sqft"].isna().all()
    )
    if price_missing:
        st.warning(
            "**Market Valuation Model Status: Unavailable (AVM Pending)**\n\n"
            "Member 1 automated valuation model (AVM) output (`predicted_price_sqft`) is 100% missing. "
            "In the AHP engine, all parcels are assigned a neutral mid-scale score of 0.50 for the market criterion. "
            "Because this criterion has an AHP weight of 24.22%, it contributes a uniform ~12.11 points to every parcel. "
            "As a result, relative parcel rankings are driven exclusively by transit, POI density, capacity gap, and environmental risk. "
            "Market-based investment prioritization is incomplete, and the platform makes **no claim of market price validation**."
        )
    st.write(
        "- **Spatial resolution:** Features are OpenStreetMap building footprints, not cadastral parcels.\n"
        "- **Regulatory assumptions:** FSI parameters are research examples (`UNVERIFIED_EXAMPLE_VALUES`).\n"
        "- **Negative capacity gaps:** Reflect estimated existing built envelope exceeding example permissible FSI. "
        "These require site-specific due diligence and are **not** legal determinations of violation."
    )

with tab_map:
    st.markdown("### Spatial Opportunity Map")

    if filtered.empty:
        st.info("No features match the current filters.")
    else:
        map_data = filtered
        if len(map_data) > 5000:
            st.info(
                "Showing a sample of 5,000 features for smooth map rendering. "
                "The full filtered set is preserved in table and download."
            )
            map_data = map_data.sample(5000, random_state=42)

        if map_data.crs is None:
            st.error("Input geometries have no CRS; cannot safely map.")
        else:
            map_data = map_data.to_crs("EPSG:4326")
            # Compute centroid in projected metric CRS (EPSG:32643) to avoid geographic-CRS warning
            _proj = map_data.to_crs("EPSG:32643")
            _centroids_4326 = _proj.geometry.centroid.to_crs("EPSG:4326")
            if _centroids_4326.empty or not (
                _centroids_4326.y.notna().any() and _centroids_4326.x.notna().any()
            ):
                center = [12.935, 77.645]  # Study-area fallback coordinates
            else:
                center = [
                    float(_centroids_4326.y.mean()),
                    float(_centroids_4326.x.mean()),
                ]

            fmap = folium.Map(
                location=center,
                zoom_start=13,
                tiles="OpenStreetMap",
                control_scale=True,
            )

            score_min_map = float(gdf["land_opportunity_score"].min())
            score_max_map = float(gdf["land_opportunity_score"].max())

            # Categorical color scheme
            CLASS_COLORS = {
                "Screening candidate": "#10b981",          # Emerald Green
                "High score — negative capacity": "#f59e0b", # Amber / Orange
                "Environmental constraint": "#ef4444",       # Crimson Red
                "Further review required": "#64748b",        # Slate Grey
            }

            def feature_style(feature):
                props = feature["properties"]
                if map_style_choice == "Opportunity classification (Categorical)":
                    classification = props.get("opportunity_classification", "Further review required")
                    color = CLASS_COLORS.get(classification, "#64748b")
                else:
                    score = props.get("land_opportunity_score", score_min_map)
                    try:
                        score = float(score)
                    except (TypeError, ValueError):
                        score = score_min_map
                    ratio = (score - score_min_map) / (
                        score_max_map - score_min_map
                    ) if score_max_map > score_min_map else 0.5
                    red = int(70 + 180 * ratio)
                    blue = int(200 - 150 * ratio)
                    color = f"#{red:02x}65{blue:02x}"

                return {
                    "color": color,
                    "weight": 1.2,
                    "fillColor": color,
                    "fillOpacity": 0.60,
                }

            # Prepare columns and sanitize for JSON serialization
            map_json = map_data.copy()

            # Clean formatting for tooltip display
            map_json["capacity_gap_sqm"] = map_json["capacity_gap_sqm"].round(2)
            map_json["land_opportunity_score"] = map_json["land_opportunity_score"].round(2)
            map_json["permissible_built_sqm"] = map_json["permissible_built_sqm"].round(2)
            map_json["existing_built_sqm"] = map_json["existing_built_sqm"].round(2)
            map_json["in_ngtn_buffer_str"] = map_json["in_ngtn_buffer"].apply(lambda b: "Yes (Disqualified)" if b else "No")

            tooltip_fields = [
                "parcel_id",
                "opportunity_classification",
                "capacity_status",
                "land_opportunity_score",
                "opportunity_rank",
                "capacity_gap_sqm",
                "total_permissible_fsi",
                "permissible_built_sqm",
                "existing_built_sqm",
                "in_ngtn_buffer_str",
                "market_price_status",
            ]
            tooltip_aliases = [
                "Parcel ID: ",
                "Opportunity Class: ",
                "Capacity Status: ",
                "AHP Opportunity Score: ",
                "Corridor Rank: ",
                "Capacity Gap (m²): ",
                "Permissible FSI: ",
                "Permissible Built (m²): ",
                "Existing Built (m²): ",
                "Inside NGTN Buffer: ",
                "Market Valuation: ",
            ]

            # Datetime and non-finite numeric sanitization
            for _col in map_json.columns:
                if _col == "geometry":
                    continue
                _s = map_json[_col]
                if hasattr(_s, "dt") or str(_s.dtype).startswith("datetime"):
                    map_json[_col] = _s.dt.strftime("%Y-%m-%d").where(_s.notna(), other=None)
                    continue
                if _s.dtype == object:
                    def _safe(v):
                        if v is None:
                            return v
                        if isinstance(v, (_dt.datetime, _dt.date)):
                            return v.isoformat()
                        try:
                            if isinstance(v, pd.Timestamp):
                                return v.isoformat()
                        except Exception:
                            pass
                        try:
                            if isinstance(v, np.generic):
                                py_v = v.item()
                                if isinstance(py_v, float) and not _math.isfinite(py_v):
                                    return None
                                return py_v
                        except Exception:
                            pass
                        return v
                    map_json[_col] = _s.map(_safe)
                    continue
                try:
                    _num = pd.to_numeric(_s, errors="coerce")
                    if _num.apply(lambda v: v is not None and _math.isfinite(float(v)) if pd.notna(v) else True).all():
                        continue
                    map_json[_col] = _s.where(_s.apply(
                        lambda v: pd.isna(v) or _math.isfinite(float(v))
                        if not isinstance(v, str) else True
                    ), other=None)
                except Exception:
                    pass

            folium.GeoJson(
                data=map_json.to_json(),
                name="Opportunity Features",
                style_function=feature_style,
                tooltip=folium.GeoJsonTooltip(
                    fields=tooltip_fields,
                    aliases=tooltip_aliases,
                    sticky=False,
                ),
            ).add_to(fmap)

            folium.LayerControl().add_to(fmap)
            st_folium(fmap, use_container_width=True, height=600)

            # Map Legend
            if map_style_choice == "Opportunity classification (Categorical)":
                st.markdown(
                    """
                    **Map Legend:**
                    - 🟩 **Screening candidate** (High AHP score ≥ 70, positive estimated capacity, outside NGTN buffer)
                    - 🟧 **High score — negative capacity** (High AHP score ≥ 70, negative capacity gap — site investigation required)
                    - 🟥 **Environmental constraint** (Inside NGTN buffer — hard disqualifier, score = 0)
                    - ⬜ **Further review required** (Score < 70 or zero spare capacity)
                    """
                )
            else:
                st.markdown(
                    "**Map Legend:** Continuous gradient from 🟦 Low AHP Opportunity Score to 🟥 High AHP Opportunity Score."
                )

with tab_rank:
    st.markdown("### Corridor Parcel Rankings & Export")

    if filtered.empty:
        st.info("No features match the current filters.")
    else:
        st.warning(
            "⚠️ **Analytical Note:** The default ranking is ordered by `land_opportunity_score`. "
            "Notice that high-scoring parcels may have negative capacity gaps. "
            "To view parcels with spare development envelope, filter by `Screening candidate` or sort by `capacity_gap_sqm`."
        )

        sort_col = st.selectbox(
            "Sort table by",
            [
                "land_opportunity_score (Highest opportunity first)",
                "capacity_gap_sqm (Largest positive capacity first)",
                "capacity_gap_sqm (Largest negative capacity first)",
                "opportunity_rank (Ascending rank 1 to N)",
            ],
        )

        if "Highest opportunity" in sort_col:
            actual_sort = "land_opportunity_score"
            ascending = False
        elif "Largest positive capacity" in sort_col:
            actual_sort = "capacity_gap_sqm"
            ascending = False
        elif "Largest negative capacity" in sort_col:
            actual_sort = "capacity_gap_sqm"
            ascending = True
        else:
            actual_sort = "opportunity_rank"
            ascending = True

        table_cols = [
            col for col in [
                "parcel_id", "opportunity_rank", "land_opportunity_score",
                "opportunity_classification", "capacity_status",
                "capacity_gap_sqm", "base_fsi", "tod_bonus_fsi", "total_permissible_fsi",
                "permissible_built_sqm", "existing_built_sqm",
                "market_price_status", "predicted_price_sqft",
                "network_dist_metro_m", "poi_count_1km",
                "environmental_risk_score", "in_ngtn_buffer",
            ] if col in filtered.columns
        ]

        table = filtered.sort_values(
            actual_sort, ascending=ascending, na_position="last"
        )[table_cols]

        st.dataframe(table.head(1000), use_container_width=True, hide_index=True)

        st.download_button(
            "Download filtered results as CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="member4_opportunity_screening.csv",
            mime="text/csv",
        )

with tab_rules:
    st.markdown("### Regulatory Framework & Illustrative FSI Rules")

    rules_path = ROOT / "data" / "external" / "fsi_rules_table.json"
    if rules_path.exists():
        with rules_path.open("r", encoding="utf-8") as f:
            rules_payload = json.load(f)
        st.warning(
            f"FSI Rule Status: "
            f"{rules_payload.get('metadata', {}).get('status', 'UNKNOWN')}"
        )
        st.json(rules_payload.get("metadata", {}))
        st.dataframe(
            pd.DataFrame(rules_payload.get("zoning_rules", [])),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("FSI rules JSON was not found.")

    st.markdown("### AHP Multi-Criteria Decision Framework")
    col_cr1, col_cr2 = st.columns(2)
    with col_cr1:
        st.metric(
            "AHP Consistency Ratio (CR)",
            f"{diagnostics.get('ahp_consistency_ratio', float('nan')):.4f}"
            if diagnostics.get("ahp_consistency_ratio") is not None else "Unavailable",
            help="Threshold for consistency pass is CR < 0.10",
        )
    with col_cr2:
        st.metric(
            "Consistency Test Status",
            "PASS (CR < 0.10)" if diagnostics.get("ahp_consistency_pass") else "FAIL",
        )

    st.markdown("##### Criteria Weights & Contribution Analysis")
    weights = diagnostics.get("ahp_weights", {})
    if weights:
        weight_table = [
            {
                "Criterion": "capacity_gap_sqm",
                "Weight": weights.get("capacity_gap_sqm", 0.0),
                "Direction": "Higher is better (+)",
                "Status in Model": "Active",
                "Interpretation": "Estimated spare permissible built area over existing built volume.",
            },
            {
                "Criterion": "predicted_price_sqft",
                "Weight": weights.get("predicted_price_sqft", 0.0),
                "Direction": "Higher is better (+)",
                "Status in Model": "⚠️ Inactive / Neutral Contribution (0.50)",
                "Interpretation": "Member 1 AVM unavailable. Imparts constant ~12.11 pts to all parcels; no price differentiation.",
            },
            {
                "Criterion": "network_dist_metro_m",
                "Weight": weights.get("network_dist_metro_m", 0.0),
                "Direction": "Lower is better (Inverse -)",
                "Status in Model": "Active",
                "Interpretation": "Shortest walking/road network distance to operational metro stations.",
            },
            {
                "Criterion": "poi_count_1km",
                "Weight": weights.get("poi_count_1km", 0.0),
                "Direction": "Higher is better (+)",
                "Status in Model": "Active",
                "Interpretation": "Count of commercial, institutional, and amenity POIs within 1 km radius.",
            },
            {
                "Criterion": "environmental_risk_score",
                "Weight": weights.get("environmental_risk_score", 0.0),
                "Direction": "Lower is better (Inverse -)",
                "Status in Model": "Active",
                "Interpretation": "Member 3 composite environmental hazard. Inversely scaled (1 - normalized risk) so higher risk strictly lowers score.",
            },
        ]
        w_df = pd.DataFrame(weight_table)
        st.dataframe(w_df, use_container_width=True, hide_index=True)

        chart_df = pd.DataFrame(
            {"Criterion": list(weights.keys()), "Weight (%)": [w * 100 for w in weights.values()]}
        ).set_index("Criterion")
        st.bar_chart(chart_df)

    st.markdown("### Critical Due Diligence Guidelines")
    st.write(
        "1. **Do not treat AHP score as financial feasibility:** The score is a technical multi-criteria index. "
        "It does not assess land assembly costs, title litigation, acquisition availability, or investor ROI."
    )
    st.write(
        "2. **Negative capacity gaps require investigation:** An existing building with estimated built area larger "
        "than permissible FSI may reflect historic regulations, compounding approvals, or data footprint assumptions. "
        "Do not classify it as illegal without formal municipal scrutiny."
    )
    st.write(
        "3. **AVM integration requirement:** Before using this ranking for capital allocation, complete Member 1 "
        "model training and integrate actual `predicted_price_sqft` values."
    )
    st.write(
        "4. **Cadastral ground truth:** OSM building polygons must be replaced with surveyed land parcel boundaries."
    )
