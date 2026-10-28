import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("member4", ROOT / "src" / "member4_regulatory_ahp.py")
member4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(member4)


def test_ahp_consistency_and_weights():
    weights, _, cr = member4.get_ahp_weights()
    assert np.isclose(weights.sum(), 1.0)
    assert cr < 0.10


def test_fsi_thresholds_and_tod_bonus():
    rules, _ = member4.load_rules(ROOT / "data" / "external" / "fsi_rules_table.json")
    assert member4.evaluate_fsi(18, True, rules)[:3] == (2.5, 0.5, 3.0)
    assert member4.evaluate_fsi(12, False, rules)[:3] == (2.0, 0.0, 2.0)
    assert member4.evaluate_fsi(7.5, False, rules)[:3] == (1.5, 0.0, 1.5)


def test_scoring_and_ngtn_disqualifier():
    rules, _ = member4.load_rules(ROOT / "data" / "external" / "fsi_rules_table.json")
    gdf = gpd.GeoDataFrame({
        "parcel_id": ["A", "B"], "parcel_area_sqm": [1000, 1000],
        "footprint_sqm": [300, 500], "estimated_floors": [2, 2],
        "access_road_width_m": [18, 9], "in_tod_zone": [True, False],
        "predicted_price_sqft": [15000, 9000], "network_dist_metro_m": [300, 1200],
        "poi_count_1km": [100, 20], "environmental_risk_score": [0.1, 0.8],
        "in_ngtn_buffer": [False, True],
    }, geometry=[box(77.62, 12.91, 77.621, 12.911), box(77.63, 12.92, 77.631, 12.921)], crs="EPSG:4326")
    scored, diag = member4.score_parcels(gdf, rules)
    assert scored.loc[0, "capacity_gap_sqm"] == 2400
    assert scored.loc[1, "land_opportunity_score"] == 0
    assert scored["opportunity_rank"].min() == 1
    assert diag["ahp_consistency_pass"] is True
