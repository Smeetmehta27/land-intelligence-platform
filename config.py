"""Shared project configuration for the AI-Powered Land Intelligence Platform.

This root-level module is the compatibility contract used by Member 1's AVM.
Member 2's existing src/config.py (BBox/Config API) is intentionally preserved.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Study extent: (north, south, east, west), matching the Member 1 guide.
BBOX = (12.97, 12.90, 77.67, 77.61)
CRS_METRIC = "EPSG:32643"
CRS_GEO = "EPSG:4326"

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"

AVM_TARGET_COL = "price_per_sqft"
AVM_FEATURE_COLS = [
    "access_road_width_m",
    "network_dist_metro_m",
    "in_tod_zone",
    "poi_count_1km",
    "elevation_m",
    "slope_degrees",
    "footprint_sqm",
]

# Road-width hierarchy is an imputation contract, not a verified legal rule.
ROAD_WIDTH_FALLBACK_M = {
    "motorway": 24.0,
    "trunk": 24.0,
    "primary": 18.0,
    "secondary": 12.0,
    "tertiary": 9.0,
    "residential": 7.5,
    "service": 6.0,
    "unclassified": 6.0,
}
TOD_DISTANCE_M = 500.0
POI_RADIUS_M = 1000.0
RANDOM_STATE = 42
