"""Canonical shared schema for all four project members.

This file documents cross-member fields and offers lightweight validation helpers.
The stricter Member 2 V1 validator remains at src/schema.py and is not overwritten.
"""
from __future__ import annotations
from typing import Any

# name: (owner, dtype, unit, description)
COLUMN_CONTRACT: dict[str, tuple[str, str, str, str]] = {
    # Common spatial identity and physical features (Member 2)
    "parcel_id": ("Member 2", "string", "identifier", "Deterministic spatial-unit ID"),
    "source_id": ("Member 2", "string", "identifier", "Source feature ID"),
    "parcel_area_sqm": ("Member 2", "float64", "m²", "Area measured in EPSG:32643"),
    "footprint_sqm": ("Member 2", "float64", "m²", "Building footprint area"),
    "estimated_floors": ("Member 2", "int64", "count", "Observed or imputed floor count"),
    "is_vacant": ("Member 2", "bool", "boolean", "Vacancy flag"),
    "access_road_width_m": ("Member 2", "float64", "m", "Nearest road width; may be imputed"),
    "network_dist_metro_m": ("Member 2", "float64", "m", "Walking-network distance to metro"),
    "in_tod_zone": ("Member 2", "bool", "boolean", "Walking distance to metro <= 500 m"),
    "poi_count_1km": ("Member 2", "int64", "count", "POI count within 1 km as defined by Member 2"),
    "source": ("Member 2", "string", "label", "Data source label"),
    "retrieval_date": ("Member 2", "string", "YYYY-MM-DD", "Source retrieval date"),
    # Environmental features (Member 3)
    "elevation_m": ("Member 3", "float64", "m", "DEM elevation at representative point"),
    "slope_degrees": ("Member 3", "float64", "degrees", "Terrain slope"),
    "dist_to_waterbody_m": ("Member 3", "float64", "m", "Distance to nearest mapped waterbody"),
    "in_ngtn_buffer": ("Member 3", "bool", "boolean", "NGT buffer flag as defined by project"),
    "environmental_risk_score": ("Member 3", "float64", "[0,1]", "Composite environmental risk estimate"),
    "ml_cluster_id": ("Member 3", "int64", "cluster ID", "Numeric cluster identifier"),
    "ml_cluster_label": ("Member 3", "string", "label", "Semantic cluster label"),
    # Compatibility aliases found in current Member 3 output; normalize at integration boundary.
    "cluster": ("Member 3 legacy output", "int64", "cluster ID", "Legacy alias for ml_cluster_id"),
    "cluster_name": ("Member 3 legacy output", "string", "label", "Legacy alias for ml_cluster_label"),
    # AVM outputs (Member 1)
    "price_per_sqft": ("Member 1", "float64", "INR/sq.ft", "Training target"),
    "predicted_price_sqft": ("Member 1", "float64", "INR/sq.ft", "Predicted price per square foot"),
    "shap_top_positive_feature": ("Member 1", "string", "feature name", "Strongest positive SHAP driver"),
    "shap_top_negative_feature": ("Member 1", "string", "feature name", "Strongest negative SHAP driver"),
    # Regulatory/capacity/AHP outputs (Member 4)
    "base_fsi": ("Member 4", "float64", "ratio", "Illustrative base FSI from road-width lookup"),
    "tod_bonus_fsi": ("Member 4", "float64", "ratio", "Illustrative TOD bonus FSI"),
    "total_permissible_fsi": ("Member 4", "float64", "ratio", "base plus TOD bonus FSI"),
    "existing_built_sqm": ("Member 4", "float64", "m²", "Footprint area times estimated floors"),
    "permissible_built_sqm": ("Member 4", "float64", "m²", "Area times illustrative total FSI"),
    "capacity_gap_sqm": ("Member 4", "float64", "m²", "Permissible less estimated existing built area"),
    "land_opportunity_score": ("Member 4", "float64", "[0,100]", "AHP composite opportunity score"),
    "opportunity_rank": ("Member 4", "int64", "rank", "Rank 1 is highest score"),
}

MEMBER_OUTPUTS = {
    "member2": ("parcel_id", "parcel_area_sqm", "footprint_sqm", "estimated_floors", "is_vacant", "access_road_width_m", "network_dist_metro_m", "in_tod_zone", "poi_count_1km"),
    "member3_environment": ("elevation_m", "slope_degrees", "dist_to_waterbody_m", "in_ngtn_buffer", "environmental_risk_score"),
    "member3_clustering": ("ml_cluster_id", "ml_cluster_label"),
    "member1": ("predicted_price_sqft", "shap_top_positive_feature", "shap_top_negative_feature"),
    "member4": ("base_fsi", "tod_bonus_fsi", "total_permissible_fsi", "existing_built_sqm", "permissible_built_sqm", "capacity_gap_sqm", "land_opportunity_score", "opportunity_rank"),
}

class SchemaError(ValueError):
    """Raised when a dataset violates the shared column contract."""

def missing_columns(frame: Any, required: tuple[str, ...] | list[str]) -> list[str]:
    """Return required fields absent from a DataFrame/GeoDataFrame-like object."""
    return sorted(set(required) - set(frame.columns))

def validate_columns(frame: Any, required: tuple[str, ...] | list[str], *, dataset_name: str = "dataset") -> None:
    """Raise SchemaError if required canonical columns are absent."""
    missing = missing_columns(frame, required)
    if missing:
        raise SchemaError(f"{dataset_name} is missing required columns: {missing}")

def normalize_cluster_aliases(frame: Any) -> Any:
    """Return a copy with Member 3's legacy cluster columns mapped to canonical names.

    Does not delete the original aliases; this is safe for existing consumers.
    """
    result = frame.copy()
    if "ml_cluster_id" not in result.columns and "cluster" in result.columns:
        result["ml_cluster_id"] = result["cluster"]
    if "ml_cluster_label" not in result.columns and "cluster_name" in result.columns:
        result["ml_cluster_label"] = result["cluster_name"]
    return result
