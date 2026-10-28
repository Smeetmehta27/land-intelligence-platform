import pandas as pd
import geopandas as gpd

PHYSICAL = ("parcel_id","source_id","parcel_area_sqm","footprint_sqm","estimated_floors","is_vacant","source","retrieval_date")
NETWORK = ("access_road_width_m","network_dist_metro_m","in_tod_zone","poi_count_1km")
MEMBER2 = PHYSICAL + NETWORK

class SchemaError(ValueError): pass

def validate(frame: gpd.GeoDataFrame):
    missing = set(MEMBER2) - set(frame.columns)
    if missing: raise SchemaError(f"missing fields: {sorted(missing)}")
    if frame.crs is None or frame.crs.to_epsg() != 4326: raise SchemaError("output CRS must be EPSG:4326")
    if frame.geometry.isna().any() or (~frame.geometry.is_valid).any(): raise SchemaError("invalid or missing geometry")
    if not frame.geometry.geom_type.isin(["Polygon","MultiPolygon"]).all(): raise SchemaError("geometry must be polygonal")
    if frame.parcel_id.isna().any() or not frame.parcel_id.is_unique: raise SchemaError("parcel_id must be unique")
    if frame.source_id.isna().any() or not frame.source_id.is_unique: raise SchemaError("source_id must be unique")
    for c in ("parcel_area_sqm","footprint_sqm","estimated_floors"):
        if frame[c].isna().any() or (frame[c] <= 0).any(): raise SchemaError(f"{c} must be positive")
    if not (frame.parcel_area_sqm.round(6) == frame.footprint_sqm.round(6)).all(): raise SchemaError("V1 footprint must equal parcel area")
    if frame.is_vacant.astype(bool).any(): raise SchemaError("building-footprint records must not be vacant")
    if (frame.access_road_width_m.dropna() <= 0).any() or (frame.network_dist_metro_m.dropna() < 0).any() or (frame.poi_count_1km < 0).any(): raise SchemaError("invalid network values")
    if not frame.in_tod_zone.astype(bool).eq(frame.network_dist_metro_m.le(500).fillna(False)).all(): raise SchemaError("TOD flag is inconsistent")
    if not frame.source.eq("OpenStreetMap").all(): raise SchemaError("unexpected source")
    if pd.to_datetime(frame.retrieval_date, format="%Y-%m-%d", errors="coerce").isna().any(): raise SchemaError("retrieval_date must be ISO date")
