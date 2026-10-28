from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class BBox:
    south: float
    north: float
    west: float
    east: float
    def __post_init__(self):
        if not (-90 <= self.south < self.north <= 90 and -180 <= self.west < self.east <= 180):
            raise ValueError("Invalid WGS84 bounds")
    @property
    def osmnx(self):
        return (self.west, self.south, self.east, self.north)
    def expand(self, degrees: float):
        if degrees < 0: raise ValueError("degrees must be non-negative")
        return BBox(self.south-degrees, self.north+degrees, self.west-degrees, self.east+degrees)
    def include(self, lons, lats):
        if len(lons) != len(lats) or not lons: raise ValueError("coordinate lists must be non-empty and equal")
        return BBox(min(self.south,min(lats)), max(self.north,max(lats)), min(self.west,min(lons)), max(self.east,max(lons)))

@dataclass(frozen=True)
class Config:
    root: Path
    study: BBox = BBox(12.90, 12.97, 77.61, 77.67)
    network_buffer_deg: float = 0.03
    geographic_crs: str = "EPSG:4326"
    metric_crs: str = "EPSG:32643"
    tod_m: float = 500.0
    poi_radius_m: float = 1000.0
    @property
    def raw(self): return self.root/"data"/"raw"
    @property
    def interim(self): return self.root/"data"/"interim"
    @property
    def processed(self): return self.root/"data"/"processed"
    @property
    def stations(self): return self.raw/"metro_stations.csv"
    def mkdirs(self):
        for p in (self.raw/"osm"/"buildings", self.raw/"osm"/"roads", self.raw/"osm"/"pois", self.interim, self.processed): p.mkdir(parents=True, exist_ok=True)
