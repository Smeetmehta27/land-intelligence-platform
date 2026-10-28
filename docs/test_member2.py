import pandas as pd
from pathlib import Path
from src.config import BBox, Config
from src.member2_geospatial import estimate_floors, load_stations, network_bbox

def test_study_box_is_fixed_and_network_box_expands():
    c=Config(Path("."))
    b=network_bbox(c)
    assert c.study.east == 77.67
    assert b.east > c.study.east
    assert b.south < c.study.south

def test_floor_priority_and_fallback():
    assert estimate_floors({"building:levels":"4","building":"office"}) == 4
    assert estimate_floors({"building:levels":None,"building":"office"}) == 5
    assert estimate_floors({"building:levels":None,"building":"unknown"}) == 2

def test_station_input_has_six_rows():
    stations=load_stations(Path("data/raw/metro_stations.csv"))
    assert len(stations)==6
    assert stations.station_id.is_unique
