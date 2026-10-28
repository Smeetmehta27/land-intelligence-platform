
import json
import math
from pathlib import Path

import tifffile
from shapely.geometry import Point, shape
from shapely.ops import transform


# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

PARCEL_FILE = DATA / "processed/parcels_network_validated.geojson"
DEM_FILE = DATA / "raw/srtm_dem_bgr.tif"
WATER_FILE = DATA / "raw/osm_water_bodies.geojson"
OUTPUT_FILE = DATA / "processed/parcels_environmental.geojson"


# Convert WGS84 coordinates to UTM Zone 43N
def to_utm43(lon, lat, z=None):
    a = 6378137.0
    e2 = 0.00669437999014
    k0 = 0.9996

    lon0 = math.radians(75)
    lat = math.radians(lat)
    lon = math.radians(lon)

    ep2 = e2 / (1 - e2)
    N = a / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    T = math.tan(lat) ** 2
    C = ep2 * math.cos(lat) ** 2
    A = math.cos(lat) * (lon - lon0)

    M = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024)
        * math.sin(2 * lat)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024)
        * math.sin(4 * lat)
        - (35 * e2**3 / 3072) * math.sin(6 * lat)
    )

    x = 500000 + k0 * N * (
        A + (1 - T + C) * A**3 / 6
        + (5 - 18 * T + T**2 + 72 * C - 58 * ep2) * A**5 / 120
    )

    y = k0 * (
        M + N * math.tan(lat) * (
            A**2 / 2
            + (5 - T + 9 * C + 4 * C**2) * A**4 / 24
            + (61 - 58 * T + T**2 + 600 * C - 330 * ep2) * A**6 / 720
        )
    )

    if y < 0:
        y += 10000000

    return (x, y) if z is None else (x, y, z)


# Load input data
def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_dem():
    return tifffile.imread(DEM_FILE).astype("float32")


# Get elevation and slope from DEM
def get_elevation_and_slope(lon, lat, dem):
    pixel_size = 1 / 3600

    row = int((13 - lat) / pixel_size)
    col = int((lon - 77) / pixel_size)

    row = max(1, min(row, dem.shape[0] - 2))
    col = max(1, min(col, dem.shape[1] - 2))

    elevation = float(dem[row, col])

    dz_dx = (
        float(dem[row, col + 1]) - float(dem[row, col - 1])
    ) / 60

    dz_dy = (
        float(dem[row + 1, col]) - float(dem[row - 1, col])
    ) / 60

    slope = math.degrees(math.atan(math.sqrt(dz_dx**2 + dz_dy**2)))

    return elevation, slope


# Calculate environmental risk
def calculate_environmental_risk(
    distance, elevation, slope, min_elevation, max_elevation, ngt_violation
):
    if ngt_violation:
        return 1.0

    water_risk = max(0, 1 - distance / 300)

    elevation_range = max_elevation - min_elevation
    elevation_risk = (
        1 - (elevation - min_elevation) / elevation_range
        if elevation_range > 0 else 0
    )

    flat_risk = 1 if slope < 1 else 0

    risk = (
        0.5 * water_risk
        + 0.3 * elevation_risk
        + 0.2 * flat_risk
    )

    return max(0, min(1, risk))


def main():
    print("Loading input data...")

    parcels = load_json(PARCEL_FILE)
    water_data = load_json(WATER_FILE)
    dem = load_dem()

    # Convert water geometries to UTM metres
    water_geometries = [
        transform(to_utm43, shape(feature["geometry"]))
        for feature in water_data["features"]
        if feature.get("geometry")
    ]

    # Get elevation range
    valid_dem = dem[dem > 0]
    min_elevation = float(valid_dem.min())
    max_elevation = float(valid_dem.max())

    print("Processing parcels...")

    for i, feature in enumerate(parcels["features"]):
        if not feature.get("geometry"):
            continue

        parcel = shape(feature["geometry"])
        point = parcel.representative_point()

        lon, lat = point.x, point.y

        elevation, slope = get_elevation_and_slope(lon, lat, dem)

        point_utm = Point(*to_utm43(lon, lat))

        nearest_distance = min(
            (point_utm.distance(water) for water in water_geometries),
            default=float("inf")
        )

        # This is a water-distance proxy, not a true NGT boundary check
        ngt_violation = nearest_distance < 30.0

        risk = calculate_environmental_risk(
            nearest_distance,
            elevation,
            slope,
            min_elevation,
            max_elevation,
            ngt_violation
        )

        properties = feature["properties"]
        properties["elevation_m"] = elevation
        properties["slope_degrees"] = slope
        properties["dist_to_waterbody_m"] = nearest_distance
        properties["in_ngtn_buffer"] = ngt_violation
        properties["environmental_risk_score"] = risk

        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1} parcels")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(parcels, file)

    print("\nEnvironmental assessment completed!")
    print(f"Total parcels: {len(parcels['features'])}")
    print(f"Output saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
