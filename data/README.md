# Data Directory & Lineage Reference

This directory contains the regulatory tables, raw source assets, diagnostic summaries, and instructions for acquiring the geospatial data layers used by the Land Intelligence Platform.

---

## Data Inventory & Provenance

| File Path | Producer / Origin | Size | In Git? | Description |
| :--- | :--- | :--- | :--- | :--- |
| `data/external/fsi_rules_table.json` | Member 4 | 1.04 KB | **Committed** | Statutory Floor Space Index (FSI) lookup matrix based on access road width and TOD transit-oriented bonus rules. |
| `data/raw/metro_stations.csv` | Member 2 | 238 B | **Committed** | GPS coordinates and line labels for operational and under-construction BMRCL Namma Metro stations along the corridor. |
| `data/raw/osm_water_bodies.geojson` | Member 3 / OpenStreetMap | 170 KB | **Committed** | Water body polygons (lakes, tanks, rajakaluve channels) queried from OpenStreetMap. |
| `data/raw/bgr_house_prices_kaggle.csv` | Kaggle / External | 938 KB | *Ignored* | Bengaluru House Price listings used to train the Automated Valuation Model. |
| `data/raw/srtm_dem_bgr.tif` | USGS SRTM / Member 3 | 24.76 MB | *Ignored* | Shuttle Radar Topography Mission (SRTM) 1-arc-second (30 m) digital elevation raster for Bengaluru. |
| `data/processed/member2_validation_report.json` | Member 2 | 308 B | **Committed** | Audit diagnostics verifying parcel footprint filtering (57,364 raw down to 56,336 valid units). |
| `data/processed/member4_diagnostics.json` | Member 4 | 1.6 KB | **Committed** | AHP eigenvalue decomposition, weight vectors, consistency ratio (`CR = 0.0248`), and NGT disqualification totals. |
| `data/processed/parcels_physical_validated.geojson` | Member 2 | 27.08 MB | *Ignored* | OSM building footprints clipped to study extent with area >= 25 m². |
| `data/processed/parcels_network_validated.geojson` | Member 2 | 33.30 MB | *Ignored* | Parcels annotated with pedestrian Dijkstra network walking distance to nearest metro station. |
| `data/processed/parcels_environmental.geojson` | Member 3 | 39.42 MB | *Ignored* | Parcels annotated with DEM elevation, slope in degrees, and distance to water bodies. |
| `data/processed/parcels_clustered.geojson` | Member 3 | 42.29 MB | *Ignored* | Parcels classified into 4 unsupervised spatial morphology archetypes via K-Means. |
| `data/processed/parcels_with_avm.geojson` | Member 1 | 50.44 MB | *Ignored* | Parcels merged with micro-market valuation features for XGBoost inference. |
| `data/processed/parcels_scored_final.geojson` | Member 4 | 71.80 MB | *Ignored* | **Primary dashboard input.** Fully scored parcels with capacity gap, AHP opportunity score, and NGT disqualification. |

---

## Data Download Instructions

### 1. Primary Dashboard Dataset (`parcels_scored_final.geojson`)
The interactive Streamlit dashboard (`app.py`) directly renders `data/processed/parcels_scored_final.geojson`. Because GitHub restricts files larger than 50 MB, this file (71.8 MB) is distributed as a GitHub Release asset:

* **Download URL:** `https://github.com/Smeetmehta27/land-intelligence-platform/releases/download/v1.0.0/parcels_scored_final.geojson` *(Placeholder: please upload the file to your GitHub Releases and confirm this tag)*
* **Placement:** Save the downloaded file directly into `data/processed/parcels_scored_final.geojson`.

### 2. Kaggle Bengaluru House Price Dataset (`bgr_house_prices_kaggle.csv`)
Used for training and validating the baseline valuation models (Ridge, Random Forest, LightGBM, XGBoost):
* **Source:** [Kaggle: Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhst/bengaluru-house-price-data)
* **Placement:** Download `bengaluru_house_prices.csv` (or renamed `bgr_house_prices_kaggle.csv`) and place it in `data/raw/`.

### 3. SRTM Digital Elevation Model (`srtm_dem_bgr.tif`)
Used by Member 3 for topographic slope and elevation analysis:
* **Source:** USGS Shuttle Radar Topography Mission (SRTM) 1-arc-second Global (30 m resolution), tile covering `12°N to 13°N, 77°E to 78°E`. Downloadable via [NASA Earthdata](https://search.earthdata.nasa.gov/) or [USGS EarthExplorer](https://earthexplorer.usgs.gov/).
* **Placement:** Save as `data/raw/srtm_dem_bgr.tif`.

---

## Attributions & Licensing

* **OpenStreetMap (OSM):** Map data, building footprints, and water body geometries are © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) and licensed under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/).
* **BMRCL Transit Data:** Metro station coordinates derived from Bangalore Metro Rail Corporation Limited public route guides.
* **Topography:** NASA JPL / USGS SRTM public domain digital elevation data.
