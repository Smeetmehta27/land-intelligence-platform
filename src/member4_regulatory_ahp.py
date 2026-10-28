
# ===================== STEP 0: IMPORT LIBRARIES =====================
import json                  # to read the FSI rules file and save the report
from pathlib import Path     # for filesystem paths
import numpy as np           # for the AHP matrix maths
import pandas as pd          # for tables and columns
import geopandas as gpd      # pandas + map geometry (reads/writes GeoJSON)

# ===================== STEP 1: FILE PATHS AND SETTINGS =====================
RULES_FILE = Path("data/external/fsi_rules_table.json")             # FSI rules (road width -> FSI)
INPUT_FILE = Path("data/processed/parcels_with_avm.geojson")        # data coming from Members 1, 2, 3
OUTPUT_FILE = Path("data/processed/parcels_scored_final.geojson")   # final file we create

# AHP pairwise matrix: row i vs column j = "how much more important is i than j?"
# Order: capacity gap, price, metro distance, POI count, environment
MATRIX = np.array([
    [1.0,  2.0,  3.0, 3.0, 4.0],   # capacity gap
    [0.5,  1.0,  2.0, 2.0, 3.0],   # price
    [1/3,  0.5,  1.0, 2.0, 3.0],   # metro distance
    [1/3,  0.5,  0.5, 1.0, 2.0],   # POI count
    [0.25, 1/3,  1/3, 0.5, 1.0],   # environment
])

# ===================== STEP 2: LOAD THE FSI RULES =====================
with open(RULES_FILE) as f:                      # open the rules file
    rules = json.load(f)["zoning_rules"]         # read the list of rules from it
# sort rules from widest road (18 m) to narrowest road (0 m)
rules = sorted(rules, key=lambda r: r["road_min_m"], reverse=True)


# ===================== STEP 3: FSI FOR ONE PARCEL =====================
def get_fsi(road_width, in_tod):
    """Look up base FSI, TOD bonus and total FSI from the road width."""
    if pd.isna(road_width):          # if the road width is missing...
        road_width = 0               # ...treat it as a narrow road (safe choice)
    if pd.isna(in_tod):              # if the TOD flag is missing...
        in_tod = False               # ...treat it as "not in a TOD zone"
    for rule in rules:                               # check rules from widest to narrowest
        if road_width >= rule["road_min_m"]:         # first rule the road qualifies for
            base = rule["base_fsi"]                  # base FSI of that rule
            bonus = rule["tod_bonus"] if in_tod else 0.0   # TOD bonus only inside a TOD zone
            return base, bonus, base + bonus, rule["zone"] # base, bonus, total, zone name


# ===================== STEP 4: AHP WEIGHTS + CONSISTENCY CHECK =====================
def get_ahp_weights(matrix):
    """Return the 5 criteria weights and the consistency ratio (CR)."""
    n = len(matrix)                                         # number of criteria (5)
    geo_mean = np.prod(matrix, axis=1) ** (1 / n)           # geometric mean of each row
    weights = geo_mean / geo_mean.sum()                     # divide by total so weights add to 1
    lambda_max = np.mean((matrix @ weights) / weights)      # principal eigenvalue estimate
    ci = max(0.0, (lambda_max - n) / (n - 1))               # Consistency Index
    ri = 1.12                                               # Random Index for n = 5 (Saaty's table)
    cr = ci / ri                                            # Consistency Ratio
    return weights, cr


# ===================== STEP 5: NORMALISE A COLUMN TO 0-1 =====================
def normalise(column: pd.Series, lower_is_better: bool = False) -> pd.Series:
    """Scale a column to 0-1 (ignoring the top/bottom 1% outliers)."""
    s = pd.Series(pd.to_numeric(column, errors="coerce"))
    low = float(s.quantile(0.01))
    high = float(s.quantile(0.99))
    if high <= low:
        return pd.Series(0.5, index=s.index)
    clipped = s.clip(low, high)
    scaled = (clipped - low) / (high - low)
    scaled = scaled.fillna(0.5)
    if lower_is_better:
        scaled = 1.0 - scaled
    return scaled


# ===================== STEP 6: MAIN PROGRAM =====================
def main():
    # ---- 6.1 Read the input data ----
    parcels = gpd.read_file(INPUT_FILE)                     # load the GeoJSON as a table
    print("Parcels loaded:", len(parcels))                  # show how many rows we have

    # ---- 6.2 Make sure number columns are really numbers ----
    number_columns = ["parcel_area_sqm", "footprint_sqm", "estimated_floors",
                      "access_road_width_m", "predicted_price_sqft",
                      "network_dist_metro_m", "poi_count_1km", "environmental_risk_score"]
    for col in number_columns:                              # go through each column
        parcels[col] = pd.to_numeric(parcels[col], errors="coerce")   # convert to numbers

    # ---- 6.3 Add FSI columns (regulatory engine) ----
    fsi_results = [get_fsi(w, t) for w, t in zip(parcels["access_road_width_m"],
                                                 parcels["in_tod_zone"])]   # run for every parcel
    parcels[["base_fsi", "tod_bonus_fsi", "total_permissible_fsi", "fsi_zone_rule"]] = \
        pd.DataFrame(fsi_results, index=parcels.index)      # save results as 4 new columns

    # ---- 6.4 Capacity gap ----
    parcels["existing_built_sqm"] = parcels["footprint_sqm"] * parcels["estimated_floors"]   # already built
    parcels["permissible_built_sqm"] = parcels["parcel_area_sqm"] * parcels["total_permissible_fsi"]  # allowed
    parcels["capacity_gap_sqm"] = parcels["permissible_built_sqm"] - parcels["existing_built_sqm"]   # room left

    # ---- 6.5 Five sub-scores (each between 0 and 1) ----
    s1 = normalise(parcels["capacity_gap_sqm"])                                  # bigger gap = better
    s2 = normalise(parcels["predicted_price_sqft"])                              # higher price = better
    s3 = normalise(parcels["network_dist_metro_m"], lower_is_better=True)        # closer to metro = better
    s4 = normalise(parcels["poi_count_1km"])                                     # more POIs = better
    s5 = normalise(parcels["environmental_risk_score"], lower_is_better=True)    # lower risk = better

    # ---- 6.6 AHP weights and the mandatory consistency check ----
    weights, cr = get_ahp_weights(MATRIX)                   # get weights and CR
    print("AHP weights:", weights.round(3))                 # show the 5 weights
    print("Consistency Ratio:", round(cr, 4))               # show CR
    if cr >= 0.10:                                          # CR must be below 0.10
        raise ValueError("AHP matrix is inconsistent (CR >= 0.10). Fix the matrix.")

    # ---- 6.7 Final Land Opportunity Score (0-100) ----
    parcels["land_opportunity_score"] = 100 * (
        weights[0] * s1 + weights[1] * s2 + weights[2] * s3 +
        weights[3] * s4 + weights[4] * s5
    )                                                       # weighted sum of the 5 sub-scores
    parcels["land_opportunity_score"] = parcels["land_opportunity_score"].clip(0, 100)  # keep within 0-100

    # ---- 6.8 NGT buffer = hard disqualifier ----
    in_buffer = parcels["in_ngtn_buffer"].astype(str).str.lower().isin(["true", "1", "yes"])  # True/False
    parcels.loc[in_buffer, "land_opportunity_score"] = 0.0  # parcels inside the buffer get score 0
    print("Parcels set to 0 (NGT buffer):", int(in_buffer.sum()))   # how many were disqualified

    # ---- 6.9 Rank (1 = best) ----
    parcels["opportunity_rank"] = parcels["land_opportunity_score"].rank(
        ascending=False, method="min").astype(int)          # highest score gets rank 1

    # ---- 6.10 Save the result ----
    parcels = parcels.to_crs("EPSG:4326")                   # convert map coordinates to lat/lon
    parcels.to_file(OUTPUT_FILE, driver="GeoJSON")          # write the final GeoJSON
    print("Saved:", OUTPUT_FILE)                            # confirmation message


# Run main() only when this file is executed directly
if __name__ == "__main__":
    main()