import json
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import matplotlib.pyplot as plt

# PATHS AND SETTINGS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "parcels_environmental.geojson"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "parcels_clustered.geojson"
MODEL_FILE = PROJECT_ROOT / "models" / "kmeans_model.pkl"
GRAPH_FILE = PROJECT_ROOT / "models" / "cluster_validation.png"

FEATURES = [
    "access_road_width_m",
    "network_dist_metro_m",
    "poi_count_1km",
    "footprint_sqm",
    "environmental_risk_score",
]

FINAL_K = 4

CLUSTER_NAMES = {
    0: "Prime TOD Hotspot",
    1: "Suburban Greenfield",
    2: "Constrained Infill",
    3: "Environmental Hazard Zone",
}

# STANDARDIZATION

def standardize(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert scaler.mean_ is not None and scaler.scale_ is not None
    return X_scaled, np.asarray(scaler.mean_), np.asarray(scaler.scale_)

# K-MEANS

def kmeans(X, k, max_iter=100, random_state=42):
    model = KMeans(
        n_clusters=k,
        max_iter=max_iter,
        random_state=random_state,
        n_init="auto"
    )
    labels = model.fit_predict(X)
    inertia = float(model.inertia_) if model.inertia_ is not None else 0.0
    return labels, model.cluster_centers_, inertia

# SILHOUETTE SCORE

def calculate_silhouette_score(X, labels):
    # Calculate Silhouette Score to evaluate clustering quality.
    score = silhouette_score(X, labels)
    return float(score)

# DAVIES-BOULDIN INDEX

def calculate_davies_bouldin_index(X, labels):
    # Calculate the Davies-Bouldin Index to evaluate clustering quality.
    score = davies_bouldin_score(X, labels)
    return float(score)

def main():
    print("\n=== MEMBER 3 CLUSTERING ===\n")

    # LOAD ENVIRONMENTAL GEOJSON

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Loaded parcels:", len(data["features"]))

    # EXTRACT VALID ML FEATURES

    rows = []
    valid_features = []
    missing_count = 0

    for feature in data["features"]:
        properties = feature.get("properties", {})
        values = [properties.get(name) for name in FEATURES]

        try:
            numeric_values = np.array(values, dtype=float)
        except (TypeError, ValueError):
            missing_count += 1
            continue

        if not np.all(np.isfinite(numeric_values)):
            missing_count += 1
            continue

        rows.append(numeric_values)
        valid_features.append(feature)

    if len(rows) < FINAL_K:
        raise ValueError("Not enough valid parcels to train K-Means.")

    X = np.asarray(rows, dtype=float)

    print("Valid parcels:", len(X))
    print("Excluded parcels with missing/invalid ML features:", missing_count)

    # STANDARDIZE FEATURES

    X_scaled, mean, std = standardize(X)

    print("Feature standardization complete.")

    # VALIDATE K = 2 THROUGH 8

    validation = []

    for k in range(2, 9):
        print(f"Testing k={k}...")

        labels_k, centers_k, inertia_k = kmeans(
            X_scaled,
            k,
            random_state=42,
        )

        silhouette = calculate_silhouette_score(X_scaled, labels_k)
        davies_bouldin = calculate_davies_bouldin_index(X_scaled, labels_k)

        result = {
            "k": k,
            "silhouette": silhouette,
            "davies_bouldin": davies_bouldin,
            "inertia": inertia_k,
        }

        validation.append(result)

        print(
            f"  Silhouette: {silhouette:.4f} | "
            f"Davies-Bouldin: {davies_bouldin:.4f} | "
            f"Inertia: {inertia_k:.2f}"
        )

    # SAVE THREE-PANEL VALIDATION GRAPH

    ks = [item["k"] for item in validation]
    silhouettes = [item["silhouette"] for item in validation]
    davies = [item["davies_bouldin"] for item in validation]
    inertias = [item["inertia"] for item in validation]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(ks, silhouettes, marker="o")
    axes[0].set_title("Silhouette Score")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Score (higher is better)")
    axes[0].set_xticks(ks)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ks, davies, marker="o")
    axes[1].set_title("Davies-Bouldin Index")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Index (lower is better)")
    axes[1].set_xticks(ks)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ks, inertias, marker="o")
    axes[2].set_title("Elbow Method (Inertia)")
    axes[2].set_xlabel("Number of Clusters (k)")
    axes[2].set_ylabel("Inertia (lower is better)")
    axes[2].set_xticks(ks)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("K-Means Cluster Validation (k=2 to 8)")
    fig.tight_layout()

    GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(GRAPH_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nValidation graph saved:", GRAPH_FILE)

    # TRAIN FINAL MODEL WITH K = 4

    print(f"\nTraining final K-Means model with k={FINAL_K}...")

    labels, centers, final_inertia = kmeans(
        X_scaled,
        FINAL_K,
        random_state=42,
    )

    # ASSIGN CLUSTER LABELS

    for feature, label in zip(valid_features, labels):
        cluster_id = int(label)
        feature["properties"]["cluster"] = cluster_id
        feature["properties"]["cluster_name"] = CLUSTER_NAMES[cluster_id]

    # SAVE CLUSTERED GEOJSON

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print("\nClustered GeoJSON saved:", OUTPUT_FILE)

    # SAVE MODEL INFORMATION

    model_data = {
        "algorithm": "KMeans",
        "n_clusters": FINAL_K,
        "features": FEATURES,
        "cluster_names": CLUSTER_NAMES,
        "centers": centers.tolist(),
        "scaler_mean": mean.tolist() if mean is not None else [],
        "scaler_std": std.tolist() if std is not None else [],
        "inertia": float(final_inertia),
        "random_state": 42,
    }

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_data, f)

    print("K-Means model saved:", MODEL_FILE)

    # PRINT CLUSTER COUNTS

    print("\nCluster counts:")

    for cluster_id in range(FINAL_K):
        count = int(np.sum(labels == cluster_id))
        print(
            f"Cluster {cluster_id}: {count} parcels - "
            f"{CLUSTER_NAMES[cluster_id]}"
        )

    print("Unclustered parcels:", missing_count)
    print("\n=== CLUSTERING COMPLETE ===")


if __name__ == "__main__":
    main()