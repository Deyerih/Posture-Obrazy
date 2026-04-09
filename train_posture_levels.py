from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def angle_to_vertical(p1: np.ndarray, p2: np.ndarray) -> float:
    vector = p2 - p1
    vertical = np.array([0.0, 1.0], dtype=np.float64)
    norm_product = np.linalg.norm(vector) * np.linalg.norm(vertical)
    if norm_product < 1e-9:
        return 90.0
    cos_theta = float(np.clip(np.dot(vector, vertical) / norm_product, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def read_keypoints_from_label(label_path: Path) -> np.ndarray | None:
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) < 17:
        return None
    values = np.array([float(x) for x in parts], dtype=np.float64)
    coords = values[5:]
    if coords.size < 12:
        return None
    points = coords.reshape(-1, 3)[:, :2]
    return points


def build_feature_vector(points: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3 = points[:4]
    fwd_head = abs(p0[0] - p1[0])
    upper_body_offset = abs(p1[0] - p2[0])
    lower_body_offset = abs(p2[0] - p3[0])
    head_angle = angle_to_vertical(p1, p0)
    torso_angle = angle_to_vertical(p2, p1)
    hip_angle = angle_to_vertical(p3, p2)
    return np.array(
        [fwd_head, upper_body_offset, lower_body_offset, head_angle, torso_angle, hip_angle],
        dtype=np.float64,
    )


def quality_score(feature_vec: np.ndarray) -> float:
    fwd_head, upper_offset, lower_offset, head_angle, torso_angle, hip_angle = feature_vec
    return (
        2.5 * fwd_head
        + 2.0 * upper_offset
        + 1.0 * lower_offset
        + 0.02 * head_angle
        + 0.05 * torso_angle
        + 0.02 * hip_angle
    )


def gather_features(labels_dir: Path) -> np.ndarray:
    all_features: list[np.ndarray] = []
    for txt_path in sorted(labels_dir.rglob("*.txt")):
        points = read_keypoints_from_label(txt_path)
        if points is None or points.shape[0] < 4:
            continue
        all_features.append(build_feature_vector(points))
    if not all_features:
        raise RuntimeError(f"No valid labels found in {labels_dir}")
    return np.vstack(all_features)


def train_model(dataset_dir: Path, output_dir: Path, levels: int) -> None:
    labels_dir = dataset_dir / "labels"
    features = gather_features(labels_dir)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=levels, random_state=42, n_init=20)
    kmeans.fit(features_scaled)

    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_quality = {i: quality_score(centers_original[i]) for i in range(levels)}
    sorted_clusters = sorted(cluster_quality, key=cluster_quality.get)
    cluster_to_level = {cluster_id: level for level, cluster_id in enumerate(sorted_clusters, start=1)}

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "posture_level_model.joblib"
    metadata_path = output_dir / "posture_level_metadata.json"

    joblib.dump(
        {
            "scaler": scaler,
            "kmeans": kmeans,
            "cluster_to_level": cluster_to_level,
        },
        artifact_path,
    )

    metadata = {
        "levels": levels,
        "num_samples": int(features.shape[0]),
        "feature_names": [
            "fwd_head",
            "upper_body_offset",
            "lower_body_offset",
            "head_angle",
            "torso_angle",
            "hip_angle",
        ],
        "cluster_quality": cluster_quality,
        "cluster_to_level": cluster_to_level,
        "model_file": artifact_path.name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved: {artifact_path}")
    print(f"Saved: {metadata_path}")
    print(f"Trained on {features.shape[0]} samples with {levels} posture levels.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train posture level model from label keypoints.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("obrazy"),
        help="Folder containing labels/train and labels/val.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model"),
        help="Directory to save trained model artifacts.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=7,
        choices=[5, 6, 7],
        help="Number of posture levels.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(dataset_dir=args.dataset_dir, output_dir=args.output_dir, levels=args.levels)
