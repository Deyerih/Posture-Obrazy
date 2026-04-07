from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
from ultralytics import YOLO


LEVEL_TEXT = {
    1: "Terrible",
    2: "Very bad",
    3: "Bad",
    4: "Average",
    5: "Good",
    6: "Very good",
    7: "Perfect",
}


def angle_to_vertical(p1: np.ndarray, p2: np.ndarray) -> float:
    vector = p2 - p1
    vertical = np.array([0.0, 1.0], dtype=np.float64)
    norm_product = np.linalg.norm(vector) * np.linalg.norm(vertical)
    if norm_product < 1e-9:
        return 90.0
    cos_theta = float(np.clip(np.dot(vector, vertical) / norm_product, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def choose_side_point(points_xy: np.ndarray, points_conf: np.ndarray, left_idx: int, right_idx: int) -> np.ndarray | None:
    left_conf = float(points_conf[left_idx])
    right_conf = float(points_conf[right_idx])
    chosen_idx = left_idx if left_conf >= right_conf else right_idx
    if float(points_conf[chosen_idx]) < 0.2:
        return None
    return points_xy[chosen_idx].astype(np.float64)


def feature_vector_from_keypoints(points_xy: np.ndarray, points_conf: np.ndarray) -> np.ndarray | None:
    ear = choose_side_point(points_xy, points_conf, 3, 4)
    shoulder = choose_side_point(points_xy, points_conf, 5, 6)
    hip = choose_side_point(points_xy, points_conf, 11, 12)
    knee = choose_side_point(points_xy, points_conf, 13, 14)
    if any(point is None for point in [ear, shoulder, hip, knee]):
        return None

    ear = np.asarray(ear)
    shoulder = np.asarray(shoulder)
    hip = np.asarray(hip)
    knee = np.asarray(knee)

    fwd_head = abs(ear[0] - shoulder[0])
    upper_body_offset = abs(shoulder[0] - hip[0])
    lower_body_offset = abs(hip[0] - knee[0])
    head_angle = angle_to_vertical(shoulder, ear)
    torso_angle = angle_to_vertical(hip, shoulder)
    hip_angle = angle_to_vertical(knee, hip)

    return np.array(
        [fwd_head, upper_body_offset, lower_body_offset, head_angle, torso_angle, hip_angle],
        dtype=np.float64,
    )


def level_to_color(level: int, max_level: int) -> tuple[int, int, int]:
    ratio = (level - 1) / max(1, max_level - 1)
    red = int(255 * (1.0 - ratio))
    green = int(255 * ratio)
    return (0, green, red)


def run_camera(model_path: Path, camera_id: int) -> None:
    artifact = joblib.load(model_path)
    scaler = artifact["scaler"]
    kmeans = artifact["kmeans"]
    cluster_to_level = artifact["cluster_to_level"]
    max_level = max(cluster_to_level.values())

    pose_model = YOLO("yolo11n-pose.pt")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera id {camera_id}")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        results = pose_model.predict(source=frame, verbose=False, conf=0.25, max_det=1)

        level = None
        if results and len(results) > 0:
            plotted = results[0].plot()
            frame = plotted
            kp = results[0].keypoints
            if kp is not None and kp.xy is not None and kp.conf is not None and kp.xy.shape[0] > 0:
                points_xy = kp.xy[0].cpu().numpy()
                points_conf = kp.conf[0].cpu().numpy()
                feat = feature_vector_from_keypoints(points_xy, points_conf)
                if feat is not None:
                    feat_scaled = scaler.transform(feat.reshape(1, -1))
                    cluster = int(kmeans.predict(feat_scaled)[0])
                    level = int(cluster_to_level[cluster])

        if level is None:
            text = "Poziom postawy: brak detekcji"
            color = (0, 180, 255)
        else:
            text_desc = LEVEL_TEXT.get(level, f"Level {level}")
            text = f"Poziom postawy: {level}/{max_level} ({text_desc})"
            color = level_to_color(level, max_level)

        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            "Q - wyjscie",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Posture Camera", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live posture level display from camera.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model") / "posture_level_model.joblib",
        help="Path to trained model artifact.",
    )
    parser.add_argument("--camera-id", type=int, default=0, help="Camera index for OpenCV.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_camera(model_path=args.model_path, camera_id=args.camera_id)
