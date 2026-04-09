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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def angle_to_vertical(p1: np.ndarray, p2: np.ndarray) -> float:
    vector = p2 - p1
    vertical = np.array([0.0, 1.0], dtype=np.float64)
    norm_product = np.linalg.norm(vector) * np.linalg.norm(vertical)
    if norm_product < 1e-9:
        return 90.0
    cos_theta = float(np.clip(np.dot(vector, vertical) / norm_product, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def choose_side_point(
    points_xy: np.ndarray, points_conf: np.ndarray, left_idx: int, right_idx: int
) -> np.ndarray | None:
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


def collect_images(image_input: Path) -> list[Path]:
    if image_input.is_file():
        if image_input.suffix.lower() in IMAGE_EXTENSIONS:
            return [image_input]
        raise ValueError(f"Unsupported image extension: {image_input.suffix}")

    if image_input.is_dir():
        files = sorted(
            p for p in image_input.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise RuntimeError(f"No images found in directory: {image_input}")
        return files

    raise FileNotFoundError(f"Path does not exist: {image_input}")


def predict_from_images(
    model_path: Path,
    image_input: Path,
    yolo_model_name: str,
    conf: float,
    save_visuals: bool,
    output_dir: Path,
) -> None:
    artifact = joblib.load(model_path)
    scaler = artifact["scaler"]
    kmeans = artifact["kmeans"]
    cluster_to_level = artifact["cluster_to_level"]
    max_level = max(cluster_to_level.values())

    pose_model = YOLO(yolo_model_name)
    image_paths = collect_images(image_input)

    if save_visuals:
        output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[SKIP] {img_path} -> cannot read image.")
            continue

        results = pose_model.predict(source=image, verbose=False, conf=conf, max_det=1)
        level = None
        plotted = image.copy()

        if results and len(results) > 0:
            plotted = results[0].plot()
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
            print(f"{img_path} -> no posture detection")
        else:
            text_desc = LEVEL_TEXT.get(level, f"Level {level}")
            text = f"Poziom postawy: {level}/{max_level} ({text_desc})"
            color = level_to_color(level, max_level)
            print(f"{img_path} -> level {level}/{max_level} ({text_desc})")

        if save_visuals:
            cv2.putText(plotted, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            out_name = f"{img_path.stem}_pred{img_path.suffix}"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), plotted)

    if save_visuals:
        print(f"Saved annotated images to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect posture level from test image(s).")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model") / "posture_level_model.joblib",
        help="Path to trained model artifact.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to one image file or directory with test images.",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolo11n-pose.pt",
        help="Ultralytics pose model name/path.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument(
        "--save-visuals",
        action="store_true",
        help="Save annotated output images with predictions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for annotated images when --save-visuals is used.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_from_images(
        model_path=args.model_path,
        image_input=args.input,
        yolo_model_name=args.yolo_model,
        conf=args.conf,
        save_visuals=args.save_visuals,
        output_dir=args.output_dir,
    )
