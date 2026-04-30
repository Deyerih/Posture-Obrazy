from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import joblib
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


from predict_posture_images import collect_images
from predict_posture_images import feature_vector_from_keypoints


@dataclass
class BenchmarkRecord:
    engine: str
    model_name: str
    image_path: str
    detected_pose: int
    inference_ms: float
    posture_level: int | None
    error: str


@dataclass
class ModelSummary:
    engine: str
    model_name: str
    samples: int
    detected_samples: int
    valid_posture_samples: int
    error_count: int
    avg_latency_ms: float
    median_latency_ms: float
    detection_rate: float
    valid_posture_rate: float


class YOLOPoseRunner:
    def __init__(self, model_name: str, conf: float) -> None:
        self.engine = "yolo"
        self.model_name = model_name
        self.conf = conf
        self.model = YOLO(model_name)

    def infer(self, image_bgr: np.ndarray) -> tuple[np.ndarray | None, float]:
        start = perf_counter()
        results = self.model.predict(
            source=image_bgr, verbose=False, conf=self.conf, max_det=1
        )
        elapsed_ms = (perf_counter() - start) * 1000.0

        if not results or len(results) == 0:
            return None, elapsed_ms
        kp = results[0].keypoints
        if kp is None or kp.xy is None or kp.conf is None or kp.xy.shape[0] == 0:
            return None, elapsed_ms

        points_xy = kp.xy[0].cpu().numpy()
        points_conf = kp.conf[0].cpu().numpy()
        feature_vec = feature_vector_from_keypoints(points_xy, points_conf)
        return feature_vec, elapsed_ms

    def close(self) -> None:
        return None


class MediaPipePoseRunner:
    EAR_LEFT = 7
    EAR_RIGHT = 8
    SHOULDER_LEFT = 11
    SHOULDER_RIGHT = 12
    HIP_LEFT = 23
    HIP_RIGHT = 24
    KNEE_LEFT = 25
    KNEE_RIGHT = 26

    def __init__(
        self,
        model_path: Path,
        complexity: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        self.engine = "mediapipe"
        self.model_name = f"blazepose-c{complexity}"
        self._min_vis = min(min_tracking_confidence, 0.5)

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        if complexity <= 0:
            preset = mp_vision.PoseLandmarkerOptions.Preset.LITE
        elif complexity == 1:
            preset = mp_vision.PoseLandmarkerOptions.Preset.FULL
        else:
            preset = mp_vision.PoseLandmarkerOptions.Preset.HEAVY

        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
            model_complexity=preset,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def _choose_side_point(
        self,
        points_xy: np.ndarray,
        points_vis: np.ndarray,
        left_idx: int,
        right_idx: int,
    ) -> np.ndarray | None:
        left_vis = float(points_vis[left_idx])
        right_vis = float(points_vis[right_idx])
        chosen_idx = left_idx if left_vis >= right_vis else right_idx
        if float(points_vis[chosen_idx]) < self._min_vis:
            return None
        return points_xy[chosen_idx].astype(np.float64)

    def infer(self, image_bgr: np.ndarray) -> tuple[np.ndarray | None, float]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp_vision.MPImage(
            image_format=mp_vision.ImageFormat.SRGB, data=image_rgb
        )

        start = perf_counter()
        result = self._landmarker.detect(mp_image)
        elapsed_ms = (perf_counter() - start) * 1000.0

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None, elapsed_ms

        landmarks = result.pose_landmarks[0]
        points_xy = np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float64)
        points_vis = np.array(
            [
                (
                    getattr(lm, "visibility", 1.0)
                    if getattr(lm, "visibility", None) is not None
                    else 1.0
                )
                for lm in landmarks
            ],
            dtype=np.float64,
        )

        ear = self._choose_side_point(
            points_xy, points_vis, self.EAR_LEFT, self.EAR_RIGHT
        )
        shoulder = self._choose_side_point(
            points_xy, points_vis, self.SHOULDER_LEFT, self.SHOULDER_RIGHT
        )
        hip = self._choose_side_point(
            points_xy, points_vis, self.HIP_LEFT, self.HIP_RIGHT
        )
        knee = self._choose_side_point(
            points_xy, points_vis, self.KNEE_LEFT, self.KNEE_RIGHT
        )
        if any(point is None for point in [ear, shoulder, hip, knee]):
            return None, elapsed_ms

        ear = np.asarray(ear)
        shoulder = np.asarray(shoulder)
        hip = np.asarray(hip)
        knee = np.asarray(knee)

        feature_vec = np.array(
            [
                abs(ear[0] - shoulder[0]),
                abs(shoulder[0] - hip[0]),
                abs(hip[0] - knee[0]),
                self._angle_to_vertical(shoulder, ear),
                self._angle_to_vertical(hip, shoulder),
                self._angle_to_vertical(knee, hip),
            ],
            dtype=np.float64,
        )
        return feature_vec, elapsed_ms

    @staticmethod
    def _angle_to_vertical(p1: np.ndarray, p2: np.ndarray) -> float:
        vector = p2 - p1
        vertical = np.array([0.0, 1.0], dtype=np.float64)
        norm_product = np.linalg.norm(vector) * np.linalg.norm(vertical)
        if norm_product < 1e-9:
            return 90.0
        cos_theta = float(np.clip(np.dot(vector, vertical) / norm_product, -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_theta)))

    def close(self) -> None:
        return None


def build_runners(
    args: argparse.Namespace,
) -> list[YOLOPoseRunner | MediaPipePoseRunner]:
    runners: list[YOLOPoseRunner | MediaPipePoseRunner] = []
    enabled_engines = {
        engine.strip().lower() for engine in args.engines.split(",") if engine.strip()
    }

    if "yolo" in enabled_engines:
        for model_name in [
            model.strip() for model in args.yolo_models.split(",") if model.strip()
        ]:
            runners.append(YOLOPoseRunner(model_name=model_name, conf=args.yolo_conf))

    if "mediapipe" in enabled_engines:
        for complexity_str in [
            value.strip()
            for value in args.mediapipe_complexities.split(",")
            if value.strip()
        ]:
            runners.append(
                MediaPipePoseRunner(
                    complexity=int(complexity_str),
                    min_detection_confidence=args.mediapipe_min_detection_conf,
                    min_tracking_confidence=args.mediapipe_min_tracking_conf,
                )
            )
            runners.append(
                MediaPipePoseRunner(
                    model_path=args.mediapipe_model_path,
                    complexity=int(complexity_str),
                    min_detection_confidence=args.mediapipe_min_detection_conf,
                    min_tracking_confidence=args.mediapipe_min_tracking_conf,
                )
            )

    if not runners:
        raise RuntimeError(
            "No model runners configured. Check --engines and model lists."
        )
    return runners


def classify_level(
    feature_vec: np.ndarray,
    scaler: object,
    kmeans: object,
    cluster_to_level: dict[int, int],
) -> int:
    feature_scaled = scaler.transform(feature_vec.reshape(1, -1))
    cluster = int(kmeans.predict(feature_scaled)[0])
    return int(cluster_to_level[cluster])


def run_warmup(
    runner: YOLOPoseRunner | MediaPipePoseRunner,
    warmup_image: np.ndarray,
    warmup_runs: int,
) -> None:
    for _ in range(max(0, warmup_runs)):
        runner.infer(warmup_image)


def summarize_records(records: list[BenchmarkRecord]) -> list[ModelSummary]:
    grouped: dict[tuple[str, str], list[BenchmarkRecord]] = {}
    for record in records:
        grouped.setdefault((record.engine, record.model_name), []).append(record)

    summaries: list[ModelSummary] = []
    for (engine, model_name), rows in sorted(grouped.items()):
        latencies = [row.inference_ms for row in rows]
        detected = sum(row.detected_pose for row in rows)
        valid_levels = sum(1 for row in rows if row.posture_level is not None)
        errors = sum(1 for row in rows if row.error)
        samples = len(rows)
        summaries.append(
            ModelSummary(
                engine=engine,
                model_name=model_name,
                samples=samples,
                detected_samples=detected,
                valid_posture_samples=valid_levels,
                error_count=errors,
                avg_latency_ms=round(float(sum(latencies) / samples), 3),
                median_latency_ms=round(float(median(latencies)), 3),
                detection_rate=round(float(detected / samples), 4),
                valid_posture_rate=round(float(valid_levels / samples), 4),
            )
        )
    return summaries


def save_outputs(
    output_dir: Path,
    run_id: str,
    records: list[BenchmarkRecord],
    summaries: list[ModelSummary],
    metadata: dict[str, object],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{run_id}_summary.csv"
    details_path = output_dir / f"{run_id}_details.json"

    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "engine",
                "model_name",
                "samples",
                "detected_samples",
                "valid_posture_samples",
                "error_count",
                "avg_latency_ms",
                "median_latency_ms",
                "detection_rate",
                "valid_posture_rate",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))

    details_payload = {
        "run_id": run_id,
        "metadata": metadata,
        "summary": [asdict(summary) for summary in summaries],
        "records": [asdict(record) for record in records],
    }
    details_path.write_text(json.dumps(details_payload, indent=2), encoding="utf-8")
    return summary_path, details_path


def benchmark(args: argparse.Namespace) -> None:
    artifact = joblib.load(args.model_path)
    scaler = artifact["scaler"]
    kmeans = artifact["kmeans"]
    cluster_to_level = artifact["cluster_to_level"]

    image_paths = collect_images(args.input)
    warmup_image = cv2.imread(str(image_paths[0]))
    if warmup_image is None:
        raise RuntimeError(f"Cannot read warmup image: {image_paths[0]}")

    runners = build_runners(args)
    records: list[BenchmarkRecord] = []
    try:
        for runner in runners:
            run_warmup(runner, warmup_image=warmup_image, warmup_runs=args.warmup_runs)
            for image_path in image_paths:
                image = cv2.imread(str(image_path))
                if image is None:
                    records.append(
                        BenchmarkRecord(
                            engine=runner.engine,
                            model_name=runner.model_name,
                            image_path=str(image_path),
                            detected_pose=0,
                            inference_ms=0.0,
                            posture_level=None,
                            error="cannot read image",
                        )
                    )
                    continue

                try:
                    feature_vec, inference_ms = runner.infer(image)
                    posture_level = None
                    detected_pose = 0
                    if feature_vec is not None:
                        posture_level = classify_level(
                            feature_vec, scaler, kmeans, cluster_to_level
                        )
                        detected_pose = 1
                    records.append(
                        BenchmarkRecord(
                            engine=runner.engine,
                            model_name=runner.model_name,
                            image_path=str(image_path),
                            detected_pose=detected_pose,
                            inference_ms=round(float(inference_ms), 3),
                            posture_level=posture_level,
                            error="",
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    records.append(
                        BenchmarkRecord(
                            engine=runner.engine,
                            model_name=runner.model_name,
                            image_path=str(image_path),
                            detected_pose=0,
                            inference_ms=0.0,
                            posture_level=None,
                            error=str(exc),
                        )
                    )
    finally:
        for runner in runners:
            runner.close()

    if not records:
        raise RuntimeError("No benchmark records created.")

    summaries = summarize_records(records)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {
        "input": str(args.input),
        "model_path": str(args.model_path),
        "engines": args.engines,
        "yolo_models": args.yolo_models,
        "mediapipe_complexities": args.mediapipe_complexities,
        "warmup_runs": args.warmup_runs,
        "records_count": len(records),
        "images_count": len(image_paths),
    }
    summary_path, details_path = save_outputs(
        output_dir=args.log_dir,
        run_id=run_id,
        records=records,
        summaries=summaries,
        metadata=metadata,
    )
    print(f"Saved summary: {summary_path}")
    print(f"Saved details: {details_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark multiple pose model engines and log results."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Path to one image or a directory."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model") / "posture_level_model.joblib",
        help="Path to trained posture level artifact.",
    )
    parser.add_argument(
        "--engines",
        type=str,
        default="yolo,mediapipe",
        help="Comma-separated engines to run. Supported: yolo, mediapipe.",
    )
    parser.add_argument(
        "--yolo-models",
        type=str,
        default="yolo11n-pose.pt,yolo11s-pose.pt,yolo11m-pose.pt",
        help="Comma-separated YOLO pose model names or paths.",
    )
    parser.add_argument(
        "--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold."
    )
    parser.add_argument(
        "--mediapipe-complexities",
        type=str,
        default="0,1,2",
        help="Comma-separated MediaPipe model_complexity values.",
    )
    parser.add_argument(
        "--mediapipe-min-detection-conf",
        type=float,
        default=0.5,
        help="MediaPipe min_detection_confidence.",
    )
    parser.add_argument(
        "--mediapipe-min-tracking-conf",
        type=float,
        default=0.5,
        help="MediaPipe min_tracking_confidence.",
    )
    parser.add_argument(
        "--mediapipe-model-path",
        type=Path,
        default=Path("models") / "pose_landmarker_full.task",
        help="Path to MediaPipe pose_landmarker .task model file.",
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup runs per model before logging.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("benchmark_logs"),
        help="Output log directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark(args)
