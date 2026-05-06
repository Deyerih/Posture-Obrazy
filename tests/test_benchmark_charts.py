from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_charts import create_benchmark_figure
from benchmark_charts import create_placeholder_figure
from benchmark_charts import discover_benchmark_runs
from benchmark_charts import load_benchmark_run
from benchmark_charts import save_benchmark_figure


def _sample_payload() -> dict[str, object]:
    return {
        "run_id": "20260506_220929",
        "metadata": {
            "input": "test_images",
            "engines": "yolo,mediapipe",
            "images_count": 3,
            "records_count": 6,
        },
        "summary": [
            {
                "engine": "yolo",
                "model_name": "yolo11n-pose.pt",
                "samples": 3,
                "detected_samples": 3,
                "valid_posture_samples": 3,
                "error_count": 0,
                "avg_latency_ms": 82.3,
                "median_latency_ms": 79.1,
                "detection_rate": 1.0,
                "valid_posture_rate": 1.0,
            },
            {
                "engine": "yolo",
                "model_name": "yolo11s-pose.pt",
                "samples": 3,
                "detected_samples": 3,
                "valid_posture_samples": 2,
                "error_count": 0,
                "avg_latency_ms": 141.2,
                "median_latency_ms": 132.4,
                "detection_rate": 1.0,
                "valid_posture_rate": 0.6667,
            },
        ],
        "records": [
            {
                "engine": "yolo",
                "model_name": "yolo11n-pose.pt",
                "image_path": "test_images/img-1.jpg",
                "detected_pose": 1,
                "inference_ms": 80.0,
                "posture_level": 2,
                "error": "",
            },
            {
                "engine": "yolo",
                "model_name": "yolo11n-pose.pt",
                "image_path": "test_images/img-2.jpg",
                "detected_pose": 1,
                "inference_ms": 79.0,
                "posture_level": 3,
                "error": "",
            },
            {
                "engine": "yolo",
                "model_name": "yolo11s-pose.pt",
                "image_path": "test_images/img-1.jpg",
                "detected_pose": 1,
                "inference_ms": 132.0,
                "posture_level": 4,
                "error": "",
            },
        ],
    }


def test_discover_benchmark_runs_returns_newest_first(tmp_path: Path) -> None:
    older = tmp_path / "20260506_220100_details.json"
    newer = tmp_path / "20260506_220929_details.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")

    assert discover_benchmark_runs(tmp_path) == [newer, older]


def test_create_benchmark_figure_and_save_png(tmp_path: Path) -> None:
    details_path = tmp_path / "20260506_220929_details.json"
    details_path.write_text(json.dumps(_sample_payload()), encoding="utf-8")

    run = load_benchmark_run(details_path)
    figure = create_benchmark_figure(run)

    assert len(figure.axes) == 4

    output_path = save_benchmark_figure(details_path, tmp_path / "charts.png")
    assert output_path.exists()


def test_create_benchmark_figure_handles_empty_summary(tmp_path: Path) -> None:
    details_path = tmp_path / "20260506_220930_details.json"
    details_path.write_text(
        json.dumps({"run_id": "20260506_220930", "summary": [], "records": []}),
        encoding="utf-8",
    )

    run = load_benchmark_run(details_path)
    figure = create_benchmark_figure(run)

    assert len(figure.axes) == 1


def test_create_placeholder_figure_has_one_axis() -> None:
    figure = create_placeholder_figure("Title", "Message")
    assert len(figure.axes) == 1
