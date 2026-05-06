from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_fake_ultralytics() -> None:
    if "ultralytics" in sys.modules:
        return
    fake_module = types.ModuleType("ultralytics")

    class _FakeYOLO:
        def __init__(self, *_args, **_kwargs):
            pass

    fake_module.YOLO = _FakeYOLO
    sys.modules["ultralytics"] = fake_module


_install_fake_ultralytics()
import benchmark_pose_models as benchmark  # noqa: E402


def test_mediapipe_model_variant_maps_complexity_levels() -> None:
    assert benchmark.mediapipe_model_variant(0) == "lite"
    assert benchmark.mediapipe_model_variant(1) == "full"
    assert benchmark.mediapipe_model_variant(2) == "heavy"
    assert benchmark.mediapipe_model_variant(99) == "heavy"


def test_resolve_mediapipe_model_path_switches_variant_for_default_name(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "models" / "pose_landmarker_full.task"

    assert benchmark.resolve_mediapipe_model_path(base_path, 0) == (
        tmp_path / "models" / "pose_landmarker_lite.task"
    )
    assert benchmark.resolve_mediapipe_model_path(base_path, 1) == base_path
    assert benchmark.resolve_mediapipe_model_path(base_path, 2) == (
        tmp_path / "models" / "pose_landmarker_heavy.task"
    )


def test_build_runners_creates_one_mediapipe_runner_per_complexity(
    monkeypatch, tmp_path: Path
) -> None:
    created_complexities: list[int] = []
    monkeypatch.setenv("POSTURE_ALLOW_UNSTABLE_MEDIAPIPE", "1")

    class _FakeMediaPipeRunner:
        def __init__(
            self,
            model_path: Path,
            complexity: int,
            min_detection_confidence: float,
            min_tracking_confidence: float,
        ) -> None:
            self.model_path = model_path
            self.complexity = complexity
            self.engine = "mediapipe"
            self.model_name = f"fake-{complexity}"
            created_complexities.append(complexity)

    monkeypatch.setattr(benchmark, "MediaPipePoseRunner", _FakeMediaPipeRunner)

    args = argparse.Namespace(
        engines="mediapipe",
        yolo_models="",
        yolo_conf=0.25,
        torchvision_models="",
        torchvision_score_threshold=0.75,
        mediapipe_complexities="0,2",
        mediapipe_min_detection_conf=0.5,
        mediapipe_min_tracking_conf=0.5,
        mediapipe_model_path=tmp_path / "models" / "pose_landmarker_full.task",
    )

    runners = benchmark.build_runners(args)

    assert len(runners) == 2
    assert created_complexities == [0, 2]


def test_get_mediapipe_runtime_error_blocks_unstable_macos(monkeypatch) -> None:
    monkeypatch.setattr(benchmark.platform, "system", lambda: "Darwin")
    monkeypatch.delenv("POSTURE_ALLOW_UNSTABLE_MEDIAPIPE", raising=False)

    error = benchmark.get_mediapipe_runtime_error()
    assert error is not None
    assert "YOLO-only comparison" in error


def test_get_mediapipe_runtime_error_allows_override(monkeypatch) -> None:
    monkeypatch.setattr(benchmark.platform, "system", lambda: "Darwin")
    monkeypatch.setenv("POSTURE_ALLOW_UNSTABLE_MEDIAPIPE", "1")

    assert benchmark.get_mediapipe_runtime_error() is None


def test_build_runners_skips_mediapipe_when_yolo_is_available_on_unstable_macos(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(benchmark.platform, "system", lambda: "Darwin")
    monkeypatch.delenv("POSTURE_ALLOW_UNSTABLE_MEDIAPIPE", raising=False)

    class _FakeYOLO:
        def __init__(self, model_name: str, conf: float) -> None:
            self.engine = "yolo"
            self.model_name = model_name
            self.conf = conf

    monkeypatch.setattr(benchmark, "YOLOPoseRunner", _FakeYOLO)

    args = argparse.Namespace(
        engines="yolo,mediapipe",
        yolo_models="yolo11n-pose.pt,yolo11s-pose.pt",
        yolo_conf=0.25,
        torchvision_models="",
        torchvision_score_threshold=0.75,
        mediapipe_complexities="0,1,2",
        mediapipe_min_detection_conf=0.5,
        mediapipe_min_tracking_conf=0.5,
        mediapipe_model_path=Path("models") / "pose_landmarker_full.task",
    )

    runners = benchmark.build_runners(args)

    assert [runner.model_name for runner in runners] == [
        "yolo11n-pose.pt",
        "yolo11s-pose.pt",
    ]
    captured = capsys.readouterr()
    assert "Skipping MediaPipe" in captured.out


def test_build_runners_creates_torchvision_runner(monkeypatch) -> None:
    created_models: list[tuple[str, float]] = []

    class _FakeTorchvisionRunner:
        def __init__(self, model_name: str, score_threshold: float) -> None:
            self.engine = "torchvision"
            self.model_name = model_name
            created_models.append((model_name, score_threshold))

    monkeypatch.setattr(benchmark, "TorchvisionPoseRunner", _FakeTorchvisionRunner)

    args = argparse.Namespace(
        engines="torchvision",
        yolo_models="",
        yolo_conf=0.25,
        torchvision_models="keypointrcnn_resnet50_fpn",
        torchvision_score_threshold=0.8,
        mediapipe_complexities="",
        mediapipe_min_detection_conf=0.5,
        mediapipe_min_tracking_conf=0.5,
        mediapipe_model_path=Path("models") / "pose_landmarker_full.task",
    )

    runners = benchmark.build_runners(args)

    assert len(runners) == 1
    assert created_models == [("keypointrcnn_resnet50_fpn", 0.8)]
