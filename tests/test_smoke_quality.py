from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_posture_levels import angle_to_vertical as train_angle_to_vertical
from train_posture_levels import build_feature_vector


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
from predict_posture_images import collect_images  # noqa: E402
from predict_posture_images import feature_vector_from_keypoints  # noqa: E402


def test_angle_to_vertical_returns_90_for_zero_vector() -> None:
    p = np.array([0.0, 0.0], dtype=np.float64)
    assert train_angle_to_vertical(p, p) == pytest.approx(90.0)


def test_angle_to_vertical_returns_0_for_vertical_up_vector() -> None:
    p1 = np.array([0.0, 0.0], dtype=np.float64)
    p2 = np.array([0.0, 1.0], dtype=np.float64)
    assert train_angle_to_vertical(p1, p2) == pytest.approx(0.0)


def test_build_feature_vector_shape_is_stable() -> None:
    points = np.array(
        [
            [0.55, 0.10],
            [0.50, 0.25],
            [0.48, 0.50],
            [0.45, 0.80],
        ],
        dtype=np.float64,
    )
    features = build_feature_vector(points)
    assert features.shape == (6,)
    assert np.all(np.isfinite(features))


def test_collect_images_raises_on_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing_dir_or_file"
    with pytest.raises(FileNotFoundError):
        collect_images(missing)


def test_collect_images_raises_on_unsupported_file_extension(tmp_path: Path) -> None:
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        collect_images(bad_file)


def test_collect_images_returns_only_supported_images_from_dir(tmp_path: Path) -> None:
    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "nested" / "b.png"
    txt = tmp_path / "nested" / "c.txt"
    img_b.parent.mkdir(parents=True, exist_ok=True)
    img_a.write_bytes(b"fake")
    img_b.write_bytes(b"fake")
    txt.write_text("skip", encoding="utf-8")

    files = collect_images(tmp_path)
    assert files == [img_a, img_b]


def test_collect_images_raises_when_directory_has_no_images(tmp_path: Path) -> None:
    (tmp_path / "readme.md").write_text("no images here", encoding="utf-8")
    with pytest.raises(RuntimeError):
        collect_images(tmp_path)


def test_feature_vector_from_keypoints_returns_none_for_low_confidence() -> None:
    points_xy = np.zeros((17, 2), dtype=np.float64)
    points_conf = np.zeros(17, dtype=np.float64)
    assert feature_vector_from_keypoints(points_xy, points_conf) is None


def test_feature_vector_from_keypoints_returns_vector_for_valid_keypoints() -> None:
    points_xy = np.zeros((17, 2), dtype=np.float64)
    points_conf = np.zeros(17, dtype=np.float64)

    points_xy[3] = [0.6, 0.1]   # left ear
    points_xy[5] = [0.5, 0.3]   # left shoulder
    points_xy[11] = [0.45, 0.55]  # left hip
    points_xy[13] = [0.42, 0.8]  # left knee
    points_conf[[3, 5, 11, 13]] = 0.95

    vector = feature_vector_from_keypoints(points_xy, points_conf)
    assert vector is not None
    assert vector.shape == (6,)
    assert np.all(np.isfinite(vector))
