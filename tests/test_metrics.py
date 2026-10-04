"""Offline tests for the NME, the baselines, the perturbation geometry and the paired blendshape statistics."""

# ruff: noqa: E501
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from mediapipe_face_landmarker_pipeline import EYE_CORNERS, metrics


def test_nme_is_mean_error_over_iod():
    truth = np.zeros((14, 2))
    truth[2], truth[3], truth[4], truth[5] = (0, 0), (10, 0), (30, 0), (40, 0)  # eye centres at x=5 and x=35
    assert metrics.inter_ocular_distance(truth, EYE_CORNERS) == 30.0
    predicted = truth + [3.0, 4.0]  # every point 5 px away
    assert metrics.nme_iod(predicted, truth, 30.0) == pytest.approx(5 / 30)
    with pytest.raises(ValueError, match="zero"):
        metrics.inter_ocular_distance(np.zeros((14, 2)), EYE_CORNERS)


def test_summary_counts_misses_against_detection_only():
    summary = metrics.summarize_nme([0.02, None, 0.04, 0.2])
    assert summary["images"] == 4 and summary["detected"] == 3 and summary["detection_rate"] == 0.75
    assert summary["nme_mean"] == pytest.approx(0.08667, abs=1e-5) and summary["failure_rate_at_0.10"] == pytest.approx(1 / 3, abs=1e-4)
    assert metrics.summarize_nme([None]) == {"images": 1, "detected": 0, "detection_rate": 0.0}
    low, high = metrics.bootstrap_ci([0.01, 0.02, 0.03, 0.04], seed=0)
    assert low <= 0.025 <= high


def test_mean_shape_baselines_leave_each_face_out():
    truths = [np.full((2, 2), float(v)) for v in (0, 3, 6)]
    loo = metrics.mean_shape_image(truths)
    assert np.allclose(loo[0], 4.5) and np.allclose(loo[2], 1.5)
    boxes = [{"x": 0, "y": 0, "width": 10, "height": 10}, {"x": 10, "y": 10, "width": 20, "height": 20}, None]
    placed = metrics.mean_shape_box(truths, boxes)
    assert placed[2] is None
    assert np.allclose(placed[0], ((3 - 10) / 20) * 10)  # face 1's box-relative shape placed in face 0's box
    with pytest.raises(ValueError):
        metrics.mean_shape_image(truths[:1])


@pytest.mark.parametrize(("kind", "value"), [("rotate", 30.0), ("rotate", -100.0), ("roll", 12.0), ("scale", 0.5), ("none", 0.0)])
def test_perturbation_maps_track_a_marked_pixel(kind, value):
    image = Image.new("RGB", (300, 200), (0, 0, 0))
    point = (220, 60)
    image.putpixel(point, (255, 255, 255))
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            image.putpixel((point[0] + dx, point[1] + dy), (255, 255, 255))
    changed, forward = metrics.perturb(image, kind, value)
    expected = metrics.apply_affine(forward, np.array([point], dtype=float))[0]
    array = np.asarray(changed.convert("L"), dtype=float)
    ys, xs = np.nonzero(array > 100)
    observed = np.array([xs.mean(), ys.mean()])
    assert np.linalg.norm(observed - expected) < 2.0, (kind, observed, expected)
    back = metrics.apply_affine(metrics.invert_affine(forward), expected[None, :])[0]
    assert np.allclose(back, point)


def test_photometric_perturbations_keep_geometry():
    image = Image.new("RGB", (100, 80), (120, 120, 120))
    for kind, value in (("blur", 3.0), ("brightness", 0.5), ("jpeg", 10)):
        changed, forward = metrics.perturb(image, kind, value)
        assert changed.size == image.size and np.allclose(forward, [[1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="unknown perturbation"):
        metrics.perturb(image, "mirror", 1)


def test_paired_sanity_and_rank_auc():
    neutral = [0.0, 0.1, 0.2, 0.3]
    smiling = [0.5, 0.6, 0.1, 0.9]
    result = metrics.paired_sanity(neutral, smiling)
    assert result["pairs"] == 4 and result["expressive_higher"] == 3 and result["fraction_expressive_higher"] == 0.75
    assert result["rank_auc"] == pytest.approx(metrics.rank_auc(smiling, neutral), abs=1e-4)
    assert metrics.rank_auc([1, 1], [1, 1]) == 0.5 and metrics.rank_auc([2], [1]) == 1.0
    assert metrics.sign_test_p(10, 10) == pytest.approx(2 / 1024) and metrics.sign_test_p(0, 0) == 1.0
    with pytest.raises(ValueError):
        metrics.paired_sanity([], [])
