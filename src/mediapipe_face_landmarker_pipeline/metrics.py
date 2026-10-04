"""Landmark and blendshape sanity metrics, mean-shape baselines and controlled perturbations (NumPy and Pillow only).

* ``nme_iod`` - mean 2D point-to-point error over the evaluated points divided by the inter-ocular distance (IOD),
  the distance between the two eye centres, each the midpoint of that eye's corners. This is the normalisation the
  upstream model card uses ("IOD MAE"); here the IOD comes from the annotated template, never from the prediction.
* ``summarize_nme`` - mean, median, standard deviation, 90th percentile, maximum and the failure rate at
  ``FAILURE_NME`` over the faces that were detected, plus the detection rate over all images.
* Two baselines that use no landmark network: the leave-one-out mean of the annotated points in image pixels
  (``mean_shape_image``), and the leave-one-out mean shape expressed in face-detector box coordinates and placed in
  each image's own detector box (``mean_shape_box``).
* ``perturb`` - rotation, downscaling, Gaussian blur, brightness and JPEG re-encoding, each with the exact 2 x 3 affine
  map from original to perturbed pixel coordinates, so a prediction on the perturbed image can be mapped back and
  compared with the prediction on the original (landmark consistency) and with the transformed annotation.
* ``paired_sanity`` - for an expression pair (neutral and smiling photographs of the same person): how often the
  smiling score is higher, the median paired difference, and the rank AUC separating the two sets.
"""
# ruff: noqa: E501
from __future__ import annotations

import io
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

FAILURE_NME = 0.10  # a face whose NME exceeds 10 % of the IOD counts as a localisation failure (a common convention)
PERTURBATION_KINDS = ("none", "rotate", "roll", "scale", "blur", "brightness", "jpeg")


def eye_centres(points: np.ndarray, eye_corners: Sequence[tuple[int, int]]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return np.stack([(points[a] + points[b]) / 2.0 for a, b in eye_corners])


def inter_ocular_distance(points: np.ndarray, eye_corners: Sequence[tuple[int, int]]) -> float:
    centres = eye_centres(points, eye_corners)
    distance = float(np.linalg.norm(centres[0] - centres[1]))
    if not distance > 0:
        raise ValueError("inter-ocular distance is zero: the annotation's eye corners coincide")
    return distance


def point_errors(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Euclidean distance per point (pixels) between (K, 2) arrays."""
    predicted = np.asarray(predicted, dtype=np.float64)[:, :2]
    truth = np.asarray(truth, dtype=np.float64)[:, :2]
    if predicted.shape != truth.shape:
        raise ValueError(f"shape mismatch: predicted {predicted.shape} vs truth {truth.shape}")
    return np.linalg.norm(predicted - truth, axis=1)


def nme_iod(predicted: np.ndarray, truth: np.ndarray, iod: float) -> float:
    return float(point_errors(predicted, truth).mean() / iod)


def summarize_nme(values: Sequence[float | None]) -> dict[str, Any]:
    """`None` marks an image where no face was detected: it counts against the detection rate, not the NME."""
    detected = np.array([v for v in values if v is not None], dtype=np.float64)
    total = len(values)
    if detected.size == 0:
        return {"images": total, "detected": 0, "detection_rate": 0.0}
    return {
        "images": total,
        "detected": int(detected.size),
        "detection_rate": round(detected.size / total, 4) if total else 0.0,
        "nme_mean": round(float(detected.mean()), 5),
        "nme_median": round(float(np.median(detected)), 5),
        "nme_std": round(float(detected.std(ddof=1)) if detected.size > 1 else 0.0, 5),
        "nme_p90": round(float(np.percentile(detected, 90)), 5),
        "nme_max": round(float(detected.max()), 5),
        "failure_rate_at_0.10": round(float((detected > FAILURE_NME).mean()), 4),
    }


def bootstrap_ci(values: Sequence[float], *, seed: int = 0, n_boot: int = 2000, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean over images (resampling images, seeded)."""
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(n_boot, data.size))].mean(axis=1)
    low, high = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return round(float(low), 5), round(float(high), 5)


# ---- baselines ---------------------------------------------------------------------------------------------------------


def mean_shape_image(truths: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Leave-one-out mean of the annotated points in image pixels: 'assume the face is where this set's faces are'."""
    stack = np.stack([np.asarray(t, dtype=np.float64) for t in truths])
    total = stack.sum(axis=0)
    n = stack.shape[0]
    if n < 2:
        raise ValueError("the leave-one-out baseline needs at least two annotated faces")
    return [(total - stack[i]) / (n - 1) for i in range(n)]


def _box_coords(points: np.ndarray, box: dict[str, float]) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64) - [box["x"], box["y"]]) / [box["width"], box["height"]]


def mean_shape_box(truths: Sequence[np.ndarray], boxes: Sequence[dict[str, float] | None]) -> list[np.ndarray | None]:
    """Leave-one-out mean shape in face-box coordinates, placed in each image's own detector box.

    Images whose detector found no face get ``None`` (a baseline miss). Only images with a box contribute to the mean.
    """
    normalised = [None if box is None else _box_coords(t, box) for t, box in zip(truths, boxes, strict=True)]
    usable = [n for n in normalised if n is not None]
    if len(usable) < 2:
        raise ValueError("the box baseline needs at least two images with a detected face")
    total = np.sum(usable, axis=0)
    out: list[np.ndarray | None] = []
    for norm, box in zip(normalised, boxes, strict=True):
        if box is None or norm is None:
            out.append(None)
            continue
        mean = (total - norm) / (len(usable) - 1)
        out.append(mean * [box["width"], box["height"]] + [box["x"], box["y"]])
    return out


# ---- perturbations -----------------------------------------------------------------------------------------------------


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)[:, :2]
    return points @ np.asarray(matrix, dtype=np.float64)[:, :2].T + np.asarray(matrix, dtype=np.float64)[:, 2]


def invert_affine(matrix: np.ndarray) -> np.ndarray:
    full = np.vstack([np.asarray(matrix, dtype=np.float64), [0.0, 0.0, 1.0]])
    return np.linalg.inv(full)[:2]


def perturb(image: Image.Image, kind: str, value: float) -> tuple[Image.Image, np.ndarray]:
    """A perturbed copy of `image` and the 2 x 3 affine map from original to perturbed pixel coordinates.

    * ``none`` - the image unchanged (the reference row of a perturbation table).
    * ``rotate`` - in-plane rotation by `value` degrees counter-clockwise about the centre; the canvas expands so no
      pixel is lost (black fill). ``roll`` - the same rotation on a canvas of unchanged size (corners are cut off),
      as in the frames of a video where the head tilts.
    * ``scale`` - downscale by factor `value` (0 < value <= 1), as when the face is further from the camera.
    * ``blur`` - Gaussian blur with radius `value` px. ``brightness`` - multiply brightness by `value`.
    * ``jpeg`` - re-encode as JPEG at quality `value`.
    """
    image = image.convert("RGB")
    width, height = image.size
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    if kind == "none":
        return image.copy(), identity
    if kind in ("rotate", "roll"):
        rotated = image.rotate(value, resample=Image.Resampling.BICUBIC, expand=kind == "rotate", fillcolor=(0, 0, 0))
        theta = math.radians(value)
        cos, sin = math.cos(theta), math.sin(theta)
        cx, cy = width / 2.0, height / 2.0
        ncx, ncy = rotated.size[0] / 2.0, rotated.size[1] / 2.0
        # Pillow rotates counter-clockwise on screen (y axis pointing down): x' = c(x-cx) + s(y-cy), y' = -s(x-cx) + c(y-cy)
        matrix = np.array([[cos, sin, ncx - cos * cx - sin * cy], [-sin, cos, ncy + sin * cx - cos * cy]])
        return rotated, matrix
    if kind == "scale":
        if not 0 < value <= 1:
            raise ValueError("scale factor must be in (0, 1]")
        size = (max(1, round(width * value)), max(1, round(height * value)))
        scaled = image.resize(size, Image.Resampling.LANCZOS)
        return scaled, np.array([[size[0] / width, 0.0, 0.0], [0.0, size[1] / height, 0.0]])
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(value))), identity
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(float(value)), identity
    if kind == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=int(value))
        return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"), identity
    raise ValueError(f"unknown perturbation {kind!r}; expected one of {PERTURBATION_KINDS}")


# ---- blendshape sanity -------------------------------------------------------------------------------------------------


def rank_auc(positives: Sequence[float], negatives: Sequence[float]) -> float:
    """P(score of a random positive > score of a random negative), ties counted as one half (Mann-Whitney AUC)."""
    pos = np.asarray(positives, dtype=np.float64)
    neg = np.asarray(negatives, dtype=np.float64)
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (pos.size * neg.size))


def sign_test_p(successes: int, trials: int) -> float:
    """Two-sided exact binomial (sign) test against p = 0.5."""
    if trials == 0:
        return 1.0
    k = min(successes, trials - successes)
    tail = sum(math.comb(trials, i) for i in range(0, k + 1)) / 2**trials
    return float(min(1.0, 2 * tail))


def paired_sanity(neutral: Sequence[float], expressive: Sequence[float]) -> dict[str, Any]:
    """Paired comparison of one score on neutral vs expressive photographs of the same people."""
    neutral_arr = np.asarray(neutral, dtype=np.float64)
    expressive_arr = np.asarray(expressive, dtype=np.float64)
    if neutral_arr.shape != expressive_arr.shape or neutral_arr.size == 0:
        raise ValueError("paired_sanity needs two equal-length, non-empty score lists")
    diff = expressive_arr - neutral_arr
    higher = int((diff > 0).sum())
    nonzero = int((diff != 0).sum())
    return {
        "pairs": int(diff.size),
        "expressive_higher": higher,
        "fraction_expressive_higher": round(higher / diff.size, 4),
        "median_neutral": round(float(np.median(neutral_arr)), 4),
        "median_expressive": round(float(np.median(expressive_arr)), 4),
        "median_paired_difference": round(float(np.median(diff)), 4),
        "rank_auc": round(rank_auc(expressive_arr, neutral_arr), 4),
        "sign_test_p": sign_test_p(higher, nonzero),
    }
