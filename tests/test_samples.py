"""Offline tests for the sample manifest, template parsing, the landmark map and the digest-checked fetch."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import io
from collections import Counter

import numpy as np
import pytest

from mediapipe_face_landmarker_pipeline import NUM_LANDMARKS, samples


def test_manifest_pins_every_file_once():
    manifest = samples.load_manifest()
    roles = Counter(f["role"] for f in manifest["files"])
    assert roles == {"info": 1, "neutral": 102, "neutral_template": 102, "smiling": 102, "composite": 10, "composite_template": 10}
    assert manifest["commit"] == "fa8b78fda2d659bb74ce62fcd99c4407551d2a77"
    assert manifest["base_url"] == f"https://raw.githubusercontent.com/debruine/webmorphR.stim/{manifest['commit']}/"
    assert all(len(f["sha256"]) == 64 and f["bytes"] > 0 for f in manifest["files"])
    assert manifest["totalBytes"] == sum(f["bytes"] for f in manifest["files"])
    assert {s["license"] for s in manifest["sets"].values()} == {"CC-BY-4.0"}


def test_every_neutral_photograph_has_a_template_and_a_smiling_pair():
    paths = {f["path"] for f in samples.load_manifest()["files"]}
    neutral = sorted(p for p in paths if p.startswith("inst/neutral_front/") and p.endswith(".jpg"))
    assert len(neutral) == 102
    for path in neutral:
        face_id = path.rsplit("/", 1)[1].split("_")[0]
        assert path.replace(".jpg", ".tem") in paths and f"inst/smiling_front/{face_id}_08.jpg" in paths


def test_landmark_map_is_well_formed():
    assert len(samples.LANDMARK_MAP) == 14
    assert len(set(samples.TEMPLATE_INDICES)) == 14 and len(set(samples.MEDIAPIPE_INDICES)) == 14
    assert all(0 <= t < samples.TEMPLATE_POINTS for t in samples.TEMPLATE_INDICES)
    assert all(0 <= m < NUM_LANDMARKS for m in samples.MEDIAPIPE_INDICES)
    # the image-left pupil of the template is the subject's right iris centre (468); the image-right one is 473
    assert samples.LANDMARK_MAP[0] == ("right_iris_centre", 0, 468) and samples.LANDMARK_MAP[1] == ("left_iris_centre", 1, 473)


def _template_text(points: np.ndarray) -> str:
    return "189\n" + "\n".join(f"{x}\t{y}" for x, y in points) + "\n3\n0 1 2\n"


def test_parse_template_and_subset():
    points = np.arange(189 * 2, dtype=float).reshape(189, 2)
    parsed = samples.parse_template(_template_text(points))
    assert parsed.shape == (189, 2) and np.array_equal(parsed, points)
    subset = samples.template_subset(parsed)
    assert np.array_equal(subset[0], points[0]) and np.array_equal(subset[-1], points[129])
    with pytest.raises(ValueError, match="189 finite points"):
        samples.parse_template("5\n1 2\n3 4\n5 6\n7 8\n9 10\n")
    with pytest.raises(ValueError, match="not a WebMorph template"):
        samples.parse_template("hello")


def test_read_info_handles_missing_ages(tmp_path):
    path = tmp_path / "info.csv"
    path.write_text("face_id,face_age,face_gender,face_eth\n001,24,female,white\n002,NA,male,black\n", encoding="utf-8")
    info = samples.read_info(path)
    assert info["001"] == {"age": 24, "gender": "female", "ethnicity": "white"} and info["002"]["age"] is None
    assert samples.age_band(None) == "not reported" and samples.age_band(24) == "18-24" and samples.age_band(40) == "35+"


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_refuses_a_mismatched_file_and_reuses_a_verified_cache(tmp_path, monkeypatch):
    good = b"pinned bytes"
    manifest = {"base_url": "https://example.invalid/", "commit": "0" * 40, "files": [{"path": "a/b.jpg", "role": "neutral", "bytes": len(good), "sha256": hashlib.sha256(good).hexdigest()}]}
    monkeypatch.setattr(samples.urllib.request, "urlopen", lambda url, timeout: _Response(b"other bytes!"))
    with pytest.raises(ValueError, match="refusing it"):
        samples.fetch_samples(tmp_path, manifest=manifest)
    assert not (tmp_path / "a" / "b.jpg").exists()
    monkeypatch.setattr(samples.urllib.request, "urlopen", lambda url, timeout: _Response(good))
    assert samples.fetch_samples(tmp_path, manifest=manifest)["fetched"] == 1
    assert samples.fetch_samples(tmp_path, manifest=manifest)["reused_from_cache"] == 1
    (tmp_path / "a" / "b.jpg").write_bytes(b"corrupted!!!")
    assert samples.fetch_samples(tmp_path, manifest=manifest)["fetched"] == 1  # a corrupted cache entry is replaced
