"""Offline tests for bundle verification and staging, input validation and the output contract (no mediapipe needed)."""
# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import mediapipe_face_landmarker_pipeline as pkg
from conftest import synthetic_image
from mediapipe_face_landmarker_pipeline import pipeline

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "weights" / pipeline.MODEL_KEY / pipeline.MANIFEST_NAME


# ---- identity ------------------------------------------------------------------------------------------------------------


def test_committed_manifest_matches_the_pinned_identity():
    manifest = pipeline.read_manifest(MANIFEST.parent)
    assert manifest["files"][0] == {"path": "face_landmarker.task", "bytes": 3758596, "sha256": pipeline.MODEL_SHA256, "md5Base64": "sOcnSQehZEQE/vZrKN1thQ=="}
    assert [m["path"] for m in manifest["bundleMembers"]] == list(pipeline.EXPECTED_MEMBERS)
    assert all(m["compression"] == "stored" for m in manifest["bundleMembers"])
    assert manifest["remoteCodeRequired"] is False and manifest["license"] == "Apache-2.0"
    assert pipeline.bundle_url().endswith("face_landmarker.task?generation=1683136941916318")
    assert pipeline.MODEL_SOURCE_URL.startswith("https://storage.googleapis.com/mediapipe-models/")


def _fake_bundle(tmp_path: Path, monkeypatch, members: dict[str, bytes] | None = None) -> Path:
    """A stand-in bundle and manifest; the module's pinned size and digest are patched to match it."""
    members = members or {name: f"payload of {name}".encode() for name in pipeline.EXPECTED_MEMBERS}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    data = buffer.getvalue()
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / pipeline.BUNDLE_NAME).write_bytes(data)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["files"][0].update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    manifest["bundleMembers"] = [{"path": n, "bytes": len(d), "sha256": hashlib.sha256(d).hexdigest(), "compression": "stored"} for n, d in members.items() if n in pipeline.EXPECTED_MEMBERS]
    (weights / pipeline.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(pipeline, "MODEL_BYTES", len(data))
    monkeypatch.setattr(pipeline, "MODEL_SHA256", hashlib.sha256(data).hexdigest())
    return weights


def test_verify_bundle_checks_every_member(tmp_path, monkeypatch):
    weights = _fake_bundle(tmp_path, monkeypatch)
    record = pipeline.verify_bundle(weights)
    assert [m["path"] for m in record["members"]] == list(pipeline.EXPECTED_MEMBERS)
    manifest = json.loads((weights / pipeline.MANIFEST_NAME).read_text())
    manifest["bundleMembers"][1]["sha256"] = "0" * 64
    (weights / pipeline.MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="bundle member face_landmarks_detector.tflite"):
        pipeline.verify_bundle(weights)


def test_verify_bundle_refuses_a_changed_file(tmp_path, monkeypatch):
    weights = _fake_bundle(tmp_path, monkeypatch)
    path = weights / pipeline.BUNDLE_NAME
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256 .* != manifest"):
        pipeline.verify_bundle(weights)


def test_manifest_identity_drift_is_refused(tmp_path):
    weights = tmp_path / "w"
    weights.mkdir()
    manifest = json.loads(MANIFEST.read_text())
    manifest["revision"] = "1683136941468629"  # the generation of a different object (the `latest` alias)
    (weights / pipeline.MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest revision"):
        pipeline.read_manifest(weights)
    (weights / pipeline.MANIFEST_NAME).write_text('{"modelId": 1, "modelId": 2}')
    with pytest.raises(ValueError, match="duplicate keys"):
        pipeline.read_manifest(weights)


def test_unsafe_member_names_are_refused():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("../face_detector.tflite", b"x")
    with pytest.raises(ValueError, match="unsafe member path"):
        pipeline.inspect_bundle(buffer.getvalue())


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_stage_bundle_downloads_the_pinned_generation_and_refuses_other_bytes(tmp_path, monkeypatch):
    weights = _fake_bundle(tmp_path, monkeypatch)
    good = (weights / pipeline.BUNDLE_NAME).read_bytes()
    (weights / pipeline.BUNDLE_NAME).unlink()
    seen = []
    monkeypatch.setattr(pipeline.urllib.request, "urlopen", lambda url, timeout: seen.append(url) or _Response(b"tampered" + good[8:]))
    with pytest.raises(ValueError, match="refusing it"):
        pipeline.stage_bundle(weights, allow_download=True)
    assert not (weights / pipeline.BUNDLE_NAME).exists()  # nothing written on a mismatch
    assert seen == [pipeline.bundle_url()]
    monkeypatch.setattr(pipeline.urllib.request, "urlopen", lambda url, timeout: _Response(good))
    assert pipeline.stage_bundle(weights, allow_download=True) == [pipeline.BUNDLE_NAME]
    assert pipeline.stage_bundle(weights, allow_download=True) == []  # cached
    with pytest.raises(FileNotFoundError):
        (weights / pipeline.BUNDLE_NAME).unlink()
        pipeline.stage_bundle(weights, allow_download=False)


# ---- validation ----------------------------------------------------------------------------------------------------------


def test_validate_image_reports_conversions(tmp_path):
    image, report = pipeline.validate_image(synthetic_image(mode="L"), image_id="grey")
    assert image.mode == "RGB" and report["changes"] == ["L -> RGB"]
    rgba = synthetic_image().convert("RGBA")
    path = tmp_path / "alpha.png"
    rgba.save(path)
    image, report = pipeline.validate_image(path)
    assert report["changes"] == ["RGBA -> RGB (alpha channel dropped)"] and report["bytes"] == path.stat().st_size


def test_validate_image_applies_exif_orientation(tmp_path):
    image = synthetic_image(width=200, height=100)
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise on display
    path = tmp_path / "rotated.jpg"
    image.save(path, exif=exif)
    out, report = pipeline.validate_image(path)
    assert out.size == (100, 200) and "EXIF orientation applied" in report["changes"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not an image at all", "not a decodable image"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 30, "not a decodable image"),
    ],
)
def test_validate_image_refuses_undecodable_bytes(payload, message):
    with pytest.raises(ValueError, match=message):
        pipeline.validate_image(payload, image_id="upload.png")


def test_validate_image_refuses_sizes_and_missing_files(tmp_path):
    with pytest.raises(ValueError, match="64..8192"):
        pipeline.validate_image(synthetic_image(width=63, height=200))
    with pytest.raises(ValueError, match="64..8192"):
        pipeline.validate_image(Image.new("RGB", (8193, 64)))
    with pytest.raises(ValueError, match="file not found"):
        pipeline.validate_image(tmp_path / "missing.jpg")


def test_pixel_ceiling_is_enforced_before_decoding(monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_IMAGE_PIXELS", 10_000)
    buffer = io.BytesIO()
    Image.new("RGB", (200, 200)).save(buffer, format="PNG")
    with pytest.raises(ValueError, match="pixel ceiling"):
        pipeline.validate_image(buffer.getvalue())


def test_num_faces_and_confidence_contract():
    assert pipeline.validate_num_faces(10) == 10
    for bad in (0, 11, 1.5, True, "2"):
        with pytest.raises(ValueError):
            pipeline.validate_num_faces(bad)
    assert pipeline.validate_confidence("c", 0.5) == 0.5
    with pytest.raises(ValueError, match="0.0..1.0"):
        pipeline.validate_confidence("c", 1.5)


# ---- output contract -----------------------------------------------------------------------------------------------------


def _face(face_id: int = 0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "face_id": face_id,
        "landmarks": np.column_stack([rng.uniform(0.3, 0.7, 478), rng.uniform(0.3, 0.7, 478), rng.uniform(-0.1, 0.1, 478)]),
        "blendshapes": list(rng.uniform(0, 1, pipeline.NUM_BLENDSHAPES)),
        "transformation_matrix": np.eye(4),
    }


def test_rows_keep_ids_and_pixel_coordinates(tmp_path):
    faces = [_face(0, 1), _face(1, 2)]
    rows = pipeline.landmark_rows("img", faces, 200, 100)
    assert len(rows) == 2 * 478 and tuple(rows[0]) == pipeline.LANDMARK_FIELDS
    first = rows[0]
    assert first["x_px"] == pytest.approx(first["x"] * 200, abs=0.01) and first["y_px"] == pytest.approx(first["y"] * 100, abs=0.01)
    assert first["z_px"] == pytest.approx(first["z"] * 200, abs=0.01)
    assert {r["face_id"] for r in rows} == {0, 1} and rows[478]["landmark_id"] == 0
    blend = pipeline.blendshape_rows("img", faces)
    assert len(blend) == 2 * 52 and blend[0]["blendshape"] == "_neutral" and blend[44]["blendshape"] == "mouthSmileLeft"
    matrices = pipeline.matrix_records("img", faces)
    assert matrices[1]["matrix"] == np.eye(4).tolist()
    path = pipeline.write_csv(rows, tmp_path / "l.csv", pipeline.LANDMARK_FIELDS)
    with path.open() as handle:
        assert next(csv.reader(handle)) == list(pipeline.LANDMARK_FIELDS)
    summary = pipeline.face_summary(faces, 200, 100, top=3)
    assert len(summary[0]["top_blendshapes"]) == 3


def test_overlay_and_contact_sheet_render():
    image = synthetic_image(width=640, height=480)
    truth = np.array([[100.0, 100.0], [200.0, 150.0]])
    overlay = pipeline.draw_overlay(image, [_face()], highlight=(1, 33), ground_truth=truth, max_side=320)
    assert overlay.size == (320, 240) and overlay.tobytes() != image.resize((320, 240)).tobytes()
    sheet = pipeline.contact_sheet([overlay, overlay, overlay], columns=2, tile=100, labels=["a", "b", "c"])
    assert sheet.size == (200, 200)


def test_blendshape_names_are_the_pinned_52():
    assert pipeline.NUM_BLENDSHAPES == 52 and len(set(pipeline.BLENDSHAPE_NAMES)) == 52
    assert "not calibrated probabilities" in pipeline.SCORE_SEMANTICS


def test_public_surface_is_exported():
    for name in pkg.__all__:
        assert hasattr(pkg, name), name


def test_landmarker_wrapper_needs_mediapipe_only_when_built(tmp_path, monkeypatch):
    """Building the wrapper imports mediapipe; validation of its arguments happens first."""
    with pytest.raises(ValueError, match="running_mode"):
        pipeline.FaceLandmarkerPipeline(tmp_path / "x.task", running_mode="LIVE_STREAM")
    with pytest.raises(ValueError, match="num_faces"):
        pipeline.FaceLandmarkerPipeline(tmp_path / "x.task", num_faces=0)


@pytest.mark.skipif(shutil.which("true") is None, reason="posix only")
def test_default_weights_dir_points_into_the_repository():
    assert pipeline.DEFAULT_WEIGHTS_DIR == ROOT / "weights" / pipeline.MODEL_KEY
