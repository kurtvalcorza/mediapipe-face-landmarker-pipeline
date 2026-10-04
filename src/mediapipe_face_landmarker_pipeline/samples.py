"""Digest-pinned sample faces with annotated landmarks, and the mapping from their templates to MediaPipe indices.

Every sample is read from the public GitHub repository ``debruine/webmorphR.stim`` at one immutable commit, and every
file is checked against the byte size and SHA-256 recorded in ``sample_manifest.json`` before it is decoded.

* **neutral** - 102 front-facing neutral photographs of the Face Research Lab London Set (DeBruine & Jones, 2017;
  CC BY 4.0), 1350 x 1350 px, each with a 189-point WebMorph template (``.tem``) placed by the dataset authors. Every
  pictured person gave signed consent for the images to be used in lab-based and web-based studies and to illustrate
  research. The participants' self-reported age, gender and ethnicity come from ``london_info.csv``.
* **smiling** - smiling photographs of the same 102 people (no templates): the blendshape sanity check pairs each
  smiling photograph with its neutral one.
* **composite** - ten composite faces, each the average of four London Set faces (DeBruine, 2016; CC BY 4.0), with
  templates: the new-data images of the tutorial.

``LANDMARK_MAP`` pairs 14 template points with the MediaPipe mesh index at the same anatomical location. "Left" and
"right" in the template's point names are image left and right; MediaPipe's indices are named for the subject, so the
template's "left pupil" (image left) is the subject's right iris centre, MediaPipe index 468.
"""
# ruff: noqa: E501
from __future__ import annotations

import csv
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_PATH = Path(__file__).with_name("sample_manifest.json")
TEMPLATE_POINTS = 189

# (name, WebMorph FRL template index, MediaPipe mesh index). Template names quoted from the FRL template definition
# (webmorphR, CC BY 4.0); "left"/"right" there are image sides.
LANDMARK_MAP: tuple[tuple[str, int, int], ...] = (
    ("right_iris_centre", 0, 468),  # FRL "left pupil" (image left)
    ("left_iris_centre", 1, 473),  # FRL "right pupil"
    ("right_eye_outer_corner", 18, 33),  # FRL "outside corner of left eye"
    ("right_eye_inner_corner", 22, 133),  # FRL "inside corner of left eye"
    ("left_eye_inner_corner", 23, 362),  # FRL "inside corner of right eye"
    ("left_eye_outer_corner", 27, 263),  # FRL "outside corner of right eye"
    ("subnasale", 55, 2),  # FRL "bottom-centre of nose"
    ("mouth_right_corner", 87, 61),  # FRL "left corner of the mouth"
    ("mouth_left_corner", 93, 291),  # FRL "right corner of the mouth"
    ("upper_lip_top_centre", 90, 0),  # FRL "centre of top of upper lip"
    ("upper_lip_bottom_centre", 96, 13),  # FRL "centre of bottom of upper lip"
    ("lower_lip_top_centre", 101, 14),  # FRL "centre of top of lower lip"
    ("lower_lip_bottom_centre", 106, 17),  # FRL "centre of bottom of lower lip"
    ("chin_bottom_centre", 129, 152),  # FRL "bottom centre of the chin"
)
LANDMARK_NAMES = tuple(name for name, _, _ in LANDMARK_MAP)
TEMPLATE_INDICES = tuple(t for _, t, _ in LANDMARK_MAP)
MEDIAPIPE_INDICES = tuple(m for _, _, m in LANDMARK_MAP)
# Eye centres for the inter-ocular distance: midpoints of each eye's two corners (as the upstream model card defines IOD).
EYE_CORNERS = ((LANDMARK_NAMES.index("right_eye_outer_corner"), LANDMARK_NAMES.index("right_eye_inner_corner")), (LANDMARK_NAMES.index("left_eye_inner_corner"), LANDMARK_NAMES.index("left_eye_outer_corner")))


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("format") != "dimer_sample_manifest" or len(manifest.get("commit", "")) != 40:
        raise ValueError("sample_manifest.json must name its format and a 40-hex commit")
    paths = [f["path"] for f in manifest["files"]]
    if len(paths) != len(set(paths)):
        raise ValueError("sample_manifest.json lists a path twice")
    for entry in manifest["files"]:
        if entry["path"].startswith("/") or ".." in Path(entry["path"]).parts:
            raise ValueError(f"unsafe sample path {entry['path']!r}")
    return manifest


def manifest_digest(path: Path = MANIFEST_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fetch_one(base_url: str, entry: dict[str, Any], cache_dir: Path, retries: int = 4) -> tuple[str, bool]:
    target = cache_dir / entry["path"]
    if target.is_file():
        data = target.read_bytes()
        if len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"]:
            return entry["path"], False
        target.unlink()
    data = b""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(base_url + entry["path"], timeout=60) as response:
                data = response.read(entry["bytes"] + 1)
            break
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != entry["bytes"] or digest != entry["sha256"]:
        raise ValueError(f"sample {entry['path']}: fetched {len(data)} bytes with sha256 {digest}, pinned {entry['bytes']} / {entry['sha256']}; refusing it")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    partial.write_bytes(data)
    partial.replace(target)
    return entry["path"], True


def fetch_samples(cache_dir: Path | str, *, roles: tuple[str, ...] | None = None, workers: int = 8, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fetch (or re-verify from the cache) every pinned sample file of the given roles; refuse any mismatch."""
    manifest = manifest or load_manifest()
    cache_dir = Path(cache_dir)
    entries = [e for e in manifest["files"] if roles is None or e["role"] in roles]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda e: _fetch_one(manifest["base_url"], e, cache_dir), entries))
    return {
        "files": len(entries),
        "fetched": sum(1 for _, new in results if new),
        "reused_from_cache": sum(1 for _, new in results if not new),
        "bytes": sum(e["bytes"] for e in entries),
        "commit": manifest["commit"],
        "cache_dir": str(cache_dir),
    }


def read_template(path: Path | str) -> np.ndarray:
    """A WebMorph ``.tem`` file -> (189, 2) pixel coordinates."""
    return parse_template(Path(path).read_text(encoding="utf-8"))


def parse_template(text: str) -> np.ndarray:
    """WebMorph template text -> (189, 2) pixel coordinates. The header line is the point count."""
    lines = [line.strip() for line in text.splitlines()]
    try:
        count = int(lines[0])
        points = np.array([[float(v) for v in line.split()[:2]] for line in lines[1 : 1 + count]], dtype=np.float64)
    except (ValueError, IndexError) as exc:
        raise ValueError(f"not a WebMorph template: {exc}") from exc
    if count != TEMPLATE_POINTS or points.shape != (TEMPLATE_POINTS, 2) or not np.isfinite(points).all():
        raise ValueError(f"template must hold {TEMPLATE_POINTS} finite points, found {points.shape}")
    return points


def template_subset(points: np.ndarray) -> np.ndarray:
    """The 14 evaluated template points, in ``LANDMARK_MAP`` order."""
    return np.asarray(points, dtype=np.float64)[list(TEMPLATE_INDICES)]


def read_info(path: Path | str) -> dict[str, dict[str, Any]]:
    """``london_info.csv`` -> {face_id: {age, gender, ethnicity}}; ``NA`` ages become None."""
    out = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            age = row["face_age"]
            out[row["face_id"]] = {"age": int(age) if age.isdigit() else None, "gender": row["face_gender"], "ethnicity": row["face_eth"]}
    return out


def age_band(age: int | None) -> str:
    if age is None:
        return "not reported"
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    return "35+"


def sample_records(cache_dir: Path | str, manifest: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    """The fetched samples as records: ``neutral`` and ``composite`` carry template paths; ``smiling`` pairs by face id."""
    manifest = manifest or load_manifest()
    cache_dir = Path(cache_dir)
    by_role: dict[str, list[dict[str, Any]]] = {}
    for entry in manifest["files"]:
        by_role.setdefault(entry["role"], []).append(entry)
    info_entry = by_role["info"][0]
    info = read_info(cache_dir / info_entry["path"])
    records: dict[str, list[dict[str, Any]]] = {"neutral": [], "smiling": [], "composite": []}
    for role in ("neutral", "smiling", "composite"):
        for entry in by_role.get(role, []):
            stem = Path(entry["path"]).stem
            face_id = stem.split("_")[0] if role != "composite" else stem
            record = {"id": f"{role}/{stem}", "face_id": face_id, "set": role, "image_path": cache_dir / entry["path"], "sha256": entry["sha256"]}
            if role in ("neutral", "composite"):
                record["template_path"] = cache_dir / entry["path"].replace(".jpg", ".tem")
            if role in ("neutral", "smiling"):
                record["info"] = info[face_id]
            records[role].append(record)
    return records
