#!/usr/bin/env python3
"""Pin the tutorial's sample faces by byte size and SHA-256 (maintainer tool; network required).

Writes ``src/mediapipe_face_landmarker_pipeline/sample_manifest.json``. Every sample comes from the public GitHub
repository ``debruine/webmorphR.stim`` at one immutable commit, read through ``raw.githubusercontent.com``:

* ``neutral`` - the 102 front-facing neutral photographs of the Face Research Lab London Set (``inst/neutral_front``,
  1350 x 1350 JPEG) and their 189-point WebMorph templates (``.tem``), CC BY 4.0, DeBruine & Jones (2017),
  https://doi.org/10.6084/m9.figshare.5047666.v5. Every pictured person gave signed consent for the images to be used
  in lab-based and web-based studies and to illustrate research.
* ``smiling`` - the smiling photographs of the same 102 people (``inst/smiling_front``; no templates), same licence.
* ``info`` - ``data-raw/london_info.csv``: self-reported age, gender and ethnicity per face id, same licence.
* ``composite`` - the ten composite faces (each an average of four people from the London Set) with templates
  (``inst/composite``), CC BY 4.0, DeBruine (2016), https://doi.org/10.6084/m9.figshare.4055130.v1.

Usage:  python tools/pin_samples.py
"""
# ruff: noqa: E501
from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "mediapipe_face_landmarker_pipeline" / "sample_manifest.json"
REPOSITORY = "debruine/webmorphR.stim"
COMMIT = "fa8b78fda2d659bb74ce62fcd99c4407551d2a77"
BASE_URL = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/"
COMPOSITES = [f"{sex}_{group}" for sex in ("f", "m") for group in ("african", "easian", "multi", "wasian", "white")]


def fetch(path: str) -> bytes:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(BASE_URL + path, timeout=60) as response:
                return response.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def entry(path: str, role: str, payload: bytes | None = None) -> dict:
    data = payload if payload is not None else fetch(path)
    return {"path": path, "role": role, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main() -> int:
    info_path = "data-raw/london_info.csv"
    info = fetch(info_path)
    ids = [row["face_id"] for row in csv.DictReader(io.StringIO(info.decode("utf-8")))]
    files = [entry(info_path, "info", info)]
    for face_id in ids:
        files.append(entry(f"inst/neutral_front/{face_id}_03.jpg", "neutral"))
        files.append(entry(f"inst/neutral_front/{face_id}_03.tem", "neutral_template"))
        files.append(entry(f"inst/smiling_front/{face_id}_08.jpg", "smiling"))
    for name in COMPOSITES:
        files.append(entry(f"inst/composite/{name}.jpg", "composite"))
        files.append(entry(f"inst/composite/{name}.tem", "composite_template"))
    manifest = {
        "format": "dimer_sample_manifest",
        "formatVersion": 1,
        "repository": REPOSITORY,
        "commit": COMMIT,
        "base_url": BASE_URL,
        "sets": {
            "neutral": {"title": "Face Research Lab London Set (neutral, front)", "license": "CC-BY-4.0", "doi": "10.6084/m9.figshare.5047666.v5"},
            "smiling": {"title": "Face Research Lab London Set (smiling, front)", "license": "CC-BY-4.0", "doi": "10.6084/m9.figshare.5047666.v5"},
            "info": {"title": "Face Research Lab London Set participant information", "license": "CC-BY-4.0", "doi": "10.6084/m9.figshare.5047666.v5"},
            "composite": {"title": "Young adult composite faces", "license": "CC-BY-4.0", "doi": "10.6084/m9.figshare.4055130.v1"},
        },
        "files": files,
        "totalBytes": sum(f["bytes"] for f in files),
    }
    OUT.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(files)} files, {manifest['totalBytes']:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
