#!/usr/bin/env python3
"""Pin the MediaPipe Face Landmarker task bundle (maintainer tool; network required).

Downloads ``face_landmarker.task`` from Google Cloud Storage at one immutable object generation, records its byte size,
SHA-256, the storage-reported MD5, and the name, size and SHA-256 of every member of the bundle (a stored, uncompressed
zip of three TFLite flatbuffers and one binary protobuf), and writes
``weights/<MODEL_KEY>/dimer-base-manifest.json``. Re-running it against an unchanged object rewrites the same file.

Usage:  python tools/pin_bundle.py [--generation N]
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mediapipe_face_landmarker_pipeline import pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generation", default=pipeline.MODEL_REVISION)
    args = parser.parse_args(argv)
    url = f"{pipeline.MODEL_SOURCE_URL}?generation={args.generation}"
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
        headers = dict(response.headers)
    generation = headers.get("x-goog-generation") or headers.get("X-Goog-Generation")
    if generation != str(args.generation):
        raise SystemExit(f"storage returned generation {generation}, asked for {args.generation}")
    md5 = base64.b64encode(hashlib.md5(payload).digest()).decode()  # noqa: S324 - recorded as storage metadata, not trusted
    members = []
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        for info in bundle.infolist():
            data = bundle.read(info)
            members.append(
                {
                    "path": info.filename,
                    "bytes": info.file_size,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "compression": "stored" if info.compress_type == zipfile.ZIP_STORED else str(info.compress_type),
                }
            )
    manifest = {
        "format": "dimer_model_snapshot",
        "formatVersion": 1,
        "modelAssetSpec": "1.1",
        "modelKey": pipeline.MODEL_KEY,
        "modelId": pipeline.MODEL_ID,
        "revision": str(args.generation),
        "version": pipeline.MODEL_VERSION,
        "sourceUrl": pipeline.MODEL_SOURCE_URL,
        "architectureSource": pipeline.ARCHITECTURE_SOURCE,
        "license": pipeline.MODEL_LICENSE,
        "redistributionStatus": "permitted",
        "serialization": "mediapipe-task-bundle",
        "remoteCodeRequired": False,
        "files": [{"path": pipeline.BUNDLE_NAME, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "md5Base64": md5}],
        "bundleMembers": members,
        "totalBytes": len(payload),
    }
    out = ROOT / "weights" / pipeline.MODEL_KEY / pipeline.MANIFEST_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(payload):,} bytes, sha256 {manifest['files'][0]['sha256']}, {len(members)} members")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
