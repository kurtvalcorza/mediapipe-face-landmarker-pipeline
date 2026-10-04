"""MediaPipe Face Landmarker: pinned task bundle, input validation, inference and the output contract.

The upstream model is Google's MediaPipe Face Landmarker task bundle ``face_landmarker.task`` (float16, version 1),
published on Google Cloud Storage. The bundle is a stored (uncompressed) zip of three TFLite flatbuffers - a BlazeFace
short-range face detector, the 478-point face-mesh landmark network and the 52-output blendshape network - and a
binary protobuf with the canonical-face geometry used for the facial transformation matrix. It is executed by the
``mediapipe`` Tasks API (``FaceLandmarker``) from PyPI; nothing is unpickled and no code is fetched with the model.

This module

* pins the bundle by object generation, byte size, SHA-256 and the SHA-256 of each zip member, stages it from the
  pinned generation only, and refuses on any mismatch (``stage_bundle``, ``verify_bundle``);
* validates an image before the model runs - decodable, size and pixel ceilings, colour-mode conversion reported
  (``validate_image``) - and the requested number of faces (``validate_num_faces``);
* runs the landmarker in IMAGE or VIDEO mode (``FaceLandmarkerPipeline``) and turns its result into named,
  identified rows: 478 landmarks per face (normalised x, y, z and pixel coordinates), 52 blendshape scores with names,
  and the 4 x 4 facial transformation matrix (``landmark_rows``, ``blendshape_rows``, ``matrix_records``).

``mediapipe`` is imported only inside the functions that run the model, so validation and verification work (and
refuse) without it.
"""
# ruff: noqa: E501  -- refusal messages name the rule, the limit and the corrective action in one line
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

# ---- identity (MODEL_ASSET_SPEC §4: canonical id, immutable revision, digest) ------------------------------------------

MODEL_ID = "mediapipe-models/face_landmarker/face_landmarker"
MODEL_VERSION = "float16/1"
MODEL_REVISION = "1683136941916318"
MODEL_LICENSE = "Apache-2.0"
MODEL_KEY = "mediapipe-face-landmarker-float16-v1"
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MODEL_BYTES = 3758596
ARCHITECTURE_SOURCE = "google-ai-edge/mediapipe"
BUNDLE_NAME = "face_landmarker.task"
MODEL_SOURCE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MANIFEST_NAME = "dimer-base-manifest.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
EXPECTED_MEMBERS = (
    "face_detector.tflite",
    "face_landmarks_detector.tflite",
    "geometry_pipeline_metadata_landmarks.binarypb",
    "face_blendshapes.tflite",
)
DETECTOR_MEMBER = "face_detector.tflite"

# ---- output contract ---------------------------------------------------------------------------------------------------

NUM_LANDMARKS = 478  # 468 face-mesh points + 10 iris points (468-472 subject's right iris, 473-477 subject's left iris)
BLENDSHAPE_NAMES = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
)  # fmt: skip
NUM_BLENDSHAPES = len(BLENDSHAPE_NAMES)  # 52, index 0 is the `_neutral` placeholder category
LANDMARK_FIELDS = ("image_id", "face_id", "landmark_id", "x", "y", "z", "x_px", "y_px", "z_px")
BLENDSHAPE_FIELDS = ("image_id", "face_id", "blendshape_index", "blendshape", "score")
SCORE_SEMANTICS = (
    "Blendshape scores are network outputs in [0, 1] that drive an avatar rig; they are not calibrated probabilities "
    "that an expression is present, and the pipeline applies no threshold to them."
)

# ---- operational ceilings (VAL6) ---------------------------------------------------------------------------------------

MIN_IMAGE_SIDE = 64
MAX_IMAGE_SIDE = 8192
MAX_IMAGE_PIXELS = 40_000_000
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_NUM_FACES = 10
DEFAULT_CONFIDENCE = 0.5  # upstream default for detection, presence and tracking confidence
INPUT_SCHEMA = {
    "modality": "a still image (JPEG, PNG, BMP, WebP or any other format Pillow decodes) or a sequence of frames",
    "colour": "decoded and converted to 8-bit RGB; any conversion (greyscale, palette, alpha, CMYK, 16-bit) is reported, never silent",
    "orientation": "an EXIF orientation tag is applied before inference and reported",
    "sides": f"{MIN_IMAGE_SIDE}..{MAX_IMAGE_SIDE} px on each side",
    "pixels": f"at most {MAX_IMAGE_PIXELS:,} pixels",
    "file_bytes": f"at most {MAX_FILE_BYTES:,} bytes",
    "num_faces": f"1..{MAX_NUM_FACES} faces requested; MediaPipe returns at most that many, and zero faces is a valid result",
    "validation": "structural only: validation cannot tell whether an image contains a face; the detector decides that",
}

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS * 2  # Pillow's own decompression-bomb guard, above the documented ceiling


# ---- bundle verification and staging -----------------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(text: str) -> Any:
    def no_duplicates(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError(f"manifest has duplicate keys: {sorted({k for k in keys if keys.count(k) > 1})}")
        return dict(pairs)

    return json.loads(text, object_pairs_hook=no_duplicates)


def read_manifest(weights_dir: Path | str = DEFAULT_WEIGHTS_DIR) -> dict[str, Any]:
    """The bundle manifest, checked against the pinned identity of this module."""
    path = Path(weights_dir) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: copy weights/{MODEL_KEY}/{MANIFEST_NAME} from the repository before staging")
    manifest = _strict_json(path.read_text(encoding="utf-8"))
    expected = {"modelId": MODEL_ID, "revision": MODEL_REVISION, "version": MODEL_VERSION, "modelKey": MODEL_KEY, "license": MODEL_LICENSE}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"manifest {key} = {manifest.get(key)!r} != pinned {value!r}; the manifest and the package disagree")
    files = manifest.get("files", [])
    if [f.get("path") for f in files] != [BUNDLE_NAME] or files[0].get("bytes") != MODEL_BYTES or files[0].get("sha256") != MODEL_SHA256:
        raise ValueError(f"manifest must list exactly {BUNDLE_NAME} with {MODEL_BYTES} bytes and sha256 {MODEL_SHA256}")
    members = [m.get("path") for m in manifest.get("bundleMembers", [])]
    if sorted(members) != sorted(EXPECTED_MEMBERS):
        raise ValueError(f"manifest bundleMembers {members} != expected {list(EXPECTED_MEMBERS)}")
    return manifest


def _check_member_name(name: str) -> None:
    if name.startswith(("/", "\\")) or ".." in Path(name).parts or ":" in name or "\\" in name:
        raise ValueError(f"unsafe member path in the task bundle: {name!r}")


def inspect_bundle(data: bytes) -> list[dict[str, Any]]:
    """Name, size, compression and SHA-256 of every zip member, refusing unsafe member paths. Nothing is extracted."""
    members = []
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        for info in bundle.infolist():
            _check_member_name(info.filename)
            if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"directory or symlink member in the task bundle: {info.filename!r}")
            payload = bundle.read(info)
            members.append(
                {
                    "path": info.filename,
                    "bytes": info.file_size,
                    "sha256": sha256_bytes(payload),
                    "compression": "stored" if info.compress_type == zipfile.ZIP_STORED else str(info.compress_type),
                }
            )
    return members


def verify_bundle(weights_dir: Path | str = DEFAULT_WEIGHTS_DIR) -> dict[str, Any]:
    """Byte size and SHA-256 of the bundle, then every member's name, size and SHA-256 against the manifest.

    Raises ``ValueError`` naming the first file or member that differs; nothing falls back to another source.
    """
    weights_dir = Path(weights_dir)
    manifest = read_manifest(weights_dir)
    path = weights_dir / BUNDLE_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: stage it with stage_bundle(allow_download=True)")
    data = path.read_bytes()
    if len(data) != MODEL_BYTES:
        raise ValueError(f"{BUNDLE_NAME}: size {len(data)} != manifest {MODEL_BYTES}; delete the file and stage it again")
    digest = sha256_bytes(data)
    if digest != MODEL_SHA256:
        raise ValueError(f"{BUNDLE_NAME}: sha256 {digest} != manifest {MODEL_SHA256}; delete the file and stage it again")
    members = inspect_bundle(data)
    expected = {m["path"]: m for m in manifest["bundleMembers"]}
    if sorted(m["path"] for m in members) != sorted(expected):
        raise ValueError(f"bundle members {[m['path'] for m in members]} != manifest {sorted(expected)}")
    for member in members:
        want = expected[member["path"]]
        if member["bytes"] != want["bytes"] or member["sha256"] != want["sha256"]:
            raise ValueError(f"bundle member {member['path']}: size/sha256 {member['bytes']}/{member['sha256']} != manifest {want['bytes']}/{want['sha256']}")
    return {
        "model_id": MODEL_ID,
        "version": MODEL_VERSION,
        "revision": MODEL_REVISION,
        "sha256": digest,
        "bytes": len(data),
        "members": members,
        "path": str(path),
    }


def bundle_url() -> str:
    """The download URL pinned to the immutable object generation (never the mutable object name alone)."""
    return f"{MODEL_SOURCE_URL}?generation={MODEL_REVISION}"


def stage_bundle(weights_dir: Path | str = DEFAULT_WEIGHTS_DIR, *, allow_download: bool = False, retries: int = 3) -> list[str]:
    """Fetch the bundle from the pinned generation if it is absent; return the names fetched (empty when cached).

    The download is written to a temporary file and moved into place only after its size and SHA-256 match.
    """
    weights_dir = Path(weights_dir)
    read_manifest(weights_dir)
    target = weights_dir / BUNDLE_NAME
    if target.is_file():
        return []
    if not allow_download:
        raise FileNotFoundError(f"{target} is missing and allow_download is False")
    data = b""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(bundle_url(), timeout=120) as response:
                data = response.read(MODEL_BYTES + 1)
            break
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    if len(data) != MODEL_BYTES or sha256_bytes(data) != MODEL_SHA256:
        raise ValueError(f"downloaded {BUNDLE_NAME}: {len(data)} bytes with sha256 {sha256_bytes(data)}, pinned {MODEL_BYTES} / {MODEL_SHA256}; refusing it")
    with tempfile.NamedTemporaryFile(dir=weights_dir, delete=False) as handle:
        handle.write(data)
    os.replace(handle.name, target)
    return [BUNDLE_NAME]


def bundle_member(weights_dir: Path | str, name: str) -> bytes:
    """One member of the verified bundle, read from the zip in memory (used for the bundled face detector)."""
    verify_bundle(weights_dir)
    if name not in EXPECTED_MEMBERS:
        raise ValueError(f"{name!r} is not a member of {BUNDLE_NAME}")
    with zipfile.ZipFile(Path(weights_dir) / BUNDLE_NAME) as bundle:
        return bundle.read(name)


# ---- input validation --------------------------------------------------------------------------------------------------


def validate_num_faces(num_faces: Any) -> int:
    if isinstance(num_faces, bool) or not isinstance(num_faces, int | np.integer):
        raise ValueError(f"num_faces must be an integer in 1..{MAX_NUM_FACES}, got {num_faces!r}")
    if not 1 <= int(num_faces) <= MAX_NUM_FACES:
        raise ValueError(f"num_faces must be in 1..{MAX_NUM_FACES}, got {num_faces}")
    return int(num_faces)


def validate_confidence(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be a number in 0.0..1.0, got {value!r}")
    return float(value)


def validate_image(source: str | Path | bytes | Image.Image, *, image_id: str | None = None) -> tuple[Image.Image, dict[str, Any]]:
    """Decode an image and check the input contract before any model runs.

    Returns the 8-bit RGB image and a report naming every change made to it (EXIF transpose, colour conversion).
    Raises ``ValueError`` naming the failed rule and the corrective action.
    """
    changes: list[str] = []
    if isinstance(source, Image.Image):
        image = source
        label = image_id or "in-memory image"
        size_bytes = None
    else:
        if isinstance(source, bytes | bytearray):
            data = bytes(source)
            label = image_id or "uploaded bytes"
        else:
            path = Path(source)
            label = image_id or path.name
            if not path.is_file():
                raise ValueError(f"{label}: file not found at {path}; check the path")
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError(f"{label}: file is {path.stat().st_size:,} bytes, above the {MAX_FILE_BYTES:,}-byte ceiling; re-encode or downscale it")
            data = path.read_bytes()
        size_bytes = len(data)
        if size_bytes > MAX_FILE_BYTES:
            raise ValueError(f"{label}: {size_bytes:,} bytes, above the {MAX_FILE_BYTES:,}-byte ceiling; re-encode or downscale it")
        try:
            probe = Image.open(io.BytesIO(data))
            probe.verify()
            image = Image.open(io.BytesIO(data))
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError(f"{label}: {width} x {height} = {width * height:,} pixels, above the {MAX_IMAGE_PIXELS:,}-pixel ceiling; downscale it")
            image.load()
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001 - Pillow raises many types for undecodable input
            raise ValueError(f"{label}: not a decodable image ({type(exc).__name__}: {exc}); supply a JPEG or PNG photograph") from exc
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE or max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"{label}: {width} x {height} px; each side must be within {MIN_IMAGE_SIDE}..{MAX_IMAGE_SIDE} px; resize the image")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"{label}: {width} x {height} = {width * height:,} pixels, above the {MAX_IMAGE_PIXELS:,}-pixel ceiling; downscale it")
    transposed = ImageOps.exif_transpose(image)
    if transposed.size != image.size or transposed.tobytes() != image.tobytes():
        changes.append("EXIF orientation applied")
        image = transposed
    mode = image.mode
    if mode != "RGB":
        if mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in image.info):
            changes.append(f"{mode} -> RGB (alpha channel dropped)")
        else:
            changes.append(f"{mode} -> RGB")
        image = image.convert("RGB")
    report = {
        "image_id": label,
        "width": image.size[0],
        "height": image.size[1],
        "source_mode": mode,
        "bytes": size_bytes,
        "changes": changes,
        "accepted": True,
    }
    return image, report


# ---- inference ---------------------------------------------------------------------------------------------------------


def _quiet_native_logs() -> None:
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class FaceLandmarkerPipeline:
    """The MediaPipe Tasks ``FaceLandmarker`` over the verified bundle, with blendshapes and transformation matrices on.

    ``running_mode`` is ``"IMAGE"`` (each image independently: the face detector runs on every call) or ``"VIDEO"``
    (frames in timestamp order: the landmarks of one frame seed the face region of the next, and the detector runs
    only when tracking is lost). Use the pipeline as a context manager or call ``close()``.
    """

    def __init__(
        self,
        bundle_path: Path | str,
        *,
        num_faces: int = 1,
        running_mode: str = "IMAGE",
        min_face_detection_confidence: float = DEFAULT_CONFIDENCE,
        min_face_presence_confidence: float = DEFAULT_CONFIDENCE,
        min_tracking_confidence: float = DEFAULT_CONFIDENCE,
    ) -> None:
        if running_mode not in ("IMAGE", "VIDEO"):
            raise ValueError(f"running_mode must be 'IMAGE' or 'VIDEO', got {running_mode!r}")
        self.bundle_path = Path(bundle_path)
        self.config = {
            "num_faces": validate_num_faces(num_faces),
            "running_mode": running_mode,
            "min_face_detection_confidence": validate_confidence("min_face_detection_confidence", min_face_detection_confidence),
            "min_face_presence_confidence": validate_confidence("min_face_presence_confidence", min_face_presence_confidence),
            "min_tracking_confidence": validate_confidence("min_tracking_confidence", min_tracking_confidence),
            "output_face_blendshapes": True,
            "output_facial_transformation_matrixes": True,
            "delegate": "CPU",
        }
        _quiet_native_logs()
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.bundle_path), delegate=BaseOptions.Delegate.CPU),
            running_mode=getattr(vision.RunningMode, running_mode),
            num_faces=self.config["num_faces"],
            min_face_detection_confidence=self.config["min_face_detection_confidence"],
            min_face_presence_confidence=self.config["min_face_presence_confidence"],
            min_tracking_confidence=self.config["min_tracking_confidence"],
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    @classmethod
    def from_weights(cls, weights_dir: Path | str = DEFAULT_WEIGHTS_DIR, **kwargs: Any) -> FaceLandmarkerPipeline:
        """Verify the staged bundle (size, SHA-256, members) and build the landmarker from it."""
        record = verify_bundle(weights_dir)
        return cls(record["path"], **kwargs)

    def close(self) -> None:
        if getattr(self, "_landmarker", None) is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> FaceLandmarkerPipeline:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @staticmethod
    def _mp_image(image: Image.Image):
        import mediapipe as mp

        array = np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=array)

    @staticmethod
    def _faces(result: Any) -> list[dict[str, Any]]:
        faces = []
        for face_id, landmarks in enumerate(result.face_landmarks):
            coords = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float64)
            if coords.shape != (NUM_LANDMARKS, 3):
                raise RuntimeError(f"landmarker returned {coords.shape[0]} landmarks, expected {NUM_LANDMARKS}")
            blend = result.face_blendshapes[face_id] if result.face_blendshapes else []
            names = tuple(c.category_name for c in blend)
            if names != BLENDSHAPE_NAMES:
                raise RuntimeError(f"landmarker returned blendshapes {names[:3]}..., expected the {NUM_BLENDSHAPES} pinned names")
            matrix = np.array(result.facial_transformation_matrixes[face_id], dtype=np.float64).reshape(4, 4)
            faces.append(
                {
                    "face_id": face_id,
                    "landmarks": coords,
                    "blendshapes": [float(c.score) for c in blend],
                    "transformation_matrix": matrix,
                }
            )
        return faces

    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        """IMAGE mode: faces in MediaPipe's returned order; an empty list when no face is found."""
        if self.config["running_mode"] != "IMAGE":
            raise RuntimeError("detect() needs running_mode='IMAGE'; use detect_frame() in VIDEO mode")
        return self._faces(self._landmarker.detect(self._mp_image(image)))

    def detect_frame(self, image: Image.Image, timestamp_ms: int) -> list[dict[str, Any]]:
        """VIDEO mode: one frame; timestamps must increase strictly."""
        if self.config["running_mode"] != "VIDEO":
            raise RuntimeError("detect_frame() needs running_mode='VIDEO'")
        if int(timestamp_ms) <= self._last_timestamp_ms:
            raise ValueError(f"timestamp_ms must increase strictly: {timestamp_ms} after {self._last_timestamp_ms}")
        self._last_timestamp_ms = int(timestamp_ms)
        return self._faces(self._landmarker.detect_for_video(self._mp_image(image), int(timestamp_ms)))


def detect_face_boxes(weights_dir: Path | str, image: Image.Image, *, min_detection_confidence: float = DEFAULT_CONFIDENCE) -> list[dict[str, Any]]:
    """Run only the bundle's own face detector (``face_detector.tflite``, read from the verified zip in memory).

    Returns boxes as ``{x, y, width, height, score}`` in pixels. Used by the detector-box mean-shape baseline.
    """
    _quiet_native_logs()
    from mediapipe.tasks.python import BaseOptions, vision

    detector_bytes = bundle_member(weights_dir, DETECTOR_MEMBER)
    options = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_buffer=detector_bytes, delegate=BaseOptions.Delegate.CPU),
        min_detection_confidence=validate_confidence("min_detection_confidence", min_detection_confidence),
    )
    with vision.FaceDetector.create_from_options(options) as detector:
        result = detector.detect(FaceLandmarkerPipeline._mp_image(image))
    boxes = []
    for detection in result.detections:
        box = detection.bounding_box
        boxes.append({"x": box.origin_x, "y": box.origin_y, "width": box.width, "height": box.height, "score": float(detection.categories[0].score)})
    return boxes


# ---- output contract ---------------------------------------------------------------------------------------------------


def pixel_coordinates(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalised (x, y, z) -> pixels. z uses the image width as its scale, as MediaPipe documents for z."""
    landmarks = np.asarray(landmarks, dtype=np.float64)
    return np.stack([landmarks[:, 0] * width, landmarks[:, 1] * height, landmarks[:, 2] * width], axis=1)


def landmark_rows(image_id: str, faces: Sequence[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    """One row per (face, landmark): normalised x, y, z and pixel x, y, z; ids keep every row traceable (OUT4)."""
    rows = []
    for face in faces:
        pixels = pixel_coordinates(face["landmarks"], width, height)
        for landmark_id, ((x, y, z), (xp, yp, zp)) in enumerate(zip(face["landmarks"], pixels, strict=True)):
            rows.append(
                {
                    "image_id": image_id,
                    "face_id": face["face_id"],
                    "landmark_id": landmark_id,
                    "x": round(float(x), 6),
                    "y": round(float(y), 6),
                    "z": round(float(z), 6),
                    "x_px": round(float(xp), 2),
                    "y_px": round(float(yp), 2),
                    "z_px": round(float(zp), 2),
                }
            )
    return rows


def blendshape_rows(image_id: str, faces: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for face in faces:
        for index, (name, score) in enumerate(zip(BLENDSHAPE_NAMES, face["blendshapes"], strict=True)):
            rows.append({"image_id": image_id, "face_id": face["face_id"], "blendshape_index": index, "blendshape": name, "score": round(float(score), 6)})
    return rows


def matrix_records(image_id: str, faces: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The 4 x 4 facial transformation matrix per face (canonical face model -> camera space, row-major)."""
    return [{"image_id": image_id, "face_id": f["face_id"], "matrix": np.round(f["transformation_matrix"], 6).tolist()} for f in faces]


def face_summary(faces: Sequence[dict[str, Any]], width: int, height: int, top: int = 5) -> list[dict[str, Any]]:
    """A compact, printable view of each face: pixel box of the landmarks and the strongest blendshapes."""
    out = []
    for face in faces:
        px = pixel_coordinates(face["landmarks"], width, height)
        order = np.argsort(face["blendshapes"])[::-1][:top]
        out.append(
            {
                "face_id": face["face_id"],
                "landmark_box_px": [round(float(px[:, 0].min()), 1), round(float(px[:, 1].min()), 1), round(float(px[:, 0].max()), 1), round(float(px[:, 1].max()), 1)],
                "top_blendshapes": {BLENDSHAPE_NAMES[i]: round(float(face["blendshapes"][i]), 3) for i in order},
            }
        )
    return out


def write_csv(rows: Iterable[dict[str, Any]], path: Path | str, fields: Sequence[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_json(value: Any, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON-serialisable: {type(value).__name__}")


def draw_overlay(
    image: Image.Image,
    faces: Sequence[dict[str, Any]],
    *,
    highlight: Sequence[int] = (),
    ground_truth: np.ndarray | None = None,
    max_side: int = 900,
) -> Image.Image:
    """Landmarks drawn on a copy of the image: all 478 points (white), `highlight` indices (orange), and optional
    ground-truth points (cyan crosses). The copy is downscaled so its longer side is at most `max_side`."""
    canvas = image.convert("RGB").copy()
    width, height = canvas.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        canvas = canvas.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(canvas)
    radius = max(1, round(1.5 * canvas.size[0] / 600))
    for face in faces:
        px = pixel_coordinates(face["landmarks"], width, height)[:, :2] * scale
        for x, y in px:
            draw.ellipse([x - radius / 2, y - radius / 2, x + radius / 2, y + radius / 2], fill=(255, 255, 255))
        for index in highlight:
            x, y = px[index]
            draw.ellipse([x - radius * 1.6, y - radius * 1.6, x + radius * 1.6, y + radius * 1.6], outline=(255, 140, 0), width=max(1, radius))
    if ground_truth is not None:
        arm = radius * 3
        for x, y in np.asarray(ground_truth, dtype=np.float64)[:, :2] * scale:
            draw.line([x - arm, y, x + arm, y], fill=(0, 220, 255), width=max(1, radius))
            draw.line([x, y - arm, x, y + arm], fill=(0, 220, 255), width=max(1, radius))
    return canvas


def contact_sheet(images: Sequence[Image.Image], *, columns: int = 5, tile: int = 300, labels: Sequence[str] | None = None) -> Image.Image:
    """Tiles of equal size in a grid, each optionally labelled in its top-left corner."""
    rows = max(1, (len(images) + columns - 1) // columns)
    sheet = Image.new("RGB", (tile * columns, tile * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for i, image in enumerate(images):
        thumb = ImageOps.contain(image.convert("RGB"), (tile, tile))
        x, y = tile * (i % columns), tile * (i // columns)
        sheet.paste(thumb, (x + (tile - thumb.size[0]) // 2, y + (tile - thumb.size[1]) // 2))
        if labels:
            draw.rectangle([x, y, x + 8 * len(labels[i]) + 6, y + 16], fill=(0, 0, 0))
            draw.text((x + 3, y + 2), labels[i], fill=(255, 255, 255))
    return sheet
