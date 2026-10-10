"""The isolated-environment tutorial path (NOTEBOOK_SPEC 2.2 §25.13): the kernel's `run_stage` helper and the carried
stage runner.

* The kernel-side tests execute the generated notebook's own carrier and `run_stage` code (no model library needed): the
  carried files are written and hash-verified into a run directory, and an invalid BYOD input stops the kernel with a
  RuntimeError that repeats the validator's refusal text.
* The offline stage test runs every stage in order against a stub landmarker, a stub face detector and synthetic
  sample files laid out at the pinned paths (NumPy, Pillow and matplotlib only). It proves the stage plumbing, the
  hand-offs through files and the exported fields, not the model. The real-model run is recorded in
  docs/release-verification.md.
"""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from conftest import synthetic_image

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE


def _infrastructure_sources() -> tuple[str, str]:
    notebook = build.render(ROOT, TEMPLATE, "test-revision")
    code = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]
    carrier = next(s for s in code if s.startswith("# @title Infrastructure: write and verify the carried"))
    install = next(s for s in code if s.startswith("# @title Infrastructure: install the locked runtime"))
    return carrier, install


def _kernel(tmp_path: Path) -> dict:
    """The kernel namespace after the carrier cell and the `run_stage` definition, with the current interpreter
    standing in for the isolated environment's Python."""
    carrier, install = _infrastructure_sources()
    run_root = tmp_path / "run"
    run_root.mkdir()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    namespace = {"ROOT": run_root, "WEIGHTS": tmp_path / "weights", "PYTHON": Path(sys.executable), "ENV": env}
    exec("import hashlib\nimport json\nimport subprocess\n" + carrier, namespace)  # noqa: S102 - the notebook's own cell
    definition = install[install.index("def run_stage(") : install.index("def load_record(")]
    exec(definition, namespace)  # noqa: S102
    return namespace


def test_carrier_writes_and_verifies_every_carried_file(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    run_root = kernel["ROOT"]
    for dest, source in TEMPLATE["carried"].items():
        assert (run_root / dest).read_bytes() == (ROOT / source).read_text(encoding="utf-8").encode("utf-8"), dest
    assert kernel["NOTEBOOK_SOURCE"]["revision"] == "test-revision"


@pytest.mark.parametrize(
    ("make", "refusal"),
    [
        (lambda folder: folder.joinpath("notes.jpg").write_text("not an image") and folder / "notes.jpg", "not a decodable image"),
        (lambda folder: _zip(folder / "evil.zip", {"../escape.jpg": b"x"}), "unsafe member path '../escape.jpg'"),
        (lambda folder: folder / "absent.jpg", "does not exist"),
    ],
)
def test_invalid_byod_input_raises_in_the_kernel_with_the_validator_message(tmp_path: Path, make, refusal) -> None:
    kernel = _kernel(tmp_path)
    folder = tmp_path / "in"
    folder.mkdir()
    target = make(folder)
    with pytest.raises(RuntimeError) as caught:
        kernel["run_stage"]("byod", "--byod", target, "--num-faces", 1)
    assert refusal in str(caught.value) and "Stage 'byod' failed (exit 2)" in str(caught.value)
    error = json.loads((kernel["ROOT"] / "state" / "byod.error.json").read_text(encoding="utf-8"))
    assert error["type"] == "ValueError" and refusal in error["message"]


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def test_kernel_cells_install_nothing_into_the_kernel() -> None:
    notebook = build.render(ROOT, TEMPLATE, "test-revision")
    kernel_code = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources"))
    assert not re.search(r"\bpip\b[^\n]*\binstall\b", kernel_code.replace("'pip', 'install', '--python', str(PYTHON)", ""))
    assert "sys.executable" not in kernel_code and "import mediapipe" not in kernel_code
    assert "'--require-hashes'" in kernel_code and "'--only-binary', ':all:'" in kernel_code
    assert "nvidia-smi" not in kernel_code and "cuda" not in kernel_code.lower()


# ---- offline pre-flight of the stage runner ---------------------------------------------------------------------------


def _canonical_template(side: int, seed: int) -> np.ndarray:
    """189 plausible points for a face centred in a `side` x `side` image (only the 14 mapped ones matter)."""
    rng = np.random.default_rng(seed)
    points = np.column_stack([rng.uniform(0.3, 0.7, 189), rng.uniform(0.3, 0.8, 189)]) * side
    layout = {0: (0.40, 0.45), 1: (0.60, 0.45), 18: (0.33, 0.45), 22: (0.45, 0.46), 23: (0.55, 0.46), 27: (0.67, 0.45), 55: (0.50, 0.60), 87: (0.42, 0.70), 93: (0.58, 0.70), 90: (0.50, 0.66), 96: (0.50, 0.69), 101: (0.50, 0.70), 106: (0.50, 0.74), 129: (0.50, 0.85)}
    for index, (x, y) in layout.items():
        points[index] = (np.array([x, y]) + rng.normal(0, 0.005, 2)) * side
    return points


def _synthetic_samples(cache: Path) -> None:
    """Small synthetic files at every pinned sample path (the stub fetch reuses them)."""
    from mediapipe_face_landmarker_pipeline import load_manifest

    manifest = load_manifest()
    for k, entry in enumerate(manifest["files"]):
        path = cache / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if entry["role"] == "info":
            ids = sorted({Path(e["path"]).stem.split("_")[0] for e in manifest["files"] if e["role"] == "neutral"})
            groups = ("white", "black", "east_asian", "west_asian")
            rows = [f"{face_id},{20 + i % 30 if i % 17 else 'NA'},{'female' if i % 2 else 'male'},{groups[i % 4]}" for i, face_id in enumerate(ids)]
            path.write_text("face_id,face_age,face_gender,face_eth\n" + "\n".join(rows) + "\n", encoding="utf-8")
        elif path.suffix == ".tem":
            points = _canonical_template(256, k)
            path.write_text("189\n" + "\n".join(f"{x:.2f}\t{y:.2f}" for x, y in points) + "\n", encoding="utf-8")
        else:
            synthetic_image(width=256, height=256, seed=k).save(path, quality=85)


class StubLandmarker:
    """Returns one face whose 14 evaluated points sit on a canonical layout (two faces when asked for two on a wide
    image, none on a flat image). Smiling photographs get higher smile scores; nothing here is a model."""

    def __init__(self, num_faces: int = 1, running_mode: str = "IMAGE", **_: object) -> None:
        from mediapipe_face_landmarker_pipeline import BLENDSHAPE_NAMES, MEDIAPIPE_INDICES

        self.config = {"num_faces": num_faces, "running_mode": running_mode, "delegate": "stub"}
        self.names = BLENDSHAPE_NAMES
        self.indices = MEDIAPIPE_INDICES
        self.last = -1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _face(self, face_id: int, x0: float, width: float, smile: float) -> dict:
        rng = np.random.default_rng(face_id)
        landmarks = np.column_stack([rng.uniform(0.3, 0.7, 478), rng.uniform(0.3, 0.8, 478), rng.uniform(-0.05, 0.05, 478)])
        layout = _canonical_template(1, 0)[[0, 1, 18, 22, 23, 27, 55, 87, 93, 90, 96, 101, 106, 129]]
        for k, index in enumerate(self.indices):
            landmarks[index, :2] = layout[k]
        landmarks[:, 0] = x0 + landmarks[:, 0] * width
        scores = [0.05] * len(self.names)
        scores[self.names.index("mouthSmileLeft")] = scores[self.names.index("mouthSmileRight")] = smile
        return {"face_id": face_id, "landmarks": landmarks, "blendshapes": scores, "transformation_matrix": np.eye(4)}

    def detect(self, image) -> list:
        array = np.asarray(image.convert("L"), dtype=float)
        if array.std() < 1.0:
            return []
        smile = 0.7 if array.mean() > 128 else 0.1
        if self.config["num_faces"] >= 2 and image.size[0] > 1.5 * image.size[1]:
            return [self._face(0, 0.0, 0.5, smile), self._face(1, 0.5, 0.5, smile)]
        return [self._face(0, 0.0, 1.0, smile)]

    def detect_frame(self, image, timestamp_ms: int) -> list:
        assert timestamp_ms > self.last
        self.last = timestamp_ms
        return self.detect(image)


@pytest.fixture
def stub_run(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    stages = _load("tutorial_stages")
    root = tmp_path / "run"
    weights = tmp_path / "weights"
    for dest, source in TEMPLATE["carried"].items():
        (root / dest).parent.mkdir(parents=True, exist_ok=True)
        (root / dest).write_text((ROOT / source).read_text(encoding="utf-8"), encoding="utf-8")
    _synthetic_samples(weights / stages.SAMPLE_CACHE)
    monkeypatch.setattr(stages, "make_landmarker", lambda run, **kw: StubLandmarker(**kw))
    monkeypatch.setattr(stages, "face_boxes", lambda run, image: [{"x": 40, "y": 40, "width": 180, "height": 180, "score": 0.9}])
    monkeypatch.setattr(stages, "fetch", lambda run: {"files": 327, "fetched": 0, "reused_from_cache": 327})
    monkeypatch.setenv("MPLBACKEND", "Agg")

    def run(stage: str, **options):
        defaults = {"byod": "", "num_faces": 1, "rotation": 120.0, "keep_stderr": True}
        namespace = argparse.Namespace(root=root, weights=weights, stage=stage, **{**defaults, **options})
        stages.STAGES[stage](stages.Run(root, weights, namespace))

    return stages, root, run


def test_every_stage_runs_in_order_against_a_stub(stub_run, tmp_path):
    stages, root, run = stub_run
    for stage in ("prepare", "inference", "evaluate", "robustness", "blendshapes", "newdata"):
        run(stage)
    (root / "outputs" / "weights.json").write_text(json.dumps({"members": []}), encoding="utf-8")  # the weights stage needs the network
    run("export")
    out = root / "outputs"
    result = json.loads((out / f"{stages.STEM}_result.json").read_text(encoding="utf-8"))
    assert result["model"]["revision"] == "1683136941916318" and result["model"]["remote_code_executed"] is False
    assert result["evaluation"]["faces"] == 102 and set(result["evaluation"]["summary"]) == {"model", "baseline_mean_shape_image", "baseline_mean_shape_detector_box"}
    assert result["blendshape_sanity"]["hypothesis"]["pairs"] == 102 and "not calibrated" in result["blendshape_sanity"]["score_semantics"]
    assert {row["perturbation"] for row in result["robustness"]} == {"none", "rotate", "scale", "blur", "brightness", "jpeg"}
    assert set(result["new_data"]["video_mode"]) == {"IMAGE", "VIDEO"}
    inference = json.loads((out / "inference.json").read_text())
    assert inference["num_faces_demo"] == {"num_faces=1": 1, "num_faces=2": 2, "blank grey image": 0}
    header = (out / f"{stages.STEM}_newdata_landmarks.csv").read_text().splitlines()[0]
    assert header == "image_id,face_id,landmark_id,x,y,z,x_px,y_px,z_px"
    assert len((out / f"{stages.STEM}_newdata_landmarks.csv").read_text().splitlines()) == 1 + 10 * 478
    assert (out / f"{stages.STEM}_newdata_blendshapes.csv").read_text().splitlines()[0] == "image_id,face_id,blendshape_index,blendshape,score"
    for name in ("_ced.png", "_evaluation_examples.jpg", "_robustness_examples.jpg", "_smile_pairs.png", "_newdata_overlays.jpg", "_walkthrough_overlay.jpg", "_two_faces_overlay.jpg"):
        assert (out / f"{stages.STEM}{name}").stat().st_size > 1000, name
    per_image = (out / f"{stages.STEM}_evaluation_per_image.csv").read_text().splitlines()
    assert len(per_image) == 103 and per_image[0].startswith("id,detected,nme_model,nme_baseline_image,nme_baseline_box")
    prepare = json.loads((out / "prepare.json").read_text())
    assert all(v.startswith("rejected") for k, v in prepare["probes"].items() if not k.startswith("greyscale"))
    # the optional branches
    run("activity", rotation=45.0)
    assert json.loads((out / "activity.json").read_text())["row"]["perturbation"] == "rotate"
    image = tmp_path / "mine.png"
    synthetic_image(width=500, height=300, seed=3).save(image)
    archive = _zip(tmp_path / "mine.zip", {"a/one.png": image.read_bytes(), "two.png": image.read_bytes(), "__MACOSX/._two.png": b"x"})
    run("byod", byod=str(archive), num_faces=2)
    record = json.loads((out / "byod" / "byod_result.json").read_text())
    assert record["data_left_runtime"] is False and record["faces_per_image"] == {"a/one.png": 2, "two.png": 2}
    assert (out / "byod" / "byod_landmarks.csv").exists() and (out / "byod" / "byod_overlays.jpg").exists()
    with pytest.raises(ValueError, match="num_faces must be in 1..10"):
        run("byod", byod=str(image), num_faces=11)


def test_stage_refuses_to_run_out_of_order(stub_run):
    _stages, _root, run = stub_run
    with pytest.raises(RuntimeError, match="data.json is missing: run the stage that writes it"):
        run("evaluate")
