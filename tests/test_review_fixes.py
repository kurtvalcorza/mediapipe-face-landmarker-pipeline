"""Regression tests for the 2026-10-05 notebook review findings (MPF-m1..m5).

They need only CI's dependencies: the generated notebook's own cell sources are executed with stand-ins (a fake
isolated interpreter, a stub landmarker), and the stage runner runs offline. None of this is model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import types
import zipfile
from pathlib import Path

import pytest

from conftest import synthetic_image
from test_tutorial_stages import TEMPLATE, _infrastructure_sources, build, stub_run  # noqa: F401,F811 - fixture reuse

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]


def _cells() -> list[dict]:
    return build.render(ROOT, TEMPLATE, "test-revision")["cells"]


def _check_cell() -> str:
    return next(c["source"] for c in _cells() if c["cell_type"] == "code" and c["source"].startswith("# @title Infrastructure: check the runtime"))


# ---- MPF-m1 -------------------------------------------------------------------------------------------------------


def test_mpf_m1_no_stale_hosted_run_statement_and_figures_are_measured_or_labelled() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "has not yet been recorded" not in text
    cells = _cells()
    opening = cells[0]["source"]
    assert "13 s" in opening and "89 s" in opening and "4 October 2026" in opening
    assert "takes the longest" not in text and "a minute or two" not in text
    prerequisites = next(c["source"] for c in cells if c["source"].startswith("## Prerequisites"))
    assert "0.52 GB" in prerequisites and "margin" in prerequisites


def test_mpf_no_leftover_placeholders_or_kernel_installs() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "{{" not in text.replace("{{type:", "") and "{MODEL_ID}" not in text and "@P:" not in text
    kernel = "\n".join(c["source"] for c in _cells() if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources"))
    assert not re.search(r"%pip|!pip|'-m', 'pip'", kernel)


# ---- MPF-m5 and the uv-template lessons ---------------------------------------------------------------------------


def _run_check_cell(namespace: dict) -> dict:
    exec(_check_cell(), namespace)  # noqa: S102 - the notebook's own cell
    return namespace


@pytest.mark.skipif(sys.platform != "linux", reason="executes the Section 1/3 kernel cells, which refuse a non-Linux x86_64 runtime and run a POSIX venv/bin/python (Linux runtimes only)")
def test_mpf_m5_section1_rerun_keeps_the_run_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    namespace = _run_check_cell({})
    first = namespace["ROOT"]
    assert first.is_dir() and first.parent == tmp_path / "outputs" / TEMPLATE["stem"]
    (first / "tutorial_stages.py").write_text("# carried", encoding="utf-8")
    _run_check_cell(namespace)  # re-running Section 1 alone
    assert namespace["ROOT"] == first, "a Section 1 re-run must not strand the later cells in a new, empty run directory"
    # NEW_RUN_DIRECTORY still gives a fresh directory on request
    source = _check_cell().replace("NEW_RUN_DIRECTORY = False", "NEW_RUN_DIRECTORY = True")
    exec(source, namespace)  # noqa: S102
    assert namespace["ROOT"] != first


@pytest.mark.skipif(sys.platform != "linux", reason="executes the Section 1/3 kernel cells, which refuse a non-Linux x86_64 runtime and run a POSIX venv/bin/python (Linux runtimes only)")
def test_mpf_m5_environment_is_keyed_on_the_lock_not_the_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    a = _run_check_cell({})
    b = _run_check_cell({})  # a second, independent session in the same runtime
    assert a["ROOT"] != b["ROOT"] and a["ENV_ROOT"] == b["ENV_ROOT"]
    assert a["ROOT"].name not in str(a["ENV_ROOT"])
    lock_sha = hashlib.sha256((ROOT / TEMPLATE["carried"][TEMPLATE["lock"]]).read_text(encoding="utf-8").encode()).hexdigest()
    basis = "\n".join((lock_sha, TEMPLATE["managed_python"], TEMPLATE["uv"]["version"]))
    assert a["ENV_ROOT"].name.endswith(hashlib.sha256(basis.encode()).hexdigest()[:16])


def _fake_ipython(monkeypatch) -> None:
    display = types.ModuleType("IPython.display")
    display.Image = display.display = lambda *a, **k: None
    package = types.ModuleType("IPython")
    package.display = display
    monkeypatch.setitem(sys.modules, "IPython", package)
    monkeypatch.setitem(sys.modules, "IPython.display", display)


@pytest.mark.skipif(sys.platform != "linux", reason="executes the Section 1/3 kernel cells, which refuse a non-Linux x86_64 runtime and run a POSIX venv/bin/python (Linux runtimes only)")
def test_mpf_m5_install_cell_reuses_a_complete_environment_without_downloading(tmp_path: Path, monkeypatch) -> None:
    _fake_ipython(monkeypatch)
    _carrier, install = _infrastructure_sources()
    env_root = tmp_path / "env"
    python = env_root / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\necho '{\"python\": \"3.12.12\"}'\n", encoding="utf-8")
    python.chmod(0o755)
    lock_sha = "a" * 64
    spec = {"lock_sha256": lock_sha, "python": TEMPLATE["managed_python"], "uv": TEMPLATE["uv"]["version"]}
    (env_root / "ready.json").write_text(json.dumps(spec), encoding="utf-8")

    def no_network(*_a, **_k):
        raise AssertionError("a matching environment must be reused, not rebuilt")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    import time

    namespace = {"ENV_ROOT": env_root, "ROOT": tmp_path, "CARRIED_HASHES": {TEMPLATE["lock"]: lock_sha}, "NOTEBOOK_SOURCE": {"revision": "r"}, "SESSION_START": time.perf_counter(), "Path": Path}
    exec("import hashlib, json, os, shutil, subprocess, time\n" + install, namespace)  # noqa: S102
    assert namespace["ENV_REUSED"] is True and namespace["RUNTIME"] == {"python": "3.12.12"}
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        assert name not in namespace["ENV"]
    assert namespace["ENV"]["MPLBACKEND"] == "Agg"
    # a different lock digest is not reused: the marker is dropped and the (forbidden here) download is attempted
    namespace["CARRIED_HASHES"] = {TEMPLATE["lock"]: "b" * 64}
    with pytest.raises(AssertionError, match="must be reused"):
        exec("import hashlib, json, os, shutil, subprocess, time\n" + install, namespace)  # noqa: S102
    assert not (env_root / "ready.json").exists()


def test_mpf_m5_run_stage_names_the_cells_to_rerun_when_the_run_directory_is_empty(tmp_path: Path) -> None:
    _carrier, install = _infrastructure_sources()
    namespace = {"ROOT": tmp_path / "empty", "WEIGHTS": tmp_path / "w", "PYTHON": Path(sys.executable), "ENV": dict(os.environ)}
    exec(install[install.index("def run_stage(") : install.index("def load_record(")], namespace)  # noqa: S102
    with pytest.raises(RuntimeError, match=r"run the three Infrastructure cells again in order \(Sections 1, 2 and 3\)"):
        namespace["run_stage"]("prepare")
    troubleshooting = next(c["source"] for c in _cells() if c["source"].startswith("## Troubleshooting"))
    assert "run the three Infrastructure cells again in order (Sections 1, 2 and 3)" in troubleshooting


# ---- MPF-m2 / m3 / m4 on the stub stage runner ---------------------------------------------------------------------


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return path


def test_mpf_m2_m3_video_reference_and_provenance_inventory(stub_run) -> None:  # noqa: F811
    stages, root, run = stub_run
    for stage in ("prepare", "inference", "evaluate", "robustness", "blendshapes", "newdata"):
        run(stage)
    (root / "outputs" / "weights.json").write_text(json.dumps({"members": []}), encoding="utf-8")
    run("export")
    out = root / "outputs"
    newdata = json.loads((out / "newdata.json").read_text(encoding="utf-8"))
    for mode in ("IMAGE", "VIDEO"):
        row = newdata["video_mode"][mode]
        assert {"nme_vs_true_motion_mean", "jitter_after_motion_removed", "self_consistency_nme_mean", "tracked"} <= set(row)
    assert "annotation moved by the known rotation" in newdata["video_mode_definitions"]["nme_vs_true_motion_mean"]
    result = json.loads((out / f"{stages.STEM}_result.json").read_text(encoding="utf-8"))
    assert result["run_id"] == root.name
    assert result["optional_branches"] == {"byod": False, "activity": False}
    listed = {row["file"]: row for row in result["files"]}
    assert f"{stages.STEM}_result.json" not in listed and "newdata.json" in listed
    for name, row in listed.items():
        data = (out / name).read_bytes()
        assert row["bytes"] == len(data) and row["sha256"] == hashlib.sha256(data).hexdigest(), name


def test_mpf_m3_main_records_stage_seconds(stub_run, monkeypatch) -> None:  # noqa: F811
    stages, root, _run = stub_run
    weights = root.parent / "weights"
    assert stages.main(["--root", str(root), "--weights", str(weights), "--stage", "prepare", "--keep-stderr"]) == 0
    timings = json.loads((root / "state" / "timings.json").read_text(encoding="utf-8"))
    assert set(timings) == {"prepare"} and timings["prepare"] >= 0


def test_mpf_m4_byod_zip_skips_and_reports_non_images_and_keeps_inputs_out_of_outputs(stub_run, tmp_path: Path) -> None:  # noqa: F811
    _stages, root, run = stub_run
    for stage in ("prepare",):
        run(stage)
    image = tmp_path / "face.png"
    synthetic_image(width=300, height=300, seed=4).save(image)
    archive = _zip(tmp_path / "phone.zip", {"face.png": image.read_bytes(), "README.txt": b"notes", "a/meta.json": b"{}"})
    run("byod", byod=str(archive), num_faces=1)
    record = json.loads((root / "outputs" / "byod" / "byod_result.json").read_text(encoding="utf-8"))
    assert record["skipped_non_image_members"] == ["README.txt", "a/meta.json"]
    assert record["faces_per_image"] == {"face.png": 1} and record["run_id"] == root.name
    produced = sorted(p.name for p in (root / "outputs" / "byod").iterdir())
    assert all(name.startswith("byod_") for name in produced), produced  # results only, no copies of the user's images
    only_text = _zip(tmp_path / "text.zip", {"README.txt": b"notes"})
    with pytest.raises(ValueError, match=r"holds no image files .*README\.txt"):
        run("byod", byod=str(only_text))
    broken = _zip(tmp_path / "broken.zip", {"photos/bad.jpg": b"not an image"})
    with pytest.raises(ValueError, match=r"^photos/bad\.jpg: not a decodable image"):
        run("byod", byod=str(broken))


# ---- stage-process import boundary (cloud-PR check, 2026-10-10) ---------------------------------------------------


def test_stage_processes_import_neither_ipython_nor_google() -> None:
    """Stages run as `tutorial_stages.py` subprocesses in the isolated environment, which has neither IPython nor
    google.colab: only kernel cells use them (`IPython.display` in the install cell, the BYOD upload dialog). A carried
    module that imported either would fail on Colab; there is no worker and no google.colab stub to give a ModuleSpec."""
    carried = [ROOT / source for dest, source in TEMPLATE["carried"].items() if dest.endswith(".py")]
    assert any(path.name == "tutorial_stages.py" for path in carried)
    offenders = [str(path) for path in carried if re.search(r"^\s*(from|import)\s+(IPython|google)\b", path.read_text(encoding="utf-8"), re.M)]
    assert not offenders, offenders
    sources = "\n".join("".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"])
    stubs = ("sys.modules['google", 'sys.modules["google', "ModuleType('google", 'ModuleType("google')
    assert not [marker for marker in stubs if marker in sources]
