"""EXE1-EXE7: the executor sets form fields in a copy and fails when a field is not found exactly once."""

# ruff: noqa: E501
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("execute_notebook", ROOT / "tools" / "execute_notebook.py")
executor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(executor)
NOTEBOOK = ROOT / "tutorials" / "mediapipe_face_landmarker_colab.ipynb"


def _notebook() -> dict:
    if not NOTEBOOK.exists():
        pytest.skip("notebook not generated yet")
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_fields_are_set_in_a_copy():
    original = NOTEBOOK.read_bytes() if NOTEBOOK.exists() else None
    notebook = executor.set_fields(_notebook(), {"USE_BYOD": "True", "BYOD_PATH": "'/data/face.jpg'", "BYOD_NUM_FACES": "3", "RUN_ACTIVITY": "True", "ACTIVITY_ROTATION": "150"})
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert 'USE_BYOD = True  # @param {type:"boolean"}' in code
    assert "BYOD_PATH = '/data/face.jpg'  # @param" in code and "BYOD_NUM_FACES = 3  # @param" in code
    assert 'ACTIVITY_ROTATION = 150  # @param {type:"number"}' in code
    assert NOTEBOOK.read_bytes() == original  # the committed notebook is never edited


def test_a_missing_field_fails_the_executor():
    with pytest.raises(SystemExit, match="found 0 times"):
        executor.set_fields(_notebook(), {"NEW_DATA_PATH": "'x'"})
