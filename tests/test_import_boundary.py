"""Import-boundary contract (fleet RTM-001): verification and validation refuse before `mediapipe` is imported."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from conftest import MODEL_LIBRARIES, synthetic_image
from mediapipe_face_landmarker_pipeline import validate_image, validate_num_faces, verify_bundle

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "mediapipe_face_landmarker_pipeline"


def test_package_import_does_not_import_model_libraries(forbid_model_imports):
    import importlib

    import mediapipe_face_landmarker_pipeline

    importlib.reload(mediapipe_face_landmarker_pipeline)


def test_no_module_level_model_imports():
    for path in PACKAGE.glob("*.py"):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert name.partition(".")[0] not in MODEL_LIBRARIES, f"{path.name} imports {name} at module level"


def test_invalid_inputs_are_rejected_before_model_imports(forbid_model_imports, tmp_path):
    with pytest.raises(ValueError, match="num_faces must be in 1..10"):
        validate_num_faces(11)
    with pytest.raises(ValueError, match="each side must be within 64..8192 px"):
        validate_image(synthetic_image(width=40, height=40))
    with pytest.raises(FileNotFoundError, match="dimer-base-manifest.json is missing"):
        verify_bundle(tmp_path)
