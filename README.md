# MediaPipe Face Landmarker Pipeline

DIMER-oriented pipeline for the **MediaPipe Face Landmarker** task bundle (`mediapipe-models/face_landmarker/face_landmarker`, version `float16/1`): 478 3D facial landmarks, 52 blendshape scores and a 4 × 4 facial transformation matrix per face, from a still image or a frame sequence, on the CPU. The bundle is pinned to an immutable Cloud Storage object generation and verified member by member. The repository adds input validation, a stable CSV/JSON output contract, landmark evaluation against annotated public faces with two mean-shape baselines, robustness and blendshape sanity checks, a `MODEL_CARD.md` at DIMER Model Card Specification 1.2, and a standalone `TASK-INFERENCE` tutorial at DIMER Notebook Specification 2.2.

**This model does not recognise or identify people.** It returns face geometry and expression coefficients. Faces are nevertheless personal data, and landmarks and blendshapes derived from them can be biometric data; see `MODEL_CARD.md` for intended and prohibited uses.

## Upstream alignment

- Model: `mediapipe-models/face_landmarker/face_landmarker`, version `float16/1`, architecture source `google-ai-edge/mediapipe`
- Revision: Cloud Storage object generation `1683136941916318` of `face_landmarker.task`
- Bundle digest: 3,758,596 bytes, SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
- Bundle members: `face_detector.tflite` (BlazeFace short range), `face_landmarks_detector.tflite` (478-point face mesh), `face_blendshapes.tflite` (52 blendshapes), `geometry_pipeline_metadata_landmarks.binarypb` (canonical face for the transformation matrix)
- Upstream licence: **Apache-2.0**
- Runtime: `mediapipe==1.0.0` (Tasks API `FaceLandmarker`) + `numpy==2.5.3` + `pillow==12.3.0`, CPU (TFLite XNNPACK), float16 weights; nothing is unpickled and no code is downloaded with the model
- Repository adaptation: **none** — the bundle is a fixed edge model; the tutorial is `TASK-INFERENCE`

## Quick start

```python
from mediapipe_face_landmarker_pipeline import (
    FaceLandmarkerPipeline, stage_bundle, validate_image, landmark_rows, blendshape_rows, matrix_records,
)

stage_bundle(allow_download=True)                  # fetches generation 1683136941916318 if absent, refuses other bytes
image, report = validate_image("portrait.jpg")      # decodable, 64..8192 px, <= 40 MP, converted to RGB (reported)
with FaceLandmarkerPipeline.from_weights(num_faces=1) as landmarker:   # verifies size, SHA-256 and every member
    faces = landmarker.detect(image)                # [] when no face is found
rows = landmark_rows("portrait", faces, *image.size)     # image_id, face_id, landmark_id, x, y, z, x_px, y_px, z_px
scores = blendshape_rows("portrait", faces)              # 52 named scores per face (not probabilities)
matrices = matrix_records("portrait", faces)             # 4 x 4 facial transformation matrix per face
```

VIDEO mode: `FaceLandmarkerPipeline.from_weights(running_mode="VIDEO")` and `detect_frame(image, timestamp_ms)` with strictly increasing timestamps.

## Weights layout

```
weights/mediapipe-face-landmarker-float16-v1/   dimer-base-manifest.json
                                                face_landmarker.task   (git-ignored, fetched at run time)
weights/face-samples/                           the pinned sample faces, cached on first fetch (git-ignored)
```

`docs/WEIGHTS.md` records the identity, the bundle members, the serialization format and the decision on the DIMER upload format (`.task` is uploaded unchanged; conversion to safetensors is not valid for this model).

## Evaluation and sample data

`fetch_samples()` downloads 327 files from `debruine/webmorphR.stim` at commit `fa8b78fda2d659bb74ce62fcd99c4407551d2a77`, each pinned by size and SHA-256: the 102 neutral and 102 smiling front-facing photographs of the Face Research Lab London Set with the authors' 189-point annotations and participant information (CC BY 4.0, DeBruine & Jones, 2017), and ten composite faces (CC BY 4.0, DeBruine, 2016). The participants gave signed consent for research and illustration use.

- **Landmarks:** 14 annotated points have an unambiguous MediaPipe counterpart (`LANDMARK_MAP`); the NME divides their mean error by the annotated inter-ocular distance. Two leave-one-out baselines need no landmark network: the mean shape in image pixels, and the mean shape placed in the bundle's own face-detector box. Detection rate, failure rate at NME 0.10, per-point errors and a self-reported-group breakdown are reported.
- **Robustness:** rotation, downscaling, Gaussian blur, brightness and JPEG on 20 faces, with exact geometric maps so predictions are compared with the unperturbed prediction (consistency) and with the annotation.
- **Blendshapes:** a paired sanity check on neutral vs smiling photographs of the same 102 people. Scores are not calibrated probabilities.

In the local CPU run, reproduced exactly in a fresh Google Colab runtime, the model's mean NME was 0.01937 against 0.0752 (detector-box mean shape) and 0.1532 (image mean shape); every one of 102 smiles raised the smile score; faces rotated by 120° or 180° were mostly still detected but with misplaced landmarks. All numbers are tutorial / sanity evidence; see `docs/release-verification.md`.

## Tests

```
pip install -e . --no-deps
pip install pytest numpy pillow matplotlib
pytest
```

Tests run offline and need no `mediapipe`: a synthetic bundle tests verification, synthetic images test validation, and the stage runner runs every stage against a stub landmarker. The real model was executed locally; see `docs/release-verification.md`.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/mediapipe-face-landmarker-pipeline/blob/main/tutorials/mediapipe_face_landmarker_colab.ipynb)

`tutorials/mediapipe_face_landmarker_colab.ipynb` is declared `TASK-INFERENCE` / `GUIDED` and is **standalone** (DIMER Notebook Specification 2.2 §4): it is generated by `tools/build_notebook.py` from `tools/notebook_template.py` and carries, byte for byte and SHA-256-verified, the package's four modules and sample manifest, the stage runner `tools/tutorial_stages.py`, the hash-locked requirements `tutorials/requirements-colab.lock.txt`, the bundle manifest and the licence.

The notebook installs nothing into its own kernel. It downloads a pinned `uv` wheel (checked by size and SHA-256), builds an isolated CPython 3.12.12 environment, and installs the lock into it with `--require-hashes --only-binary :all:`; each stage (`weights`, `prepare`, `inference`, `evaluate`, `robustness`, `blendshapes`, `newdata`, `export`, optional `byod` and `activity`) then runs in its own process and hands results to the next only through files. A hosted runtime's preloaded NumPy and OpenCV — which `mediapipe` would otherwise replace — are never touched, so `Run all` needs no restart. It runs on Linux x86_64 only and refuses other platforms with a clear message; a CPU runtime is enough. The lock is compiled from the `pyproject.toml` pins (`uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes --only-binary :all: -o tutorials/requirements-colab.lock.txt`); recompile it and regenerate the notebook whenever a pin changes. `tools/execute_notebook.py` executes a copy with form fields set (EXE1–EXE7).

## Release status

**Candidate** — the notebook's default path, the BYOD branch (two compatible and two incompatible inputs) and the optional activity were executed end to end in a local CPU-only Linux container, and a fresh Google Colab runtime completed `Run all` of the same notebook in one pass with identical results. The hosted BYOD journey (REL12), including the upload dialog, is not yet recorded, and `.task` acceptance by the DIMER upload path is an open item (`docs/WEIGHTS.md` §3). See `STATUS.md` and `docs/release-verification.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
