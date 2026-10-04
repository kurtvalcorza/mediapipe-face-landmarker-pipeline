# Release verification

`tutorials/mediapipe_face_landmarker_colab.ipynb` (`TASK-INFERENCE`, `GUIDED`, **standalone** carrier) is a **release
candidate** until the exact notebook revision has executed top to bottom in a clean supported hosted runtime. Unit
tests, JSON validation, code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are
necessary checks but are **not** runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the
durable release-gate record.

## Automatic coverage (static and offline, every pull request)

CI runs, without `mediapipe` and without network access to the model or samples:

- `ruff check src tests tools`;
- `pytest`: bundle verification against a synthetic bundle (size, SHA-256, every member, unsafe member names, manifest
  identity drift, duplicate keys), staging that refuses tampered bytes and writes nothing on a mismatch, input
  validation (undecodable bytes, side and pixel ceilings, missing files, reported RGB and EXIF changes, `num_faces` and
  confidence ranges), the output contract (field names, ids, pixel conversion, 52 blendshape names), the metrics and the
  perturbation geometry (a marked pixel tracked through each geometric map and back), the paired statistics, the sample
  manifest (327 pinned files, every neutral photograph with a template and a smiling pair) and its digest-checked fetch;
- `tests/test_tutorial_stages.py`: executes the generated notebook's carrier and `run_stage` code, shows that an invalid
  BYOD input stops the kernel with the validator's message, and runs every stage (`prepare` → `export`, then `activity`
  and `byod`) against a stub landmarker and synthetic files laid out at the pinned sample paths — plumbing only, not the
  model;
- `tests/test_notebook_parity.py` and `tools/build_notebook.py --check`: carried files, hashes, lock and manifest equal
  the repository; the notebook is byte-identical to the generator output;
- `tools/validate_release_assets.py`: notebook structure (nbformat 4, no outputs, every code cell compiles and follows a
  markdown cell, no magics), the isolated install (pinned `uv` URL, size and 64-hex SHA-256, managed CPython 3.12.12,
  `--require-hashes --only-binary :all:`, no `pip install`, `sys.executable` or `importlib` in kernel cells, kernel
  imports limited to the standard library, `IPython.display` and `google.colab`), the stage order, the three BYOD and two
  activity form fields, the guided layer, the citations, the carried runner's required calls and exports, identity
  consistency (object generation and bundle SHA-256 in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md`), weight facts
  (every byte count and digest quoted in those documents comes from a manifest), the release-status tokens, and the
  model-card front matter and 19 required headings.

## Manual coverage

The model itself is not executed in CI. It is executed by running the notebook, either in a hosted runtime or with
`tools/execute_notebook.py`, which sets form fields in a copy and fails if a field is not found (EXE6).

## Recorded executions

### Google Colab hosted runtime, 2026-10-04 (default path)

- **Subject:** `tutorials/mediapipe_face_landmarker_colab.ipynb`, git blob `9b8a54dcc45294998578e00dcb7d67e0a27b6203`,
  generated from commit `e49deff733c8aa7a91cd99d12f9fb4980a181935`, opened in Colab from branch `ccr-24656dfc-ax1ln2`
  at `fd237b1`. Every source cell of the executed copy is byte-identical to that blob. The executed copy is
  `docs/execution-evidence/2026-10-04/mediapipe_face_landmarker_colab_9b8a54d_colab-t4.ipynb`, and
  `colab_t4_run_summary.json` beside it summarises the run.
- **Runtime:** fresh Google Colab runtime with a T4 GPU attached (the notebook does not use it; MediaPipe ran on the CPU
  delegate, TFLite XNNPACK), 2 vCPUs, x86_64, kernel CPython 3.13.15. Stages: the notebook's isolated environment —
  CPython 3.12.12 managed by `uv`, the carried 19-package lock (`mediapipe` 1.0.0, `numpy` 2.5.3, `pillow` 12.3.0),
  built in 13 s.
- **Procedure:** `Run all` with defaults unchanged (`USE_BYOD = False`, `RUN_ACTIVITY = False`).
- **Observed result:** all 13 code cells completed in one pass in execution order 1–13, with no error, no restart, no
  credential and no upload dialog. The bundle and its four members verified; 327 sample files fetched and verified; 214
  photographs validated; the four refusal probes refused and the greyscale probe accepted with `L -> RGB` reported.
  Stage times: weights 0.4 s, prepare 15.7 s, inference 1.7 s, evaluate 15.4 s, robustness 37.3 s, blendshapes 13.6 s,
  newdata 4.7 s, export 0.2 s (89 s in stages, plus the 13 s environment build). Every printed metric equals the local
  CPU pre-flight below: `nme_mean` 0.01937 (95 % bootstrap [0.01868, 0.02006]), detection rate 1.0, baselines 0.0752 and
  0.1532, the same group, robustness and VIDEO-mode values, rotation 180° detection 0.9 and failure 0.9, smile score
  higher in 102/102 pairs, composite-face NME mean 0.01432.
- **Caveats:** the optional BYOD branch and activity were not run, so the REL12 BYOD journey on the hosted runtime,
  including the upload dialog, is still open. One pass, no repeated runs. `mediapipe` 1.0.1 was not tried on Colab.

### Local CPU pre-flight, 2026-10-04 (not hosted-runtime evidence)

- **Subject:** `tutorials/mediapipe_face_landmarker_colab.ipynb`, git blob `9b8a54dcc45294998578e00dcb7d67e0a27b6203`,
  generated from commit `e49deff733c8aa7a91cd99d12f9fb4980a181935` (the carried files are that commit's files byte for
  byte). The executed copies and `run_summary.json` are in `docs/execution-evidence/2026-10-04/`. An earlier execution of
  the pre-commit working tree gave identical values.
- **Runtime:** CPU-only Linux x86_64 container, 4 CPUs, no GPU. Kernel: CPython 3.11.15 with `nbconvert` 7.17.1 and
  `ipykernel` 7.4.0. Stages: the notebook's isolated environment — `uv` 0.12.15 from the pinned wheel, CPython 3.12.12
  downloaded fresh by `uv` (empty `UV_PYTHON_INSTALL_DIR`), the carried lock (19 packages, `mediapipe` 1.0.0, `numpy`
  2.5.3, `pillow` 12.3.0, `opencv-contrib-python` 5.0.0.93, `matplotlib` 3.11.2).
- **Procedure:** `jupyter nbconvert --execute` of a copy, from an empty working directory (no cached bundle, samples,
  environment or Python), defaults unchanged. Then four copies via `tools/execute_notebook.py` with `USE_BYOD = True`,
  `BYOD_PATH` set (no upload dialog) and `RUN_ACTIVITY = True`.
- **Observed result (default path):** all 13 code cells completed in one pass, no restart, no credential, 129 s wall
  clock including the environment build (120 s for the committed blob; 129 s for the pre-commit tree). The bundle was fetched and its four members verified; 327 sample files fetched
  and verified; 214 photographs validated; the four refusal probes refused and the greyscale probe accepted with
  `L -> RGB` reported. Walkthrough: 1 face, 478 × 3 landmarks, 52 blendshapes, 4 × 4 matrix; `num_faces` 1 → 1 face and 2 →
  2 faces on the two-face collage; 0 faces on a blank image.
  - Landmarks (102 neutral photographs, 14 points): detection rate 1.0; `nme_mean` 0.01937 (95 % bootstrap
    [0.01868, 0.02006]), median 0.0194, p90 0.02449, max 0.02821, failure rate 0.0. Baselines: mean shape in the
    detector box 0.0752 (failure rate 0.1961); mean shape in the image 0.1532 (failure rate 0.7059). Largest per-point
    errors: `chin_bottom_centre` 0.0389, `subnasale` 0.0295.
  - Groups (self-reported, one run): female 0.0180 (n 49), male 0.0206 (n 53); black 0.0224 (n 13), East Asian 0.0192
    (n 9), West Asian 0.0167 (n 10), white 0.0193 (n 69).
  - Robustness (20 faces): detection 1.0 and failure 0.0 for rotation up to 90°, downscaling to 12 %, blur radius up to
    12 px, brightness 0.35 and 1.8, and JPEG quality 20 and 5 (worst annotation NME 0.0362 at blur 12). Rotation 180°:
    detection 0.9, failure 0.9 (annotation NME 1.6589).
  - Blendshapes (102 pairs): smile score higher on the smiling photograph in 102/102 pairs, median 0.0016 → 0.6378,
    rank AUC 0.948, sign-test p 3.9e-31. Descriptive: `eyeBlink` rose in 91 % of pairs, `eyeSquint` in 89 %.
  - New data: 10 composite faces, all detected, NME mean 0.01432; outputs exported. VIDEO-mode demo (30 frames, ±8°
    roll): consistency NME 0.004 (IMAGE) vs 0.0119 (VIDEO).
- **Observed result (optional branches):** for the committed blob, a zip of two composite faces with `BYOD_NUM_FACES = 2`
  (and, for the pre-commit tree, also a single composite image) produced landmarks, blendshapes, matrices, overlays and `byod_result.json` with `data_left_runtime: false`.
  A zip with a `../escape.jpg` member stopped with `Stage 'byod' failed (exit 2): ValueError: BYOD zip has an unsafe member
  path '../escape.jpg'; refusing the archive` (committed blob and pre-commit tree); a text file named `notes.jpg` stopped
  with `ValueError: notes.jpg: not a decodable image …` (pre-commit tree). The activity at 120° gave detection 0.8, failure 0.8.
- **Caveats:** a local container, not a clean hosted Colab runtime (REL1/REL10 not met); the BYOD inputs were public
  composite faces, and the Colab upload dialog was not exercised; one pass, no repeated runs. The container's proxy
  environment printed a `UV_NATIVE_TLS` deprecation warning from `uv`, judged harmless (it concerns certificate settings
  of this container only).

## Promotion rule

The status moves to `Release-grade` only after a fresh Google Colab CPU runtime completes `Run all` of the release
revision in one pass, and the REL12 BYOD journey (one compatible input, one incompatible input) is recorded here with the
commit, the notebook blob, the runtime and the outcome.
