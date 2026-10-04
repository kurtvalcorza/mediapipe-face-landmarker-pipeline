# Release status

Current status: **Candidate** — the `TASK-INFERENCE` / `GUIDED` notebook `tutorials/mediapipe_face_landmarker_colab.ipynb` runs its stages in an isolated hash-locked environment (pinned `uv` 0.12.15 wheel, managed CPython 3.12.12, `uv pip install --require-hashes --only-binary :all:`) and installs nothing into the kernel. Its default path, the BYOD branch and the optional activity were executed end to end in a local CPU-only Linux x86_64 container on 2026-10-04, at notebook blob `9b8a54d` generated from commit `e49deff` (recorded in `docs/release-verification.md`). That local run is pre-flight evidence only: it is **not** the clean hosted-runtime (Google Colab) execution that NOTEBOOK_SPEC 2.2 REL1/REL10 require, and the tutorial must stay Candidate until such a run is recorded for the release revision.

## Open items before Release-grade

1. A fresh Google Colab CPU runtime `Run all` of the release revision, recorded with commit, notebook blob, runtime and outcome (REL1, REL2, REL10, REL11).
2. The REL12 BYOD journey on the hosted runtime, including the Colab upload dialog (the local run used `BYOD_PATH`).
3. Maintainer confirmation that the DIMER upload path accepts the upstream `.task` bundle unchanged (`docs/WEIGHTS.md` §3); `.task` is not one of the upload formats the fleet inventory lists.
4. Confirmation of the upstream publication date: the card records 2023-05-03, the upload time of the pinned object generation, because no public source for a later announcement date was checked here.

## What is in place

- Bundle pinned by Cloud Storage object generation `1683136941916318`, size and SHA-256, with each of the four zip members verified before every load; no fallback source.
- 327 sample files pinned by size and SHA-256 at a fixed commit of `debruine/webmorphR.stim` (CC BY 4.0, consented participants).
- Offline CI: ruff, unit tests (bundle verification, validation, output contract, metrics and perturbation geometry, sample manifest, every stage against a stub landmarker, the kernel's carrier and `run_stage`, the executor), `tools/validate_release_assets.py` and `tools/build_notebook.py --check`.
