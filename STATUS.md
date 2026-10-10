# Release status

Current status: **Candidate** — the `TASK-INFERENCE` / `GUIDED` notebook `tutorials/mediapipe_face_landmarker_colab.ipynb` runs its stages in an isolated hash-locked environment (pinned `uv` 0.12.15 wheel, managed CPython 3.12.12, `uv pip install --require-hashes --only-binary :all:`) and installs nothing into the kernel. Its default path, the BYOD branch and the optional activity were executed end to end in a local CPU-only Linux x86_64 container on 2026-10-04, at notebook blob `9b8a54d` generated from commit `e49deff` (recorded in `docs/release-verification.md`). That local run is pre-flight evidence only. On 2026-10-04 a fresh Google Colab runtime completed `Run all` of the same blob in one pass with defaults unchanged, reproducing every local metric (REL1, REL2, REL10, REL11 for the default path; recorded in `docs/release-verification.md`). The tutorial stays Candidate until the hosted BYOD journey below is recorded.

## Open items before Release-grade

1. The REL12 BYOD journey on the hosted runtime: one compatible input and one incompatible input, including the Colab upload dialog (the local run used `BYOD_PATH`; the Colab run left BYOD off).
2. Maintainer confirmation that the DIMER upload path accepts the upstream `.task` bundle unchanged (`docs/WEIGHTS.md` §3); `.task` is not one of the upload formats the fleet inventory lists.
3. Confirmation of the upstream publication date: the card records 2023-05-03, the upload time of the pinned object generation, because no public source for a later announcement date was checked here.

## Recorded runs since the review fixes

- **Executed (Google Colab T4, 2026-10-10):** a one-pass default run of the review-fix blob `3f4143f` (commit `7c0f9a2`, MPF-m1..m5) on a fresh session with the Colab CLI: 13/13 code cells, no error, no restart; every metric equals the 2026-10-04 run. Evidence in `docs/execution-evidence/2026-10-10-7c0f9a2/`. Status stays Candidate: the hosted BYOD journey (item 1 above) is still open.

## What is in place

- Bundle pinned by Cloud Storage object generation `1683136941916318`, size and SHA-256, with each of the four zip members verified before every load; no fallback source.
- 327 sample files pinned by size and SHA-256 at a fixed commit of `debruine/webmorphR.stim` (CC BY 4.0, consented participants).
- Offline CI: ruff, unit tests (bundle verification, validation, output contract, metrics and perturbation geometry, sample manifest, every stage against a stub landmarker, the kernel's carrier and `run_stage`, the executor), `tools/validate_release_assets.py` and `tools/build_notebook.py --check`.
