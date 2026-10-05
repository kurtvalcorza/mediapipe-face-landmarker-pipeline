# MediaPipe Face Landmarker notebook — review fixes

**Review:** `mediapipe_face_landmarker_colab_Review.md` (5 October 2026, MPF-m1..m5, no Majors).
**Fixed in:** the generator (`tools/build_notebook.py`, `tools/notebook_template.py`, `tools/tutorial_stages.py`); the notebook was regenerated and `--check` passes.
**Readiness:** **Verification pending** until the hosted gates below are recorded. `STATUS.md` and every release label are unchanged.

This repository is the pilot for the uv-template re-run fixes (idempotent Section 1, environment reuse keyed on the lock, `PYTHONSTARTUP` dropped from the stage environment). The same `build_notebook.py` change is applied to the other isolated-runtime notebooks in this fix cycle.

## Findings

| ID | Status | Change | Cells / files | Evidence |
|---|---|---|---|---|
| MPF-m1 | Fixed | The opening cell quotes the recorded Colab run (4 October 2026, 2 vCPUs, revision `e49deff`, 13 s environment build, 89 s in stages) instead of "has not yet been recorded". *Running it* and the install note quote the 13 s build and the 0.2–37 s stage range; the Prerequisites give the measured 0.52 GB + 0.11 GB environment and label the 1.5 GB disk check as including a margin; the disk row in Troubleshooting says the same. | `run_all`, How to use, Prerequisites, install note, Troubleshooting (`tools/notebook_template.py`) | `test_mpf_m1_no_stale_hosted_run_statement_and_figures_are_measured_or_labelled`; `grep "has not yet been recorded"` on the notebook returns nothing |
| MPF-m2 | Fixed | Both running modes are scored against the composite's own annotation moved by the known roll (`nme_vs_true_motion_mean`). Jitter is now frame-to-frame landmark displacement after the known motion is removed (`jitter_after_motion_removed`). The old metric is kept as `self_consistency_nme_mean`, labelled as favouring IMAGE mode by construction. `still_prediction_nme_vs_annotation` is reported. Definitions are written to `newdata.json` and `result.json`. Section 9 names the reference; the untested tracking-lag explanation is removed. Suggestion S4 ("same London Set people") was folded into the same note. | `stage_newdata`, `stage_export` (`tools/tutorial_stages.py`); Section 9 prose | `test_mpf_m2_m3_video_reference_and_provenance_inventory`; real run below: IMAGE 0.0105, VIDEO 0.0168 against the moved annotation (the review's P3 values), still prediction 0.0103 |
| MPF-m3 | Fixed | `result.json` now holds `run_id` (the run directory name), `stage_seconds` (written by the runner after each successful stage to `state/timings.json`), `optional_branches`, and a `files` inventory: relative path, bytes and full SHA-256 of every file in `outputs/` except the record itself. `byod_result.json` carries `run_id`. Section 10 prose describes the binding and says how to match a downloaded CSV. | `stage_export`, `stage_byod`, `main` (`tools/tutorial_stages.py`); Section 10 prose | `test_mpf_m2_m3_video_reference_and_provenance_inventory` (every listed digest equals the file's SHA-256), `test_mpf_m3_main_records_stage_seconds`; real run below |
| MPF-m4 | Fixed | Zip members without an image extension are skipped and listed (`skipped_non_image_members`, printed and recorded); a zip with no image member is refused naming the skipped members. Every message and output row uses the member path inside the archive (for example `photos/face.jpg`), not a renamed copy. Members are extracted to `byod_inputs/` in the run directory, outside `outputs/`, so `outputs/byod/` holds only results. Section 11 and the Troubleshooting row state both rules. | `_byod_inputs`, `stage_byod`; Section 11 prose; Troubleshooting | `test_mpf_m4_byod_zip_skips_and_reports_non_images_and_keeps_inputs_out_of_outputs`; real run below |
| MPF-m5 | Fixed (stronger than the acceptance check) | Re-running the Section 1 cell keeps this session's run directory, so later cells keep working instead of failing; a `NEW_RUN_DIRECTORY` form field (off) gives a fresh one on request. The isolated environment directory is keyed on the lock digest, managed Python and `uv` version; a `ready.json` marker written after a complete install lets a later Run all reuse it (`environment_reused: True`) with no download. `run_stage` checks that the run directory has the carried runner and that the interpreter exists, and otherwise names the cells to re-run (Sections 1, 2 and 3); the same text is in a new Troubleshooting row. | `CHECK_CELL`, `INSTALL_CELL`, `run_stage` (`tools/build_notebook.py`); How to use, Section 1 prose, Troubleshooting; validator title | `test_mpf_m5_section1_rerun_keeps_the_run_directory`, `test_mpf_m5_environment_is_keyed_on_the_lock_not_the_run`, `test_mpf_m5_install_cell_reuses_a_complete_environment_without_downloading`, `test_mpf_m5_run_stage_names_the_cells_to_rerun_when_the_run_directory_is_empty`; the review's P5 driver below |

Suggestions S1–S3, S5 and S6 are not taken in this cycle.

## User-visible changes

- Section 1 has a new form field, `NEW_RUN_DIRECTORY` (off). Re-running Section 1 in the same session keeps the run directory; a second Run all in the same kernel writes into the same run directory and reuses the isolated environment.
- The isolated environment lives at `<tmp>/mediapipe_face_landmarker_env_<lock key>` instead of `<tmp>/mediapipe_face_landmarker_env_<run id>`; the install cell prints `environment_reused`.
- `newdata.json` / `result.json`: `video_mode.<mode>` now has `nme_vs_true_motion_mean`, `jitter_after_motion_removed`, `self_consistency_nme_mean` (was `consistency_nme_mean`, `frame_to_frame_change`), plus `video_mode_definitions`.
- `result.json` gains `run_id`, `stage_seconds`, `optional_branches` and `files`.
- BYOD zips: non-image members are skipped and reported instead of refusing the archive; output `image_id`s are the archive member paths (were `NN_<basename>`); extracted copies go to `byod_inputs/`, not `outputs/byod/`.
- The stage environment also drops `PYTHONSTARTUP`.

## Verification (offline, not clean-runtime evidence)

- `python tools/build_notebook.py --check`: OK. `python tools/validate_release_assets.py`: PASS. `ruff check src tests tools`: clean.
- `pytest` with CI's dependencies (no mediapipe): 58 passed before, **67 passed** after (9 new tests in `tests/test_review_fixes.py`; one expectation in `test_tutorial_stages.py` updated for the member-path image ids).
- **Real input, real bundle, local CPU container (not a hosted runtime):** the review's own P5 driver (`p5_rerun.py` from the probe zip) executed the regenerated notebook in one Jupyter kernel: two complete Run alls, then the Section 1 cell alone, then the Section 4 cell. Pass 1: 13/13 cells, environment built; pass 2: "Keeping the run directory of this session", "Reusing the isolated environment built earlier in this runtime from the same lock", samples `reused_from_cache: 327`; partial re-run: **cell 18 succeeded**. Default-path numbers unchanged: NME mean 0.01937 (95 % [0.01868, 0.02006]), detection 102/102. `result.json`: `run_id` equal to the run directory, `stage_seconds` for all eight stages, `optional_branches` false/false, 24 files whose digests matched `sha256` before the partial re-run rewrote `prepare.json`. VIDEO comparison against the moved annotation: IMAGE 0.0105, VIDEO 0.0168; jitter 0.0020 vs 0.0039; self-consistency 0.0039 vs 0.0118.
- **Real BYOD stage, same environment:** a zip with `photos/face.jpg` (a pinned London Set photograph) and `README.txt` processed the image as `photos/face.jpg` (1 face) and recorded `skipped_non_image_members: ['README.txt']`; `outputs/byod/` held only the five `byod_*` result files. A zip with only `README.txt` was refused: `BYOD zip byod_text.zip holds no image files (… skipped ['README.txt'])`.
- The Colab upload dialog was not exercised.

## Remaining gates

1. A hosted one-pass **Run all** of the regenerated notebook in a fresh runtime (no restart), then a re-run of the export cell.
2. The REL12 BYOD journey on a hosted runtime: one compatible and one incompatible input, including the Colab upload dialog, recorded in `docs/release-verification.md`.
3. Maintainer decisions carried from the review: whether 1.5 GB stays as the disk margin; `STATUS.md` items (`.task` acceptance, upstream publication date).
