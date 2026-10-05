# MediaPipe Face Landmarker Task-Inference Notebook — Review

**Verdict: Needs revision** (no Major findings; five Minors, one of which breaches a MUST)  
**Review date:** 5 October 2026  
**Repository:** `kurtvalcorza/mediapipe-face-landmarker-pipeline`  
**Notebook:** `tutorials/mediapipe_face_landmarker_colab.ipynb`  
**Reviewed commit:** `1594338b1b1ee8443e7126daab19d6d7baffcb0c` (`main`, the merge of PR #1)  
**Notebook Git blob:** `9b8a54dcc45294998578e00dcb7d67e0a27b6203`, generated from `e49deff`. This is the blob executed in the recorded Colab run of 2026-10-04 and in this review's own run. At the reviewed commit `tools/build_notebook.py --check` and `tools/validate_release_assets.py` both exit 0.  
**Finding prefix:** `MPF`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2, `ml-worker` `origin/main` at `b9fdd1f`.

## Executive assessment

The notebook is in good shape. It is a standalone, generator-built `TASK-INFERENCE` / `GUIDED` notebook that:

- installs nothing into the kernel: it uses a pinned `uv` wheel, managed CPython 3.12.12 and a 19-package hash lock, and runs every stage in a subprocess;
- stages the `.task` bundle by immutable Cloud Storage object generation and verifies all four members;
- fetches 327 digest-pinned public sample files;
- evaluates 14 landmarks on 102 annotated faces beside two honest mean-shape baselines;
- probes robustness with exact inverse geometric maps;
- runs a pre-registered, paired blendshape sanity check;
- applies the output contract to new images and a frame sequence;
- exports provenance.

The guided layer is complete:

- audience, how-to-use and roadmap;
- task contract and glossary;
- a prediction before every principal result, "What to notice" notes and collapsed checkpoint answers;
- a Predict → Change → Run → Observe → Explain activity;
- troubleshooting, a conclusion scaffold and a transfer prompt;
- Infrastructure-titled collapsed cells.

Score semantics, privacy and biometric caveats, and the limits of a 102-face studio set are stated accurately.

This review's own run reproduced the Colab record exactly:

| Measure | This review (CPU container, fresh working dir, defaults) | Colab record 2026-10-04 (same blob) |
|---|---|---|
| Code cells completed | 13/13, one pass, no restart (89 s of cell time incl. a 7 s environment build) | 13/13, one pass, no restart (89 s stages + 13 s build) |
| Landmark NME, model | mean 0.01937 (95 % bootstrap [0.01868, 0.02006]), detection 102/102 | identical |
| Baselines (detector box / image) | 0.0752 / 0.1532 | identical |
| Robustness, rotate 180° | detection 0.9, failure 0.9 | identical |
| Smile score higher on smiling photo | 102/102, rank AUC 0.948 | identical |
| Composite NME / VIDEO vs IMAGE | 0.01432 / 0.0119 vs 0.004 | identical |

Five Minors stand between this notebook and `Ready for intended use`:

1. **The opening cell says no hosted run exists (MPF-m1).** It also quotes setup and disk estimates that the recorded runs do not support.
2. **The IMAGE-versus-VIDEO comparison is anchored to an IMAGE-mode reference (MPF-m2).** It favours IMAGE mode by construction, and the explanation offered for the result is not tested.
3. **The provenance record does not bind the files it describes (MPF-m3).** It has no output inventory with digests and no run identifier.
4. **BYOD zip handling (MPF-m4).** One stray non-image member refuses the whole archive, under an internal file name.
5. **Re-running the Section 1 cell alone strands later cells (MPF-m5).** The error is unguided, and repeated Run alls each build a new 0.5 GB environment.

The REL12 BYOD journey on a hosted runtime, including the Colab upload dialog, has still not been recorded. This review exercised the BYOD branch locally (§4) and does not replace that gate.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `TASK-INFERENCE` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.2**, standalone (metadata, opening cell) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Stated (How to use): learners who can run hosted notebook cells and read short Python; no face-landmark background assumed; CPU runtime |
| Supported runtime | Fresh Linux x86_64 (Colab CPU, Kaggle, Linux Jupyter); kernel Python irrelevant; stages in CPython 3.12.12 |
| Promised outcomes | <ul><li>Isolated hash-locked install with no restart</li><li>Digest-verified bundle</li><li>327 pinned sample files</li><li>Validation with four refusals and one reported conversion</li><li>Full output contract on one photograph, plus the `num_faces` and no-face behaviour</li><li>NME on 102 faces versus two baselines and groups</li><li>Robustness on 20 faces × 15 settings</li><li>Paired blendshape sanity check</li><li>New-data export on 10 composites</li><li>IMAGE-vs-VIDEO comparison</li><li>Provenance record listing every file</li><li>BYOD (image or zip of ≤ 20 images)</li><li>Optional rotation activity</li></ul> |
| Generator | `tools/build_notebook.py` (`build_notebook.py/3.0-cpu`) + `tools/notebook_template.py`; generating revision `e49deff` |
| Release status | `Candidate` (`STATUS.md`, `docs/release-verification.md`): default path recorded on Colab; REL12 hosted BYOD pending; `.task` upload acceptance and publication date open (maintainer items, not notebook defects) |

### Evidence actually obtained

- **Source inspection.**
  - All 46 cells (13 code). Cell 9 is the carrier: it holds 10 files, hash-checked against `CARRIED_HASHES`.
  - `tools/tutorial_stages.py` (every stage, `_byod_inputs`, `stage_export`).
  - `src/mediapipe_face_landmarker_pipeline/{pipeline,metrics,samples}.py`: `validate_image`, `perturb` and the metrics.
  - `README.md`, `MODEL_CARD.md`, `STATUS.md`, `docs/release-verification.md`, `docs/WEIGHTS.md`.
  - The upstream Face Mesh V2 model card (PDF): its IOD definition and the 2.56 % human-annotator figure.
  - `debruine/webmorphR.stim` `README.md` at the pinned commit `fa8b78f`, for the composite construction.
- **Documented execution evidence.**
  - Source: `docs/release-verification.md` and `docs/execution-evidence/2026-10-04/`.
  - Colab, the reviewed blob, default path, 13/13. The BYOD and activity branches were not run on Colab.
  - The local CPU pre-flight of the same blob also ran BYOD and the activity.
- **Direct execution (this review).**
  - **Environment:**
    - CPU-only Linux x86_64 container with 4 CPUs.
    - Kernel CPython 3.11.15 with `nbconvert` and `ipykernel`.
    - Stages ran in the environment the notebook itself built: `uv` 0.12.15, CPython 3.12.12 freshly downloaded into an empty `UV_PYTHON_INSTALL_DIR`, and the carried lock (`mediapipe` 1.0.0, `numpy` 2.5.3, `pillow` 12.3.0).
  - **Clean working directory:** `tools/execute_notebook.py --workdir <empty>`, with no cached bundle, samples, environment or Python. Everything was fetched and verified fresh. Not a hosted runtime.
  - **Executed:**

    | Probe | What was run |
    |---|---|
    | P1 | Every code cell at defaults |
    | P2 | The 180° rotation mechanism (where the mesh lands on the rotated image), plus a rotation sweep with the notebook's own `perturbation_table` |
    | P3 | The VIDEO-mode comparison re-scored against the composite's annotation |
    | P4 | The BYOD stage through the stage runner the cell calls: six compatible and six incompatible inputs, then a repeated BYOD run |
    | P5 | A repeated Run all in one kernel, and a partial re-run of Section 1 followed by Section 4 |
    | P6 | The exported records |

    Scripts, logs and results are in `mediapipe_face_landmarker_colab_Review_Probes.zip`.
- **Learner observation:** none. No claim here is about measured learning effectiveness.

## 2. Separate judgments

- **Technical correctness:** good.
  - P1 is 13/13 and identical to the Colab record.
  - Bundle and sample verification are strict.
  - The inverse geometric maps are right: P2 confirms the 180° failures are genuine model behaviour, not a mapping bug.
  - Validation refusals name the rule.
  - One methodological weakness: MPF-m2 (the IMAGE-mode-anchored VIDEO comparison).
- **Promise fulfilment:** every promised stage runs and writes what it says, with three gaps:
  - the IMAGE-vs-VIDEO comparison measures agreement with IMAGE mode rather than "the known motion" (MPF-m2);
  - "each file is traceable … to the exact bundle that produced it" relies on files sitting together in one directory (MPF-m3);
  - the opening cell's runtime statement is out of date (MPF-m1).
- **Learner experience:** strong. Accurate concept prose; predictions with honest sample answers, for example that 180° detection survives while localisation fails; small groups are not over-read; the conclusion scaffold names baseline, failure mode and limits. The friction points are recovery after a partial re-run (MPF-m5) and BYOD zip errors (MPF-m4).
- **Spec conformance:**
  - One MUST is unresolved: SRC3, the knowingly stale statement (MPF-m1). UX12 also applies to its estimates.
  - SHOULD deviations: OUT3, OUT7 and OUT9 (MPF-m3), UX10 and DAT19 wording (MPF-m4), EVAL3 and UX4 (MPF-m2).
  - REL12 hosted BYOD evidence is absent from the release record (a gate, not a defect).
  - GDL1–GDL15 are all satisfied.

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass `Run all`, nothing installed into the kernel, no restart | cells 6, 9, 12 (`uv` venv + `--require-hashes`) | Colab 13/13; P1 13/13; kernel packages untouched | Section 2 prose explains why no restart is needed | Met |
| Digest-verified bundle at an immutable generation | cell 15 / `weights` | 4/4 members verified, generation `1683136941916318` | clear; "never edit a manifest" | Met |
| 327 pinned samples; validation; refusals | cell 18 / `prepare` | 214 validated; 4 refusals naming the rule; `L -> RGB` reported | blank-wall checkpoint | Met |
| Output contract on one photo; `num_faces` is a maximum; no-face is empty | cell 21 / `inference` | 478×3, 52, 4×4; 1 → 1, 2 → 2, blank → 0 | blendshape-not-probability checkpoint | Met |
| NME on 102 faces vs two baselines; groups | cell 24 / `evaluate` | 0.01937 vs 0.0752 / 0.1532; group `n` shown, including n = 1 and 2 | definition-difference and small-group checkpoint | Met |
| Robustness; "detection is not localisation" | cell 27 / `robustness` | 180°: detection 0.9, failure 0.9 | checkpoint explains the wrong-way-up mesh. P2 confirms it (18/18 detected meshes upright on the inverted face), but the figure does not show 180° (MPF-S1) | Met |
| Paired blendshape sanity check with a stated expectation | cell 30 / `blendshapes` | 102/102, AUC 0.948, sign-test p 3.9e-31; descriptive scores separate | "not an accuracy, no threshold" | Met |
| New-data export on 10 composites | cell 33 / `newdata` | 10/10 detected, NME 0.01432; three files written | "sanity check, not generalisation" | Met (see MPF-S4) |
| IMAGE vs VIDEO "closer to the known motion" | cell 33 | 0.004 vs 0.0119, scored against the IMAGE-mode still prediction | attributes the gap to tracking lag | **Partly met** (MPF-m2) |
| One provenance record; every file listed with size and digest | cell 36 / `export` | `result.json` written; listing printed with 16-hex digest prefixes, not saved | "each file is traceable … to the exact bundle" | **Partly met** (MPF-m3) |
| BYOD image or zip, same contract, data stays local | cell 39 / `byod` | P4: single, EXIF-rotated, RGBA, two-face and three-image zip accepted, with conversions reported; 6 incompatible inputs refused | privacy and contract stated first | Met locally; hosted REL12 pending; MPF-m4 |
| Optional activity: change the rotation angle | cell 42 / `activity` | local pre-flight 120° → detection 0.8, failure 0.8; P2 sweep; 200° refused | sample answer matches | Met |
| Runtime and setup statements | cells 0, 1, 4 | "a hosted Colab run has not yet been recorded"; "a minute or two" to install; 1.5 GB environment | — | **Not met** (MPF-m1) |

All eight learning objectives are phrased as observable learner actions (explain, diagnose, interpret, distinguish, apply, predict, write). Each is exercised by a prediction or checkpoint with a sample answer.

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| **First-time learner** | Source inspection, all 46 cells | The guided layer is complete and accurate (§2). The opening cell tells the learner no hosted run exists, which the record contradicts (MPF-m1). The VIDEO-mode "What to notice" note offers an untested causal explanation (MPF-m2). |
| **Clean default** | Documented (Colab, reviewed blob) + direct (P1, CPU container, empty working dir) | 13/13 in one pass on both. Every printed metric is identical across Colab, the earlier local pre-flight and P1. |
| **Active learning** | Direct (P2) + documented (local pre-flight) | The activity reruns only the rotation on the same 20 faces, writes `activity.json`, and leaves the canonical outputs untouched. Out-of-range input (200°) is refused with the rule. P2 sweep: <ul><li>100°: detection 1.0, failure 0.15</li><li>120°: detection 0.8, failure 0.8</li><li>135°: detection 0.75, failure 0.75</li><li>150°: detection 0.85, failure 0.85</li><li>165°: detection 0.85, failure 0.85</li><li>−120°: detection 0.6, failure 0.6</li><li>−150°: detection 0.4, failure 0.4</li></ul> The sample answer ("somewhere beyond 90° … detection often survives while the failure rate jumps") holds; the sign asymmetry is a possible extension (MPF-S6). |
| **Reuse and recovery** | Direct (P4, P5); Colab upload dialog not verified | See the two parts below. |

**Reuse and recovery, BYOD (P4).** Accepted, each with the conversion reported where one applied:

- a single photograph;
- an EXIF-orientation-6 JPEG (`EXIF orientation applied`; landmarks match the upright image);
- an RGBA PNG (`RGBA -> RGB (alpha channel dropped)`);
- a two-face image (`BYOD_NUM_FACES` 2 → 2 faces, 1 → 1);
- a three-image zip.

Refused, each naming the rule:

- `num_faces` 0 and 11;
- a missing path;
- text named `.jpg`;
- a 21-image zip;
- a zip holding one image plus `README.txt`. The whole archive is refused as `01_README.txt: not a decodable image` (MPF-m4).

A second BYOD run replaces `outputs/byod/` entirely, so stale outputs cannot mix.

**Reuse and recovery, re-runs (P5).** **Repeated Run all in one kernel:**

- second pass 70.3 s, versus 80.2 s for the first;
- a new run directory;
- bundle and samples reused (`fetched` 0, `reused_from_cache` 327);
- metrics unchanged;
- the isolated environment rebuilt from scratch in a new directory, so a second 516 MB environment sits beside the first.

**Partial re-run:** re-running only the Section 1 cell, then the Section 4 cell, failed with `RuntimeError: Stage 'prepare' failed (exit 2): see the log above`, after `python: can't open file '…/<new run dir>/tutorial_stages.py'` (MPF-m5).

## 5. Findings

### Minor

#### MPF-m1 — The opening cell says no hosted run exists, and setup/disk statements are unmeasured

- **Cell/section:** cell 0 (**Run all** paragraph), cell 1 (*Running it*), cell 4 (Prerequisites, disk). Generator: `tools/notebook_template.py`, opening-cell text.
- **Observed issue:**
  - Cell 0 ends with "In the one local CPU run recorded for this revision … the model stages took under two minutes after the environment was built; a hosted Colab run has not yet been recorded."
  - `docs/release-verification.md`, `STATUS.md`, `README.md` and `MODEL_CARD.md` all record the 2026-10-04 Colab run of this exact blob.
  - Cell 1 says the isolated install "takes the longest" and cell 11 says "a minute or two". The Colab build took 13 s and P1's 7 s.
  - Cell 4 asks for "about 1.5 GB for the isolated environment" (the Section 1 disk check enforces it). P1's environment was 0.52 GB, plus 0.11 GB of managed CPython.
- **Consequence:** the first thing a learner reads about runtime evidence is false, and the time and disk figures are unlabelled estimates, not measurements.
- **Evidence:**
  - Documented: the Colab record, with `setup_seconds` 13 and 89 s in stages.
  - Direct (P1): cell 12 took 7.4 s; `du` of the environment gave 516 MB + 106 MB.
- **Recommended correction:**
  - Quote the recorded Colab run in cell 0: date, runtime, 13 s environment build, 89 s of stages, revision.
  - Label the remaining figures as estimates, or replace them with measurements. Keep a margin in the disk check if wanted, but say it is a margin.
  - Regenerate.
- **Acceptance check:** `grep -n "has not yet been recorded" tutorials/mediapipe_face_landmarker_colab.ipynb` returns nothing. Every time or disk figure in cells 0, 1, 4 and 11 either matches a run in `docs/release-verification.md` (with its environment) or is labelled an estimate.
- **Spec:** SRC3, UX12.

#### MPF-m2 — The IMAGE-versus-VIDEO comparison is scored against an IMAGE-mode reference

- **Cell/section:** Section 9 (cells 32–34); `stage_newdata` VIDEO block in `tools/tutorial_stages.py`.
- **Observed issue:**
  - **The metric.** The "consistency NME" of each mode is the distance from the IMAGE-mode still prediction moved by the known roll. IMAGE mode is therefore compared with itself: frame 0 scores exactly 0.0 for IMAGE. The cell's question asks which mode "will be closer to the known motion".
  - **The explanation.** The "What to notice" note attributes VIDEO's larger value to tracking lag. Nothing in the stage tests that.
  - **The jitter column.** `frame_to_frame_change` is the mean absolute change of the *error* series, not of landmark positions, and is not defined for the learner.
  - **The annotation is unused.** The composite has a 189-point annotation, which the stage loads for its NME but does not use here.
- **Consequence:** the learner reads a 3× gap (0.004 vs 0.0119) as a property of the running modes, when part of it is built into the reference. They are also handed a causal story the evidence does not test. Objective 6 ("compare IMAGE and VIDEO running modes") rests on this comparison.
- **Evidence:**
  - Direct (P3). Against the composite's annotation moved by the same roll: IMAGE 0.0105, VIDEO 0.0168. Against the notebook's reference: 0.004 and 0.0119.
  - The still prediction itself is 0.0103 from the annotation.
  - The direction holds; the gap shrinks from ~3× to ~1.6×.
- **Recommended correction:**
  - Score both modes against the annotation moved by the known roll, the true motion. Optionally keep the self-consistency number, labelled as such.
  - Define the jitter measure as frame-to-frame landmark displacement after removing the known motion.
  - Replace the lag explanation with what was measured. If lag is to be claimed, show error against angular velocity.
- **Acceptance check:** the `video_mode_demo` record reports each mode's error against the annotation moved by the known motion. The Section 9 prose names that reference and makes no untested causal claim.
- **Spec:** EVAL3, UX4.

#### MPF-m3 — The provenance record does not bind the output files it describes

- **Cell/section:** Section 10 (cells 35–37); `stage_export`.
- **Observed issue:**
  - `mediapipe_face_landmarker_result.json` records model, runtime, configuration, samples and every summary. It does not record:
    - the output inventory: the listing is printed with 16-hex digest prefixes only, and not saved;
    - a run identifier, although the run directory has one;
    - stage timings;
    - which optional branches ran.
  - No CSV carries a run id or bundle id.
  - Cell 37 says "each file is traceable to the image, face and landmark ids it describes and to the exact bundle that produced it". That holds only while the files stay in the same directory as `result.json`.
- **Consequence:** once a learner downloads `newdata_landmarks.csv` on its own, nothing ties it to the bundle, revision or run that produced it. Two runs' files cannot be told apart.
- **Evidence:** direct (P6). `result.json` keys: `blendshape_sanity`, `evaluation`, `evidence_label`, `inference_config`, `model`, `new_data`, `notebook_source`, `robustness`, `runtime`, `samples`. There is no `files`, `run_id` or timing key.
- **Recommended correction:**
  - Add `run_id` (the run directory name) and per-stage seconds to `result.json`.
  - Write a `files` inventory: relative path, bytes and full SHA-256 for every output except `result.json` itself.
  - Optionally add a `run_id` column to the CSVs.
- **Acceptance check:** after a default run, `result.json` holds `run_id` and a `files` list whose digests match `sha256sum` of each listed file.
- **Spec:** OUT3, OUT7, OUT9.

#### MPF-m4 — BYOD zip: one stray non-image member refuses the archive, under a renamed file name

- **Cell/section:** Section 11 (cells 38–40); `_byod_inputs`, `stage_byod`.
- **Observed issue:**
  - Zip members are extracted as `NN_<basename>`.
  - Hidden files and `__MACOSX` are skipped, but any other non-image member stops the whole run. The message names the renamed copy (`01_README.txt`), not the archive member (`README.txt`, or `a/face.jpg` for nested paths).
  - Section 11 says "hidden files ignored" but not that every other file must be an image.
  - Extracted copies of the user's images stay in `outputs/byod/` beside the results. This is mentioned only generically ("Delete them with the file browser").
- **Consequence:** a typical phone or desktop zip with a stray `.txt` or `.json` fails with a name the user never created. The user's face images are duplicated into the results folder they are most likely to download or share.
- **Evidence:** direct (P4): `ValueError: 01_README.txt: not a decodable image (UnidentifiedImageError …); supply a JPEG or PNG photograph`. Listing after a three-image zip: `00_f0.jpg`, `01_f1.jpg` and `02_f2.jpg` beside the CSVs.
- **Recommended correction:**
  - Either skip non-image members by extension and report them, or keep the refusal but name the archive member and say "a zip may contain only images".
  - Extract to a scratch directory outside `outputs/`.
  - State both rules in Section 11.
- **Acceptance check:** a zip with one image and one `README.txt` either processes the image and reports `README.txt` as skipped, or refuses naming `README.txt`. After any BYOD run, `outputs/byod/` holds only results.
- **Spec:** UX10, DAT12, DAT19.

#### MPF-m5 — Re-running the Section 1 cell alone strands every later cell with an unguided error

- **Cell/section:** cell 6 (Section 1) creates `ROOT` and `ENV_ROOT` from a fresh `uuid4`; cells 9 and 12 write the carried files and the environment under those names; Troubleshooting (cell 44). Generator: `tools/notebook_template.py`, runtime cell and troubleshooting table.
- **Observed issue:**
  - Re-running only the Section 1 cell, a natural "start over" or "check the disk again" action, points `ROOT` at a new, empty run directory.
  - `PYTHON` still names the earlier environment.
  - The next learner cell fails: the interpreter cannot open `<new ROOT>/tutorial_stages.py`, and the kernel raises `Stage 'prepare' failed (exit 2): see the log above`. No stage error file exists.
  - No Troubleshooting row covers it.
  - Separately, every full Run all builds a new isolated environment (≈ 0.5 GB) in a new temporary directory with its own `uv` cache, so repeated runs accumulate copies.
- **Consequence:** a learner who re-runs the first cell meets a raw file-not-found message that does not say "run Sections 1–3 again". Repeated Run alls slowly consume disk that the Section 1 check then counts against them.
- **Evidence:** direct (P5), shown in §4: the partial re-run error text, and the second Run all's new 516 MB environment beside the first.
- **Recommended correction:**
  - Make `run_stage` check that `ROOT/tutorial_stages.py` and `PYTHON` exist, and raise "Section 1 was re-run, which starts a new run directory: run Sections 2 and 3 again (or Run all)".
  - Add that row to Troubleshooting.
  - Optionally key the environment directory on the lock digest rather than the run id, so a repeated Run all reuses a verified environment.
- **Acceptance check:** after a default Run all, re-running cell 6 and then cell 18 stops with a message naming the cells to re-run, and the same text appears in the Troubleshooting table.
- **Spec:** SRC2, UX10, GDL13.

### Suggestions

- **MPF-S1** — Show the 180° case in the robustness figure. P2 confirms the mechanism the checkpoint describes: on 18/18 detected inverted faces the mesh was fitted upright, so its "mouth" sits on the subject's eyes. That is the notebook's most instructive failure, and the learner is asked to reason about it without seeing it.
- **MPF-S2** — The CED plot's x-axis stops at 0.25, but the image-mean baseline reaches 0.52; about 12 % of its curve is off-plot without a note. Extend the axis or annotate the clipped share.
- **MPF-S3** — State the units of the transformation matrix's translation column (MediaPipe's canonical face geometry). The learner sees `-38.311` with no unit.
- **MPF-S4** — The composites are built from London Set individuals (webmorphR.stim `README.md` at `fa8b78f`: "4 individuals per composite … from the London set above"). Say "built from the same people as Section 6" rather than "the same population".
- **MPF-S5** — "The same normalisation as the upstream model card": the upstream card divides by a **3D** IOD from ground truth; the notebook uses the 2D annotated IOD with the same eye-centre definition. Say so; on frontal studio faces the difference is small.
- **MPF-S6** — The activity's sample answer could mention what P2 found: failures start between 90° and 100° (3/20 at 100°), and the response is asymmetric (−150°: detection 0.4; +150°: 0.85).

## 6. Readiness

**Needs revision.** There are no Majors.

- MPF-m1 breaches SRC3, a MUST: a knowingly stale statement in the opening cell. It and the other Minors are small generator edits.
- After the fixes, the remaining gates are:
  - a one-pass hosted Run all of the regenerated blob;
  - the REL12 BYOD journey on a hosted runtime: one compatible and one incompatible input, including the Colab upload dialog, recorded in `docs/release-verification.md`.
- At that point the notebook moves to **Verification pending**, then Release-grade on a human promotion.
- The two `STATUS.md` items (DIMER acceptance of the `.task` format; the upstream publication date) are maintainer decisions outside the notebook.

## 7. Verified versus inferred

- **Verified by direct execution (CPU container, labelled above):**
  - the default path, 13/13, identical to Colab;
  - the 180° mechanism and the rotation sweep;
  - the VIDEO comparison against the annotation;
  - every BYOD acceptance and refusal listed;
  - the BYOD output replacement;
  - the activity range check;
  - the contents of `result.json`;
  - the re-run behaviour in §4.
- **Verified from documented evidence:** the Colab one-pass run of this blob; the local pre-flight's BYOD and activity runs.
- **Inferred from source:** the Colab upload-dialog path (`files.upload()`, single file, replaced per upload). It is not executed here and is part of the REL12 gate.
- **Only Kurt can confirm:**
  - whether the 1.5 GB disk requirement is a deliberate margin (MPF-m1);
  - whether BYOD zips should skip non-image members or refuse them (MPF-m4).
- **Most likely to be wrong:** MPF-m2's framing. A maintainer could argue that self-consistency against the still prediction is a legitimate stability measure. It is rated a finding because the prompt asks about "the known motion" and the note draws a causal conclusion from it.
