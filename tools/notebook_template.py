"""Per-repository template for tools/build_notebook.py /3.0-cpu (NOTEBOOK_SPEC 2.2 §4 standalone, §25.13 isolated environment).

The generator writes the infrastructure cells (runtime check, carrier, isolated install + stage runner, bundle staging)
from repository files; this template holds the learner-facing prose, the list of carried files and the learner cells.
Every learner cell calls ``run_stage(...)``: the carried ``tutorial_stages.py`` (``tools/`` in the repository) runs one
stage per process in an isolated, hash-locked environment, so nothing is installed into the notebook kernel.

This template configures a TASK-INFERENCE workflow for the MediaPipe Face Landmarker task bundle: the pinned bundle is
staged and digest-verified, 214 digest-pinned public face photographs are fetched and validated, the task runs on one
photograph with the full output contract, landmark accuracy is measured against the dataset authors' annotations of
102 faces beside two mean-shape baselines, robustness is measured under controlled perturbations, blendshape scores
are sanity-checked on paired neutral and smiling photographs, new images and a short frame sequence (VIDEO mode) are
processed and exported, and one provenance record is written. BYOD and a change-one-thing activity are optional.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "mediapipe-face-landmarker-pipeline"

BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/mediapipe_face_landmarker_colab.ipynb",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-google--ai--edge%2Fmediapipe-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/google-ai-edge/mediapipe",
    ),
    (
        "Model card",
        "https://img.shields.io/badge/Upstream%20model%20card-Face%20Mesh%20V2-4285F4?style=flat",
        "https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf",
    ),
    ("License", "https://img.shields.io/badge/License-Apache--2.0-blue.svg", "https://www.apache.org/licenses/LICENSE-2.0"),
]

ATTRIBUTION = (
    "*Photographs: Face Research Lab London Set, DeBruine & Jones (2017), CC BY 4.0; composite faces: DeBruine (2016), "
    "CC BY 4.0. The overlays drawn by this notebook are adaptations of those images.*"
)

TEMPLATE = {
    "package": "mediapipe_face_landmarker_pipeline",
    "repo_name": REPO,
    "stem": "mediapipe_face_landmarker",
    "notebook_name": "mediapipe_face_landmarker_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh Linux x86_64 runtime — a **CPU** runtime is enough; no GPU is used — builds an isolated "
        "Python environment from the carried hash-locked requirements (`mediapipe`, `numpy`, `pillow` and their dependencies) "
        "without touching the notebook kernel's own packages, then runs each stage below in its own process: it stages and "
        "digest-verifies the pinned 3.76 MB MediaPipe Face Landmarker task bundle from Google Cloud Storage, fetches 214 "
        "public face photographs and 112 annotation files (about 103 MB, every file pinned by size and SHA-256), validates every image and "
        "demonstrates the refusals, runs the task on one photograph and writes the full output contract, measures landmark "
        "accuracy on 102 annotated faces against two mean-shape baselines, measures robustness under rotation, downscaling, "
        "blur, brightness and JPEG compression, sanity-checks the blendshape scores on 102 neutral/smiling pairs, processes "
        "ten new composite faces and a 30-frame sequence in VIDEO mode, and writes one provenance record. The default path "
        "needs no repository clone, no DIMER worker or service, no credential, no upload dialog, no configuration edit and "
        "no runtime restart (NOTEBOOK_SPEC 2.2 §5). The recorded hosted run (Google Colab, 4 October 2026, a 2-vCPU x86_64 runtime with "
        "the default settings, notebook generated from revision `e49deff`, an earlier revision with the same stages and lock) built the "
        "isolated environment in 13 s and spent 89 s in the model stages; see `docs/release-verification.md`. A slower network or "
        "CPU takes longer. A second **Run all** in the same runtime reuses that environment instead of building another."
    ),
    "byod": (
        "After the canonical path completes, set `USE_BYOD = True` in Section 11 and run that cell to process your own image, "
        "or a zip of up to 20 images, through the same validation, inference and output contract (landmarks CSV, blendshapes "
        "CSV, transformation matrices JSON, overlays and a result record). Set `BYOD_PATH` to a file already in the runtime "
        "to skip the upload dialog, and `BYOD_NUM_FACES` (1..10) for group photographs. **Faces are personal data, and "
        "landmarks and blendshapes derived from them can be biometric data:** upload only images you are authorised to "
        "process in this runtime. Uploaded files and every output stay inside this runtime; nothing is sent to any service. "
        "BYOD is optional and never part of the default path."
    ),
    "weights_key": "mediapipe-face-landmarker-float16-v1",
    # Carried byte for byte (UTF-8 text, LF newlines) into the run directory and verified against CARRIED_HASHES.
    "carried": {
        "src/mediapipe_face_landmarker_pipeline/__init__.py": "src/mediapipe_face_landmarker_pipeline/__init__.py",
        "src/mediapipe_face_landmarker_pipeline/pipeline.py": "src/mediapipe_face_landmarker_pipeline/pipeline.py",
        "src/mediapipe_face_landmarker_pipeline/samples.py": "src/mediapipe_face_landmarker_pipeline/samples.py",
        "src/mediapipe_face_landmarker_pipeline/metrics.py": "src/mediapipe_face_landmarker_pipeline/metrics.py",
        "src/mediapipe_face_landmarker_pipeline/sample_manifest.json": "src/mediapipe_face_landmarker_pipeline/sample_manifest.json",
        "tutorial_stages.py": "tools/tutorial_stages.py",
        "requirements.txt": "tutorials/requirements-colab.lock.txt",
        "weights/mediapipe-face-landmarker-float16-v1/dimer-base-manifest.json": "weights/mediapipe-face-landmarker-float16-v1/dimer-base-manifest.json",
        "LICENSE": "LICENSE",
    },
    "stage_runner": "tutorial_stages.py",
    "lock": "requirements.txt",
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "disk_gib": {"weights": 0.2, "environment": 1.5},
    "runtime_modules": ["mediapipe", "numpy", "PIL"],
    "title": "MediaPipe Face Landmarker — DIMER guided task-inference notebook (standalone)",
    "badges": BADGES,
    "capability": "478-point 3D facial landmark estimation, 52 blendshape scores and a 4 × 4 facial transformation matrix per face from a still image or a frame sequence, with landmark evaluation against annotated faces, robustness and blendshape sanity checks, and a CSV/JSON output contract",
    "intro": (
        "The MediaPipe Face Landmarker (Google, 2023) is a fixed, pretrained pipeline of three small networks shipped together "
        "as one *task bundle*: a BlazeFace short-range **face detector** (Bazarevsky et al., 2019) finds each face and its "
        "rotation; a **face-mesh network** (Kartynnik et al., 2019; Grishchenko et al., 2020) predicts 478 landmarks — 468 points on the facial surface "
        "plus 10 iris points — in normalised image coordinates with a relative depth `z`; a **blendshape network** turns 146 of "
        "those landmarks into 52 scores in [0, 1] (`_neutral` plus 51 expression coefficients such as `jawOpen` or "
        "`mouthSmileLeft`); and the canonical-face geometry in the bundle yields a **facial transformation matrix** from a "
        "canonical 3D face to the camera. The `mediapipe` Tasks API (Lugaresi et al., 2019) runs it on the CPU.\n\n"
        "Three facts shape everything this notebook measures. **Nothing is trained here:** the bundle is a fixed edge model, "
        "so this is a task-inference notebook — it evaluates and uses the model, it does not adapt it. **Landmarks have "
        "ground truth only where someone annotated it:** the dataset authors placed 189 points on each of 102 photographs, and "
        "14 of them have an unambiguous MediaPipe counterpart; the notebook measures those 14, normalised by the distance "
        "between the eyes, beside two baselines that use no landmark network. **Blendshape scores have no ground truth "
        "here:** they drive an avatar rig and are not calibrated probabilities, so the notebook only checks that they move "
        "in the expected direction between a neutral and a smiling photograph of the same person — a sanity check, not an "
        "evaluation.\n\n"
        "**This model does not recognise or identify people.** It outputs geometry and expression scores, not an identity, and "
        "the notebook never compares faces with each other. Faces are nevertheless personal data, and landmarks and blendshapes "
        "derived from them can be biometric data; the sample faces here were published under CC BY 4.0 by researchers whose "
        "participants gave signed consent for research and illustration use."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to —\n\n"
        "1. **Explain** the Input → Model → Output contract of the Face Landmarker: what a landmark, a blendshape score and a "
        "facial transformation matrix are, and which of the three networks in the bundle produces each (Sections 3, 5).\n"
        "2. **Diagnose** an invalid input from a validation refusal, and **identify** what validation can and cannot check "
        "about a face image (Section 4).\n"
        "3. **Interpret** a normalised mean error (NME) against annotated landmarks, and **compare** the model with two "
        "mean-shape baselines and across self-reported groups without over-reading small groups (Section 6).\n"
        "4. **Distinguish** a detection from a correct localisation, using consistency and annotation errors under "
        "controlled perturbations (Section 7).\n"
        "5. **Explain** why blendshape scores are not probabilities, and **apply** a paired sanity check whose expected "
        "direction is stated before it runs (Section 8).\n"
        "6. **Apply** the same output contract to new images and a frame sequence, and **compare** IMAGE and VIDEO running "
        "modes (Section 9).\n"
        "7. **Predict**, run and **explain** the effect of one change — the rotation angle — in an optional activity (Section 12).\n"
        "8. **Write** an evidence-based conclusion that names the baseline, the main failure mode and the limits of 102 "
        "studio photographs (Conclusion)."
    ),
    "exclusions": (
        "face recognition, identification, verification or any comparison between faces (the model does none of these); "
        "emotion, age, gender or ethnicity inference (the self-reported attributes of the sample are used only to break "
        "down the error); fine-tuning or adaptation (the bundle is a fixed edge model); GPU or other delegates; the "
        "`live stream` running mode; hand, pose or full-body landmarks; and any claim that a blendshape score is a "
        "calibrated probability that an expression is present."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh **Linux x86_64** runtime — a Google Colab CPU runtime is enough, and so is Kaggle or a Linux Jupyter kernel. The kernel's own Python version does not matter: the notebook installs nothing into it, and runs every stage with CPython 3.12.12 in an isolated environment built from {n_locked} hash-locked packages (`mediapipe` 1.0.0, `numpy` 2.5.3, `pillow` 12.3.0). Inference runs on the CPU through TFLite's XNNPACK delegate in the bundle's float16 weights; no GPU or other accelerator is used, and a GPU runtime gives no speed-up here. About 0.2 GB of disk is needed for the bundle and the sample faces. The isolated environment measured 0.52 GB plus 0.11 GB of managed CPython in the review's local run; the Section 1 check asks for 1.5 GB, which includes a margin for `uv`'s download cache.",
        "- **Knowledge:** what an image coordinate is (x to the right, y down, in pixels or as a fraction of the width and height), what a mean and a median are, and how to read a Python dictionary printed by a cell. No prior experience with face landmarks is assumed: each term is explained where it is first needed, and the glossary collects them.",
        "- **Model file:** one MediaPipe task bundle, `face_landmarker.task` (a stored zip of three TFLite flatbuffers and one binary protobuf). TFLite flatbuffers are data read by the TFLite interpreter; nothing is unpickled and no code is downloaded with the model. The bundle is released under Apache-2.0.",
        "- **Data contract:** an image Pillow can decode, 64..8192 px on each side, at most 40,000,000 pixels and 50 MB; it is converted to 8-bit RGB (any conversion is reported) and an EXIF orientation is applied (reported). `num_faces` is 1..10. Validation is structural: it cannot tell whether an image contains a face — the detector decides, and zero faces is a valid result.",
        "- **Sample data:** the 102 neutral and 102 smiling front-facing photographs of the Face Research Lab London Set (DeBruine & Jones, 2017; 1350 × 1350 px; CC BY 4.0) with the authors' 189-point annotation templates and the participants' self-reported age, gender and ethnicity, and ten composite faces (DeBruine, 2016; CC BY 4.0), each the average of four London Set faces. All participants gave signed consent for their images to be used in lab-based and web-based studies and to illustrate research. The files are read from the public GitHub repository `debruine/webmorphR.stim` at one immutable commit, and each is pinned by size and SHA-256.",
        "- **Privacy:** Faces are personal data and the landmarks and blendshapes derived from them can be biometric data. Do not upload confidential, restricted, sensitive or personal images — including photographs of people who have not agreed to this processing — unless you are authorised to process them in this runtime. Nothing in this notebook sends an image or a result to any service; the default path uploads nothing.",
    ],
    "guided": {
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who can open a hosted notebook (Google Colab or Jupyter), run cells in "
                "order and read short Python, and who want to see how a pretrained face-landmark model is evaluated honestly and "
                "used safely. No prior experience with face landmarks is assumed: each term is explained where it is first "
                "needed, and the glossary below collects them. No GPU is needed — a CPU runtime is enough. The "
                "**Prerequisites** give the details.\n\n"
                "**Running it.** In Colab, keep the default CPU runtime and choose *Runtime → Run all*. The default path needs no "
                "edit, no upload, no account, no token and no runtime restart. Section 2 builds an isolated environment from "
                "hash-locked packages (13 s in the recorded Colab run; reused by a later Run all in the same runtime); the "
                "model stages that follow took between 0.2 s and 37 s each in that run, 89 s in all. You can also run the "
                "notebook one cell at a time with *Shift + Enter*. Re-running the Section 1 cell on its own is safe: it keeps "
                "this session's run directory, so the cells after it keep working.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell "
                "calls `run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated "
                "environment's Python, streams what it prints, and stops the notebook with the stage's own error message if it "
                "fails. Stages hand results to each other only through files in the run directory — the verified bundle, the "
                "verified sample faces and JSON records. MediaPipe's own informational log lines (for example *Created "
                "TensorFlow Lite XNNPACK delegate for CPU*) go to `logs/<stage>.native.log` in the run directory instead of the "
                "cell output.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–12) are the machine-learning workflow; each runs one stage "
                "and prints compact dictionaries for you to read. *Infrastructure cells* (Sections 1–3: the runtime check, the "
                "carried code, the isolated install and the pinned-bundle staging) are collapsed and titled **Infrastructure**. "
                "You may run them without studying their implementation: they exist for reproducibility and provenance, not as "
                "prerequisite machine-learning knowledge. Open one with *Show code* if you are curious.\n\n"
                "**Form controls.** Two learner cells start with fields that Colab renders as a form: `USE_BYOD`, `BYOD_PATH` and "
                "`BYOD_NUM_FACES` (Section 11), and `RUN_ACTIVITY` and `ACTIVITY_ROTATION` in the optional activity (Section 12). "
                "The Section 1 infrastructure cell has one more, `NEW_RUN_DIRECTORY`, off by default. "
                "Leave them at their defaults for the first run: the notes and sample answers describe the default path.\n\n"
                "**Section tags.** Each numbered heading carries one tag. **[Concept]** — what the model does and why. "
                "**[Evaluation practice]** — how the evidence is produced and how to read it. **[Engineering]** — "
                "reproducibility, provenance and packaging.\n\n"
                "**Predict, then check.** Before each principal result a **Predict before running** prompt asks you to commit to "
                "an expectation; after it, **What to notice** describes normal output and a collapsed **Check your reasoning** "
                "answer follows each checkpoint. Write your own answer first, then open it. Exact numbers can vary slightly "
                "between CPUs and library builds, so the notes describe the shape of a normal result rather than fixed values."
            ),
            (
                "## The task: Input → Model/System → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Detect** | an RGB image | BlazeFace short-range face detector (`face_detector.tflite`) | a box and a rotation per face, up to `num_faces` |\n"
                "| **Landmarks** | the face region, rotated upright | face-mesh network (`face_landmarks_detector.tflite`) | 478 landmarks: `x`, `y` as fractions of the image width and height, `z` a relative depth on the same scale as `x` |\n"
                "| **Blendshapes** | 146 of the landmarks | blendshape network (`face_blendshapes.tflite`) | 52 named scores in [0, 1] (`_neutral` + 51 coefficients) |\n"
                "| **Pose** | the landmarks | canonical-face geometry (`geometry_pipeline_metadata_landmarks.binarypb`) | a 4 × 4 facial transformation matrix (canonical face → camera) |\n"
                "| **Evaluation** | 102 annotated photographs | 14 landmarks compared with the authors' annotation, beside two mean-shape baselines | NME per face, detection rate, failure rate, group breakdown |\n\n"
                "Every output row carries an `image_id` and a `face_id`, and every landmark row a `landmark_id` (0..477), so a "
                "value can always be traced to the image, face and mesh point it came from.\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1. Check the runtime | [Engineering] | Linux x86_64 and disk checked; a fresh run directory | the machine and directories |\n"
                "| 2. Carry the code, install the runtime | [Engineering] | carried files verified; an isolated hash-locked environment | versions |\n"
                "| 3. Pin, stage and verify the bundle | [Engineering] | the task bundle downloaded and digest-checked, member by member | the four members |\n"
                "| 4. Sample faces and validation | [Evaluation practice] | 214 photographs fetched and validated; four refusals | counts, refusals |\n"
                "| 5. Run the task on one photograph | [Concept] | landmarks, blendshapes, matrix; `num_faces`; a no-face image | the output contract |\n"
                "| 6. Landmark accuracy | [Evaluation practice] | NME on 102 annotated faces vs two baselines; groups | the principal result |\n"
                "| 7. Robustness | [Evaluation practice] | rotation, scale, blur, brightness, JPEG on 20 faces | where it breaks |\n"
                "| 8. Blendshape sanity check | [Concept] | neutral vs smiling photographs of the same 102 people | the paired direction |\n"
                "| 9. New images and VIDEO mode | [Concept] | ten composite faces exported; a 30-frame sequence | new-data outputs |\n"
                "| 10. Export and provenance | [Engineering] | one result record; every file listed | the provenance |\n"
                "| 11. Bring your own data (optional) | [Engineering] | your image or zip through the same contract (off by default) | your outputs |\n"
                "| 12. Optional activity | [Concept] | change the rotation angle (off by default) | your comparison |\n"
                "| Troubleshooting | [Engineering] | common hosted-runtime failures | when something fails |\n"
                "| Interpretation and conclusion | [Evaluation practice] | limits and an evidence-based conclusion | your conclusion |\n\n"
                "**Fast path.** Short on time? Run all, then read Sections 5, 6 and 8 and the conclusion: they carry the "
                "principal results. The canonical path ends with Section 10; Sections 11 and 12 change nothing unless you switch "
                "them on."
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **Landmark** | A named point on the face, such as the inner corner of the left eye, given as image coordinates. |\n"
                "| **Face mesh** | MediaPipe's fixed set of 468 surface landmarks; point *k* is always the same place on every face. |\n"
                "| **Iris landmarks** | Points 468–477: five per iris; 468 and 473 are the iris centres. |\n"
                "| **Normalised coordinates** | `x` and `y` as fractions of the image width and height (0..1); multiply by the width or height for pixels. |\n"
                "| **`z` (relative depth)** | Depth of a landmark relative to the face centre, on roughly the same scale as `x`; smaller is nearer the camera. Not a metric distance. |\n"
                "| **Subject's left / image left** | MediaPipe names points for the person in the photograph; in an unmirrored photo the subject's right eye is on the image left. |\n"
                "| **Blendshape** | A coefficient of a facial-animation rig (for example `jawOpen`); the 52 scores here drive an avatar, they are not probabilities. |\n"
                "| **Facial transformation matrix** | A 4 × 4 matrix mapping a canonical 3D face model into the camera's coordinate frame: head rotation and position. |\n"
                "| **Task bundle** | MediaPipe's `.task` file: a zip of the models and metadata one task needs. |\n"
                "| **Face detector** | The first network: finds faces and their rotation; the mesh network runs only on detected faces. |\n"
                "| **IMAGE / VIDEO mode** | IMAGE runs the detector on every image; VIDEO reuses the previous frame's landmarks to place the face region (tracking) and needs increasing timestamps. |\n"
                "| **Annotation template** | The dataset authors' 189 hand-placed points per photograph (`.tem` file); 14 are matched to mesh points. |\n"
                "| **Inter-ocular distance (IOD)** | Distance between the two eye centres, each the midpoint of that eye's corners, from the annotation. |\n"
                "| **NME** | Normalised mean error: the mean distance between predicted and annotated points divided by the IOD. 0.02 means 2 % of the eye distance. |\n"
                "| **Failure rate at 0.10** | The share of faces whose NME exceeds 0.10. |\n"
                "| **CED** | Cumulative error distribution: for each NME value, the share of faces at or below it. |\n"
                "| **Mean-shape baseline** | A prediction that ignores the image content: the average annotated shape, placed in the image (or in the detector's box). |\n"
                "| **Leave-one-out** | Each face's baseline is computed from the other faces only, so a face never predicts itself. |\n"
                "| **Detection rate** | The share of images in which the model returned a face. |\n"
                "| **Consistency NME** | The prediction on a perturbed image, mapped back to the original coordinates, compared with the prediction on the original. |\n"
                "| **Rank AUC** | The chance that a randomly chosen smiling photograph scores higher than a randomly chosen neutral one; 0.5 is chance. |\n"
                "| **Sign test** | A test of whether \"higher in the pair\" happens more often than a coin flip would give. |\n"
                "| **Biometric data** | Measurements of a person's body, such as face geometry, that can single the person out; protected in many jurisdictions. |\n"
                "| **Digest (SHA-256)** | A fingerprint of a file's bytes; a single changed byte changes it. |\n"
                "| **Hash-locked environment** | A separate Python environment built from a requirements file that pins every package to one version and one set of SHA-256 digests. |\n"
                "| **Stage** | One step of the workflow run as its own process by `run_stage`. |\n"
                "| **BYOD** | Bring Your Own Data: an optional switch to run the same contract on your own images. |\n\n"
                "</details>"
            ),
        ],
    },
    "setup": [
        {
            "cell": "check",
            "md": (
                "## 1. Check the runtime · [Engineering]\n\n"
                "> **Infrastructure.** The code cells in Sections 1–3 are collapsed. You may run them without studying their "
                "implementation; they exist for reproducibility and provenance. The learning activities start in Section 4.\n\n"
                "**Input:** a fresh hosted runtime. **System:** checks that it is Linux x86_64 with enough free disk, and creates "
                "a run directory. **Output:** the machine, the CPU count and the directories this run will use. Each new session "
                "writes to a new directory under `outputs/{stem}/`, so an earlier export cannot be mistaken for a current result. "
                "Running this cell again in the same session keeps that directory (the cells after it keep working); tick "
                "`NEW_RUN_DIRECTORY` for a fresh one, then run Sections 2 and 3 again. "
                "The verified bundle and sample faces are kept in `weights/` and reused by a later run."
            ),
            "after": (
                "**Expected result:** one dictionary naming the machine (`x86_64`), the number of CPUs, the kernel's Python "
                "version, the run directory, the weights directory, the isolated environment's directory and the free disk. If "
                "the cell stops with a platform or disk message, see **Troubleshooting**."
            ),
        },
        {
            "cell": "carrier",
            "md": (
                "## 2. Carry the code and install the locked runtime · [Engineering]\n\n"
                "> **Infrastructure.** The next two code cells are collapsed. The first **is** the repository's code, carried so "
                "that this notebook works on its own; the second builds the environment every stage runs in.\n\n"
                "The first cell holds, as text, the files the workflow needs: the package's four modules under "
                "`src/mediapipe_face_landmarker_pipeline/` (identity constants, bundle verification and staging, validation, the "
                "landmarker wrapper and output contract, the sample loader and the metrics), the pinned sample manifest, the stage "
                "runner `tutorial_stages.py`, the hash-locked `requirements.txt` ({n_locked} packages), the bundle manifest and the "
                "licence. It writes each file into the run directory and checks its SHA-256 against `CARRIED_HASHES`, stopping on "
                "any mismatch. The text is the repository's files byte for byte; the repository's parity test "
                "(`tests/test_notebook_parity.py`) fails whenever the two diverge, so what runs here is what the repository tests. "
                "Nothing in this cell runs a model."
            ),
            "after": (
                "**Expected result:** `carried_files`, `verified: True`, and the repository revision the notebook was generated "
                "from.\n\n"
                "The next cell installs nothing into this notebook's kernel. It downloads one pinned file — the `uv` installer "
                "wheel, refused unless its size and SHA-256 match — creates a separate virtual environment with its own CPython "
                "3.12.12, and installs `requirements.txt` into it with `--require-hashes --only-binary :all:`: every package must "
                "be the locked version, a prebuilt wheel, and match a locked digest. The hosted runtime's own packages — "
                "including its preinstalled NumPy and OpenCV, which `mediapipe` would otherwise replace — are never touched, "
                "which is why no restart is needed. The cell also defines `run_stage`, `load_record` and `show_image`, the three "
                "helpers the learner cells use."
            ),
        },
        {
            "cell": "install",
            "md": (
                "**Infrastructure: the isolated environment.** Installation messages from `uv` are normal; the build took 13 s "
                "in the recorded Colab run and can take longer on a slow network. If an environment built from the same lock "
                "already exists in this runtime (a second Run all), the cell reuses it and prints `environment_reused: True`. "
                "A failed download or a hash mismatch stops the cell; never remove a pin or a hash to get past one."
            ),
            "after": (
                "**Expected result:** one dictionary with the generating revision, the isolated environment's Python (3.12.12), "
                "the `mediapipe`, `numpy` and `PIL` versions, the number of locked packages, whether an existing environment was "
                "reused, and the setup time."
            ),
        },
        {
            "cell": "weights",
            "md": (
                "## 3. Pin, stage and verify the task bundle · [Engineering]\n\n"
                "> **Infrastructure.** The next code cell is collapsed. It downloads one 3.76 MB file and checks it; you may run it "
                "without studying its implementation.\n\n"
                "The model identity is carried twice — `MODEL_ID`, `MODEL_VERSION`, `MODEL_REVISION` and `MODEL_SHA256` in the "
                "carried `pipeline.py`, and the bundle manifest (size and SHA-256 of the bundle and of each of its four members) — "
                "and the `weights` stage first checks that they agree. Google publishes the bundle as `{MODEL_ID}`, version "
                "`{MODEL_VERSION}`. The object name alone is mutable, so the stage downloads the **object generation** "
                "`{MODEL_REVISION}`, an immutable identifier of exactly these bytes, writes it only after its size and SHA-256 "
                "match, then opens the zip **without extracting it**, refuses unsafe member paths and checks each member. "
                "There is no fallback to another source.\n\n"
                "The bundle holds four members: `face_detector.tflite` (BlazeFace short-range), `face_landmarks_detector.tflite` "
                "(the 478-point face mesh), `face_blendshapes.tflite` (the blendshape network) and "
                "`geometry_pipeline_metadata_landmarks.binarypb` (the canonical face used for the transformation matrix)."
            ),
            "after": (
                "**What to notice:** the model id, version, object generation, byte count and SHA-256; `fetched` lists the bundle "
                "on a first run and is empty on a rerun, because staging only fetches a missing file; then one line per member "
                "with `verified: True`. A size or SHA-256 mismatch stops the cell with a `ValueError` naming the file — see "
                "**Troubleshooting**, and never edit a manifest to get past one."
            ),
        },
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample faces and validation · [Evaluation practice]\n\n"
                "From here on, every code cell runs one stage of the carried runner with `run_stage`; its printed dictionaries "
                "appear under the cell. This cell runs the `prepare` stage. It fetches 327 files — 102 neutral and 102 smiling "
                "photographs of the same people from the Face Research Lab London Set, the authors' 189-point annotation for each "
                "neutral photograph, the participants' self-reported age, gender and ethnicity, and ten composite faces with their "
                "annotations — and refuses any file whose size or SHA-256 differs from the pinned value (`fetch_samples`). Then "
                "`validate_image` decodes every photograph and checks the input contract, and `read_template` checks that every "
                "annotation holds 189 finite points.\n\n"
                "**Validation** checks each input's structure before any model sees it; a **refusal probe** is a deliberately "
                "broken input used to show that the check works. The stage prints the operational ceilings first, so you know "
                "the limits before anything is checked.\n\n"
                "**Expected result:** the ceilings; 327 files with `fetched` (first run) or `reused_from_cache` (rerun); 214 "
                "validated photographs, all 1350 × 1350 RGB with no conversion; 112 annotated; four refusals — text bytes named "
                "`.jpg`, a 40 × 40 image, a 9000 × 100 image and `num_faces = 0` — each with a message naming the rule; and one "
                "greyscale image accepted **with a reported conversion**, because silently changing an input is not allowed but "
                "converting it openly is.\n\n" + ATTRIBUTION
            ),
            "code": "run_stage('prepare')",
        },
        {
            "md": (
                "**What to notice:** each refusal names the input, the rule and the corrective action; none of them reaches the "
                "model. `run_dir/outputs/{stem}_sample_manifest.csv` lists every photograph with its digest, size, any "
                "conversion and the self-reported attributes used later to break down the error.\n\n"
                "**Checkpoint:** the validator would accept a photograph of a blank wall. Why is that correct behaviour, and what "
                "decides whether there is a face?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Validation checks the **contract** — decodable, within the size and pixel ceilings, convertible to RGB — not the "
                "content. Whether an image contains a face is the detector's job, and \"no face found\" is a valid, reportable "
                "result (Section 5 shows it on a blank image), not an error. A validator that tried to judge content would "
                "either duplicate the model or refuse legitimate inputs. What validation buys is that a broken file, an "
                "enormous image or an impossible `num_faces` fails early with a message you can act on, instead of failing "
                "later inside the model or, worse, producing output from a silently altered image.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 5. Run the task on one photograph · [Concept]\n\n"
                "The `inference` stage builds a `FaceLandmarkerPipeline` from the verified bundle in **IMAGE** mode with "
                "`num_faces=1`, blendshapes and transformation matrices switched on, and the upstream default confidence of 0.5 "
                "for detection and presence. It runs one neutral photograph and writes the full output contract:\n\n"
                "- `{stem}_walkthrough_landmarks.csv` — one row per landmark: `image_id`, `face_id`, `landmark_id` (0..477), "
                "normalised `x`, `y`, `z`, and pixel `x_px`, `y_px`, `z_px` (`z_px` uses the image width as its scale);\n"
                "- `{stem}_walkthrough_blendshapes.csv` — one row per blendshape: `blendshape_index` (0..51), `blendshape` "
                "name and `score`;\n"
                "- `{stem}_walkthrough_matrices.json` — the 4 × 4 facial transformation matrix per face;\n"
                "- `{stem}_walkthrough_overlay.jpg` — the 478 landmarks drawn on the photograph, the 14 evaluated points ringed "
                "in orange.\n\n"
                "Then it asks for `num_faces=1` and `num_faces=2` on a collage of two composite faces, and runs a blank grey "
                "image.\n\n"
                "**Predict before running:** on the two-face collage with `num_faces=1`, how many faces will be returned — and "
                "on the blank image, will the stage raise an error or return something?"
            ),
            "code": (
                "run_stage('inference')\n"
                "show_image('{stem}_walkthrough_overlay.jpg', 'All 478 landmarks (white); the 14 evaluated points ringed in orange')\n"
                "show_image('{stem}_two_faces_overlay.jpg', 'num_faces=2 on a collage of two composite faces')"
            ),
        },
        {
            "md": (
                "**What to notice:** the input report, the configuration, one face, `landmarks_shape [478, 3]`, 52 blendshapes, a "
                "4 × 4 matrix, three example landmark rows, the five strongest blendshapes, and the matrix — its upper-left 3 × 3 "
                "block is close to the identity because the face looks straight at the camera, and its last column holds the "
                "translation. `faces_returned` shows 1, then 2 faces on the collage, and 0 on the blank image: `num_faces` is a "
                "maximum, and \"no face\" is an empty result, not an error. The inference time is for one CPU call.\n\n"
                "**Checkpoint:** a neutral face here still shows non-zero scores such as `browOuterUpLeft` around 0.6. Does that "
                "mean the person is raising an eyebrow with 60 % probability?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "No. A blendshape score is a coefficient for an avatar rig — how much of a predefined deformation best "
                "reproduces this face's landmark geometry — not the probability that an expression is present. Part of every "
                "score reflects the person's resting facial shape (brow height, eye shape, lip shape) rather than any movement, "
                "which is why a neutral face is not all zeros. The scores are not calibrated, the pipeline applies no threshold "
                "to them, and comparing a score across different people mixes anatomy with expression. Section 8 therefore "
                "compares the **same person** in two photographs instead of reading one score in isolation.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 6. Landmark accuracy against annotated faces · [Evaluation practice]\n\n"
                "**Question tested:** how close are the model's landmarks to the dataset authors' annotations, and how much of "
                "that comes from the landmark network rather than from simply knowing where faces usually are?\n\n"
                "The `evaluate` stage runs every one of the 102 neutral photographs once, with no selection. For each it takes the "
                "14 MediaPipe points that have an annotated counterpart (`LANDMARK_MAP`: iris centres, eye corners, the base of "
                "the nose, mouth corners, four lip midpoints and the bottom of the chin) and computes the **NME**: the mean "
                "distance to the annotation divided by the **inter-ocular distance** from the annotation — the same "
                "normalisation as the upstream model card. Two baselines use no landmark network:\n\n"
                "- **mean shape in the image** — the average annotated shape of the *other* 101 photographs, in pixels (these "
                "photographs are framed alike, so this is a stronger trivial baseline than it sounds);\n"
                "- **mean shape in the detector box** — the same average expressed relative to a face box and placed in the box "
                "that the bundle's own face detector finds in this photograph.\n\n"
                "The stage also breaks the error down by the participants' **self-reported** gender, ethnicity and age band. "
                "These groups are small; the breakdown shows where to look, not a fairness result.\n\n"
                "**Predict before running:** will the model's mean NME be nearer 0.02, 0.05 or 0.10? Which of the 14 points do "
                "you expect to disagree most with the human annotation, and why?"
            ),
            "code": (
                "run_stage('evaluate')\n"
                "show_image('{stem}_ced.png', 'Cumulative error distribution: the model and the two baselines')\n"
                "show_image('{stem}_evaluation_examples.jpg', 'Top: three typical faces; bottom: the three largest errors. Orange rings: model; cyan crosses: annotation')"
            ),
        },
        {
            "md": (
                "**What to notice:** the detection rate, the model's NME summary (mean, median, spread, 90th percentile, maximum, "
                "failure rate at 0.10 and a bootstrap interval for the mean over faces) beside the two baselines; the per-point "
                "errors, largest first; and the group tables with their sizes `n`. In the CED plot a curve further to the left "
                "is better. The per-image numbers are in `{stem}_evaluation_per_image.csv`.\n\n"
                + ATTRIBUTION
                + "\n\n**Checkpoint:** the largest per-point errors are usually at the bottom of the chin and the base of the nose, "
                "not at the eyes. Is the model worst there, or is something else going on? And how should you read a group whose "
                "mean NME is a few thousandths higher than another's?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Part of every \"error\" is a **definition difference**: the annotators' \"bottom centre of the chin\" and MediaPipe "
                "point 152 are placed by different conventions, and on a soft contour such as the chin or the nose base a "
                "consistent offset of a few pixels is likely. Points on sharp, well-defined structures (lip midlines, inner eye "
                "corners) agree best. So the per-point table measures agreement with *this* annotation, not absolute truth; "
                "human annotators also disagree with each other (the upstream model card reports 2.56 % IOD between its own "
                "annotators). For the groups: each mean comes from 1 to 69 faces of studio photographs, one photograph per person, "
                "with no repeated runs. A difference of a few thousandths between groups of 13 and 69 faces is within what "
                "person-to-person variation can produce, and it cannot be attributed to the attribute itself — lighting, "
                "contrast, facial hair, the annotators and the point definitions all differ between faces. What you may say is "
                "\"on this sample, group X had a mean NME of a, n = k\"; a fairness claim needs a larger, purpose-built "
                "evaluation set. Both baselines are much worse than the model, which shows that the network, not the framing of "
                "the photographs, produces the accuracy.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 7. Robustness under controlled perturbations · [Evaluation practice]\n\n"
                "**Question tested:** when the same face is rotated, made smaller, blurred, darkened or brightened, or compressed, "
                "does the model still find it — and are the landmarks still in the right place?\n\n"
                "The `robustness` stage takes the first 20 annotated faces and applies 15 settings with `perturb`, which also "
                "returns the exact geometric map from original to perturbed pixels. Each prediction is mapped back into the "
                "original coordinates and compared twice: with the prediction on the unperturbed photograph (**consistency "
                "NME** — does the output stay put?) and with the annotation (**annotation NME** — is it still right?). A face "
                "counts as a **failure** when it is detected but its annotation NME exceeds 0.10.\n\n"
                "**Predict before running:** which will hurt more — downscaling to 12 % of the width, or rotating by 90°? And "
                "what do you expect at 180° (upside down): no detection, or something else?"
            ),
            "code": (
                "run_stage('robustness')\n"
                "show_image('{stem}_robustness_examples.jpg', 'One face under four perturbations, with the predicted landmarks')"
            ),
        },
        {
            "md": (
                "**What to notice:** one row per setting with the detection rate, the failure rate, and the two NMEs. Mild "
                "settings change the consistency NME by a fraction of the model's own error; strong blur and strong downscaling "
                "raise both. Look closely at the 180° row: the detector still reports faces on most images, but every one of "
                "them is a failure.\n\n"
                "**Checkpoint:** why is a detection rate alone not enough to say that the model \"works\" on upside-down faces, and "
                "what would a deployment need to do about it?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "A detection only says that the detector found something face-like; it does not say the 478 points are on the "
                "right features. At 180° the detector fires, but the mesh is fitted with the wrong orientation, so the landmarks "
                "land on the wrong parts of the face and the NME is far above 0.10 — a confident-looking wrong answer. The model "
                "returns no quality flag that would reveal this. A deployment that may see unusual orientations therefore needs "
                "its own check: restrict the input (for example, upright selfie video, which is what the model was designed "
                "for), add a plausibility test on the output geometry, or measure the failure rate on its own data. Note also "
                "that the perturbations are synthetic: a real face that is far away, out of focus or badly lit differs from a "
                "resized, blurred or darkened studio photograph, so these rows are a sanity check of sensitivity, not a field "
                "evaluation.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 8. Blendshape sanity check on neutral and smiling photographs · [Concept]\n\n"
                "**Question tested:** do the blendshape scores move in the direction a smile should move them? Blendshape scores "
                "have no ground truth in this sample, so this is a **sanity check**, not an evaluation.\n\n"
                "The London Set photographed each of the 102 participants twice: with a neutral expression and smiling. The "
                "expression label comes from the dataset, not from the model. The `blendshapes` stage runs both photographs of "
                "every person and compares the **same person** across the pair, which removes most of the person-to-person "
                "differences in resting face shape. One expectation is stated before the stage runs: the mean of "
                "`mouthSmileLeft` and `mouthSmileRight` is higher on the smiling photograph. The stage reports how often that "
                "holds, the median paired difference, a **rank AUC** and a **sign test**. Five other scores — `cheekSquint`, "
                "`eyeSquint`, `eyeBlink`, `jawOpen` and `mouthClose` — are reported descriptively, with no expectation tested.\n\n"
                "**Predict before running:** in how many of the 102 pairs will the smile score rise? And will `eyeBlink` stay "
                "flat between the two photographs, given that every participant has their eyes open in both?"
            ),
            "code": (
                "run_stage('blendshapes')\n"
                "show_image('{stem}_smile_pairs.png', 'Smile score for each person, neutral vs smiling (grey: rose; red: fell)')"
            ),
        },
        {
            "md": (
                "**What to notice:** the stated expectation with the share of pairs where it held, the medians, the rank AUC "
                "(0.5 is chance) and the sign-test p-value; then the descriptive scores. The paired values are in "
                "`{stem}_blendshapes_paired.csv`.\n\n"
                "**Checkpoint:** `eyeBlink` usually rises on the smiling photographs although nobody blinked. Is that a model "
                "error? And does a rank AUC near 0.95 mean the smile score is a 95 %-accurate smile detector?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Not necessarily an error. A genuine smile raises the cheeks and narrows the eyes, so the eye opening really "
                "does shrink; a coefficient that measures eyelid closure from geometry will partly respond to it, and "
                "`eyeSquint` and `cheekSquint` rise for the same reason. That is why only one expectation was stated in advance "
                "and the others are described, not tested: looking at many scores and then picking the ones that moved would "
                "make chance findings look like confirmations. On the AUC: it says how well the smile score *ranks* smiling "
                "above neutral photographs in this studio set, where every smile is deliberate and the lighting is constant. "
                "It is not an accuracy, there is no threshold, and the scores are not probabilities; turning them into a "
                "smile/no-smile decision would need a threshold chosen and checked on data from the intended setting. The "
                "sign-test p-value only says that \"higher in the pair\" is very unlikely to be a coin flip on these 102 "
                "pairs.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 9. New images and VIDEO mode · [Concept]\n\n"
                "The `newdata` stage applies the same contract to images that played no part in Sections 6–8: ten composite "
                "faces, each the average of four London Set participants, so none of them is a real individual. It writes "
                "`{stem}_newdata_landmarks.csv`, `{stem}_newdata_blendshapes.csv`, `{stem}_newdata_matrices.json` and a contact "
                "sheet of overlays, and — because the composites carry annotations too — prints their NME.\n\n"
                "Then it compares the two running modes on a 30-frame sequence made from one composite: the face rolls between "
                "−8° and +8°, so the true motion of every landmark is known. **IMAGE** mode runs the detector on every frame; "
                "**VIDEO** mode passes increasing timestamps and reuses the previous frame's landmarks to place the face region "
                "(tracking), which is how the model is meant to run on a camera stream. Both modes are scored against the same "
                "reference: the composite's own annotation moved by the known rotation — the true motion — divided by the "
                "annotated inter-ocular distance.\n\n"
                "**Predict before running:** will VIDEO mode, which tracks the face from frame to frame, be closer to the known "
                "motion than IMAGE mode, or further from it?"
            ),
            "code": (
                "run_stage('newdata')\n"
                "show_image('{stem}_newdata_overlays.jpg', 'New images: ten composite faces with their landmarks')"
            ),
        },
        {
            "md": (
                "**What to notice:** one line per composite with its strongest blendshapes and its NME, a composite NME summary, "
                "the three written files, and the VIDEO-mode comparison. For each mode: `tracked` (frames with a face); "
                "`nme_vs_true_motion_mean`, the error against the moved annotation; `jitter_after_motion_removed`, how much each "
                "landmark's error changes from one frame to the next once the known motion is subtracted (0 would be a perfectly "
                "steady offset); and `self_consistency_nme_mean`, the distance from IMAGE mode's own still-image prediction moved "
                "by the rotation. The last one favours IMAGE mode by construction — IMAGE mode is compared with itself and scores "
                "exactly 0 on the unrotated frame — so use it only as a stability measure. `still_prediction_nme_vs_annotation` "
                "is how far the still prediction itself sits from the annotation: it sets the floor both modes start from.\n\n"
                "Read your own numbers for the direction and the size of the gap between the modes on `nme_vs_true_motion_mean`. "
                "This stage measures *that* the modes differ on this sequence, not *why*: it does not test tracking lag or any "
                "other cause, and VIDEO mode's purpose — speed and stability on real camera streams — is not what a 30-frame "
                "synthetic roll measures. Composites are smoother than real faces and were built from the same London Set people "
                "as Section 6, so their NME is a sanity check of the contract on new images, not evidence of generalisation.\n\n" + ATTRIBUTION
            ),
        },
        {
            "md": (
                "## 10. Export and provenance · [Engineering]\n\n"
                "The `export` stage writes `{stem}_result.json` in the run's `outputs/`: the model identity (id, version, object "
                "generation, SHA-256, download URL, the four member digests, licence, and that nothing was unpickled and no "
                "remote code ran), the notebook's source revision, the runtime versions, the inference configuration, the "
                "sample repository, commit and manifest digest, and every summary from Sections 6–9 labelled as tutorial "
                "evidence, the run id (the run directory's name), the seconds each stage took, which optional branches had "
                "already run, and a `files` inventory: the relative path, size and full SHA-256 of every file the run wrote to "
                "`outputs/` except the record itself. It then prints that inventory with shortened digests. No artifact is "
                "produced: the bundle is used as published and nothing in this notebook changes it.\n\n"
                "**Expected result:** the result path, the run id, the model id and revision, the runtime, and the file listing — "
                "CSV and JSON outputs, overlays and figures, and the per-stage JSON records."
            ),
            "code": "run_stage('export')",
        },
        {
            "md": (
                "**What to notice:** every number this notebook printed is in a machine-readable file. Inside a file, each row is "
                "traceable to the image, face and landmark ids it describes; `{stem}_result.json` then binds each file to this "
                "run and to the exact bundle by its full SHA-256, so a CSV downloaded on its own can still be matched to the run "
                "that produced it (`sha256sum` the file and look it up in `files`). This is the end of the canonical path."
            ),
        },
        {
            "md": (
                "## 11. Bring your own data (optional) · [Engineering]\n\n"
                "This branch is **off by default** and never part of *Run all*: with `USE_BYOD = False` the next cell only prints "
                "how to switch it on. When it is on, your input goes through the same contract as the samples — "
                "`validate_image`, the same `FaceLandmarkerPipeline` configuration (IMAGE mode, confidence 0.5) and the same "
                "writers.\n\n"
                "**What you can supply.** One image, or a zip of up to 20 images (flat or in folders; at most 200 MB; absolute "
                "paths, `..` and symlinks refuse the whole archive). Inside a zip, only members with an image extension (`.jpg`, "
                "`.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`, `.tif`, `.tiff`) are read; hidden files and `__MACOSX` are ignored, and "
                "any other member — a stray `README.txt` or `.json` — is skipped and listed under `skipped_non_image_members`. "
                "Every message names the member path as it appears in your zip. Members are extracted to a scratch folder "
                "(`byod_inputs/` in the run directory, outside `outputs/`), so `outputs/byod/` holds only results. Each image: "
                "any format Pillow decodes, 64..8192 px "
                "per side, at most 40,000,000 pixels and 50 MB; it is converted to RGB and an EXIF orientation applied, and both "
                "are reported. `BYOD_NUM_FACES` sets the maximum faces per image (1..10). A refusal stops the cell with a message "
                "naming the file and the rule.\n\n"
                "**How.** Set `USE_BYOD = True`. Either set `BYOD_PATH` to a file already in the runtime (no dialog opens), or "
                "leave it empty in Colab to get an upload dialog for one file; each upload replaces the previous one. Outputs go "
                "to `outputs/byod/` in the run directory: `byod_landmarks.csv`, `byod_blendshapes.csv`, `byod_matrices.json`, "
                "`byod_overlays.jpg` and `byod_result.json`.\n\n"
                "**Privacy.** Faces are personal data, and landmarks and blendshapes derived from them can be biometric data, "
                "which many jurisdictions protect specifically. Upload only images you are authorised to process in this "
                "runtime, preferably of people who have agreed to it. Do not upload confidential, restricted or regulated "
                "images. Your files and every output stay inside this runtime — `run_stage` runs locally and nothing is sent to "
                "any service — and they disappear when the runtime is recycled unless you download them. Delete them with the "
                "file browser when you are done."
            ),
            "code": (
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                'BYOD_PATH = \'\'  # @param {{type:"string"}}\n'
                'BYOD_NUM_FACES = 1  # @param {{type:"integer"}}\n\n'
                'if USE_BYOD:\n'
                '    if BYOD_PATH:\n'
                '        byod_path = Path(BYOD_PATH)\n'
                '    else:\n'
                '        from google.colab import files\n'
                "        upload_dir = ROOT / 'byod_upload'\n"
                '        shutil.rmtree(upload_dir, ignore_errors=True)\n'
                '        upload_dir.mkdir(parents=True)\n'
                '        uploaded = files.upload()\n'
                '        if len(uploaded) != 1:\n'
                "            raise ValueError('Upload exactly one file: an image, or a zip of up to 20 images.')\n"
                '        file_name, payload = next(iter(uploaded.items()))\n'
                '        byod_path = upload_dir / Path(file_name).name\n'
                '        byod_path.write_bytes(payload)\n'
                "    run_stage('byod', '--byod', byod_path.resolve(), '--num-faces', BYOD_NUM_FACES)\n"
                "    show_image('byod/byod_overlays.jpg', 'Your images with their landmarks')\n"
                'else:\n'
                "    print({{'byod': 'skipped (optional)', 'to_run': 'set USE_BYOD = True, and BYOD_PATH to an image or zip in the runtime (or leave it empty for the Colab upload dialog), then run this cell'}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the contract, one validation report per image (with any conversion), the faces found per "
                "image and their strongest blendshapes, the written files and `data_left_runtime: False`. Your images have no "
                "annotation, so there is no NME: what you can check is the overlay. Expect the model to work best on frontal, "
                "upright, well-lit faces that fill a reasonable part of the image — the setting of Sections 6–7."
            ),
        },
        {
            "md": (
                "## 12. Optional activity: change one thing — the rotation angle · [Concept]\n\n"
                "**Predict → Change one thing → Run → Observe → Explain.** This activity is off by default and changes nothing the "
                "canonical path produced: with `RUN_ACTIVITY = False` the next cell only prints how to switch it on. It runs the "
                "`activity` stage, which repeats the Section 7 rotation test on the same 20 faces at one angle of your choice and "
                "prints it beside the Section 7 rows.\n\n"
                "**Change one thing:** set `ACTIVITY_ROTATION` (degrees, −180..180; 120 by default) and `RUN_ACTIVITY = True`, "
                "then run the cell. Faces, bundle and settings stay fixed; only the angle changes.\n\n"
                "**Predict before running:** Section 7 showed reliable results at 90° and failures at 180°. At 120°, will the "
                "detection rate drop, the failure rate rise, or both? Where between 90° and 180° do you expect the change?"
            ),
            "code": (
                'RUN_ACTIVITY = False  # @param {{type:"boolean"}}\n'
                'ACTIVITY_ROTATION = 120  # @param {{type:"number"}}\n\n'
                'if RUN_ACTIVITY:\n'
                "    run_stage('activity', '--rotation', ACTIVITY_ROTATION)\n"
                'else:\n'
                "    print({{'activity': 'skipped (optional)', 'to_run': 'set RUN_ACTIVITY = True and ACTIVITY_ROTATION in -180..180, then run this cell'}})"
            ),
        },
        {
            "md": (
                "**Observe:** the unperturbed row and the Section 7 rotation rows, then your angle with its detection rate, "
                "failure rate and the two NMEs.\n\n"
                "**Explain:** did the result match your prediction? Try a second angle to bracket the change.\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "The detector estimates each face's rotation and the mesh network runs on an upright crop, so moderate rotations "
                "cost little. Somewhere beyond 90° the rotation estimate becomes unreliable and the mesh is fitted the wrong way "
                "up: detection often survives while the failure rate jumps, which is the pattern of Section 7's 180° row. The "
                "exact angle depends on the faces and on the bundle; with 20 faces each face is 5 % of a rate, so a change of one "
                "or two faces is within noise. Because only the angle changed, any difference is caused by it. The lesson "
                "transfers: a model can degrade by producing confident wrong output rather than by producing nothing, and only "
                "a check against ground truth or a plausibility test reveals it.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## Troubleshooting · [Engineering]\n\n"
                "| Symptom | Likely cause | What to do |\n"
                "|---|---|---|\n"
                "| Section 1 stops with `This notebook needs a Linux x86_64 runtime` | a local Windows or macOS kernel, or an ARM "
                "machine | Use Google Colab, Kaggle, or a Linux x86_64 Jupyter kernel: the locked environment is built for "
                "manylinux x86_64 wheels. |\n"
                "| Section 1 stops with `Not enough free disk` | the check asks for 1.5 GB for the environment (0.63 GB measured, plus a "
                "margin) and about 0.2 GB for the data | Start a fresh runtime; a `weights/` directory and an environment built "
                "from the same lock earlier in this runtime are reused and counted. |\n"
                "| `Carried file integrity failure` in Section 2 | a carried file was edited in the notebook | Do not edit the "
                "infrastructure cells; open a fresh copy of the notebook from the repository. |\n"
                "| `uv 0.12.15 wheel size/hash mismatch`, or a `URLError` / timeout while downloading it | a network failure or "
                "an unexpected response from PyPI | Re-run the Section 2 install cell. Never replace the pinned URL or digest. |\n"
                "| `CalledProcessError` from `uv venv` or `uv pip install` (a hash mismatch, `Failed to download`, HTTP 5xx) | a "
                "transient PyPI or network failure | Re-run the Section 2 install cell: `uv` reuses what it already downloaded. If "
                "a hash mismatch repeats, stop and report it — never remove `--require-hashes`, a pin or a hash. |\n"
                "| `RuntimeError: Stage '…' failed (exit 2): …` | the stage raised an error; the message after the colon is the "
                "stage's own error, its traceback is printed above it, and MediaPipe's native messages are in the run "
                "directory's `logs/<stage>.native.log` | Find the message in the rows below. A stage reads only files, so after "
                "fixing the cause you can re-run that cell and the cells after it. |\n"
                "| `… is missing: run the stage that writes it before …` | a learner cell was run before an earlier stage | Run the "
                "notebook from the top, or re-run the earlier cells in order. |\n"
                "| `The run directory … has no carried files, or the isolated environment is gone: run the three Infrastructure "
                "cells again in order (Sections 1, 2 and 3)` | Section 1 was run with `NEW_RUN_DIRECTORY` ticked (a fresh, empty run "
                "directory), or the runtime's temporary directory was cleared | Run Sections 1, 2 and 3 again in order, then the "
                "cell you wanted, or choose *Runtime → Run all*. Re-running the Section 1 cell on its own with the default setting "
                "keeps the run directory and needs nothing else. |\n"
                "| `the sample manifest changed since 'prepare'` | the carried files were regenerated mid-run | Re-run from Section 4. |\n"
                "| A download error in Section 3, or `downloaded face_landmarker.task: … refusing it` | a transient Cloud Storage "
                "failure or a partial download | Re-run the Section 3 cell; nothing is written until the bytes match. |\n"
                "| `ValueError: face_landmarker.task: size … != manifest …` or `sha256 … != manifest …` | a corrupted file in "
                "`weights/` | Delete `weights/mediapipe-face-landmarker-float16-v1/face_landmarker.task` and re-run the Section 3 "
                "cell. Never edit a manifest to get past a mismatch. |\n"
                "| A sample fetch fails in Section 4 (`URLError`, HTTP 429, or `sample …: fetched … pinned …; refusing it`) | a "
                "network failure, rate limiting by `raw.githubusercontent.com`, or a changed file | Wait a minute and re-run the "
                "Section 4 cell; verified files are kept in `weights/face-samples/` and only missing ones are fetched. |\n"
                "| `OSError: libEGL.so.1` or `libGLESv2.so.2: cannot open shared object file` in a stage | a different `mediapipe` "
                "build than the locked 1.0.0 (some releases link OpenGL ES libraries that minimal Linux images lack) | Keep the "
                "carried lock; do not change the `mediapipe` pin. |\n"
                "| `ModuleNotFoundError: No module named 'google.colab'` with `USE_BYOD = True` | the upload dialog needs Google "
                "Colab | Set `BYOD_PATH` to a file already in the runtime, or use Colab for the upload. |\n"
                "| `not a decodable image`, `each side must be within 64..8192 px`, `above the … ceiling` | a BYOD file outside the "
                "contract | Convert it to JPEG or PNG, or resize it, and try again. |\n"
                "| `BYOD zip has an unsafe member path`, `holds … image files` or `holds no image files` | an archive with absolute or "
                "`..` paths, symlinks, more than 20 images, or no member with an image extension | Re-create the zip with plain "
                "relative names and 1–20 images; other files in it are skipped and listed, not refused. |\n"
                "| BYOD returns 0 faces | the face is too small, strongly turned or rotated, partly hidden, or there is no face | "
                "Try a frontal, upright photograph in which the face fills a larger part of the image; zero faces is a valid "
                "result, not an error. |"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits · [Evaluation practice]\n\n"
        "The question this notebook can answer is narrow: on 102 front-facing studio photographs, how well do 14 of the Face "
        "Landmarker's 478 points agree with the dataset authors' annotations, how does that compare with knowing only where "
        "faces usually are, how sensitive is it to simple image changes, and do its smile scores move in the right direction "
        "between two photographs of the same person? Your run answers each part in Sections 6–8; read the values there rather "
        "than from this text.\n\n"
        "The numbers are tutorial / sanity evidence, not a benchmark. The 102 participants were photographed in one studio, "
        "facing the camera, under the same lighting; most are young adults, and 69 of 102 self-reported as white. Each "
        "measurement is one pass with no repeated runs, and only the mean NME carries an interval (a bootstrap over faces). "
        "14 points are not 478: the forehead, cheeks, jaw line and iris contours are not checked. Part of each per-point error "
        "is a difference between the annotators' and MediaPipe's point definitions. The group breakdown has groups of 1 to 69 "
        "faces and cannot support a fairness conclusion; the upstream model card reports its own geographic, gender and "
        "skin-tone evaluation on 1,700 images, which this notebook does not reproduce. The perturbations are synthetic, the "
        "VIDEO-mode sequence is a synthetic roll, and the blendshape check tests one stated direction on deliberate studio "
        "smiles. Public sample faces may have been seen by the upstream model during training; that cannot be ruled out. "
        "Blendshape scores are not calibrated probabilities, and the pipeline sets no threshold on them.\n\n"
        "Three things to carry to real data. **Measure on your own setting:** camera, distance, lighting, pose and the people "
        "in front of the camera all change the error; the upstream design target is a front-facing phone camera at arm's "
        "length. **A detection is not a correct result:** Section 7 showed faces detected with landmarks in the wrong place, "
        "and the model reports no quality flag that would catch it. **Faces are personal and biometric data:** this model "
        "does not identify anyone, but its outputs describe a person's face geometry and expressions; process them only with a "
        "lawful basis, keep them local where possible, and never use them to identify, profile or monitor people.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules and stage runner, carried in "
        "this standalone notebook and run in an isolated hash-locked environment, can stage and digest-verify the pinned task "
        "bundle, fetch and validate digest-pinned public face photographs, run the landmarker in IMAGE and VIDEO modes on the "
        "CPU, evaluate 14 landmarks against annotations beside two baselines, measure robustness and blendshape direction, "
        "and emit the shown machine-readable outputs and provenance — without the repository being reachable. It does "
        "**not** establish benchmark superiority, production fitness, fairness across demographic groups, or accuracy on "
        "faces unlike the ones shown.\n\n"
        "## Conclude with evidence · [Evaluation practice]\n\n"
        "Complete this in your own words, using the numbers your run printed:\n\n"
        "> On [102 neutral studio photographs of the London Set / your own annotated images], the MediaPipe Face Landmarker "
        "(task bundle version float16/1, object generation {MODEL_REVISION}) detected [detection rate] of the faces and placed 14 "
        "annotated landmarks with a mean NME of [mean] (median [median], 90th percentile [p90]), against [baseline] for the "
        "mean shape in the detector box and [baseline] for the mean shape in the image. The largest per-point disagreement "
        "was at [point]. Under perturbation, [setting] was the first to produce failures, and at 180° [what happened]. The "
        "smile score was higher on the smiling photograph in [k] of [n] pairs. The most important failure mode or "
        "uncertainty is [for example: confident wrong landmarks on upside-down faces; small groups; one studio setting]. "
        "These numbers do not show [accuracy in my deployment / fairness across groups / that blendshape scores are "
        "probabilities / ...]. Next I would [specific next experiment].\n\n"
        "<details>\n<summary>Check your reasoning: what makes a conclusion strong? (open after writing yours)</summary>\n\n"
        "A strong conclusion names the data (102 front-facing studio photographs, one per person), the measure (NME over 14 "
        "points, normalised by the annotated inter-ocular distance) and both baselines beside the model's number. It keeps the "
        "measured landmark accuracy apart from the sanity checks (perturbations, blendshape direction, VIDEO mode), and it says "
        "that blendshape scores are not probabilities. It states the main failure mode — detected faces with wrong landmarks "
        "at large rotations — and the limits: one studio, small groups, synthetic perturbations, possible training overlap. It "
        "does not generalise to other cameras, poses or populations and makes no fairness claim. It ends with a specific next "
        "step, such as annotating a sample from the intended setting. A weak conclusion says only that \"the model is accurate "
        "and robust\".\n\n"
        "</details>\n\n"
        "**Transfer:** switch on BYOD (Section 11) with frontal photographs from your own intended setting — a webcam, a phone "
        "camera, a document photo — that you are authorised to use. Before running, predict which of Section 7's perturbations "
        "your images resemble most, and check the overlays for landmarks that are detected but misplaced.\n\n"
        "## References\n\n"
        "- Bazarevsky, V., Kartynnik, Y., Vakunov, A., Raveendran, K., & Grundmann, M. (2019). *BlazeFace: Sub-millisecond neural face detection on mobile GPUs* (arXiv:1907.05047). arXiv. https://doi.org/10.48550/arXiv.1907.05047\n"
        "- DeBruine, L. (2016). *Young adult composite faces* [Data set]. figshare. https://doi.org/10.6084/m9.figshare.4055130.v1\n"
        "- DeBruine, L., & Jones, B. (2017). *Face Research Lab London Set* (Version 5) [Data set]. figshare. https://doi.org/10.6084/m9.figshare.5047666.v5\n"
        "- Google. (2023). *MediaPipe Face Landmarker task bundle `face_landmarker.task`, float16, version 1* [Model]. https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task\n"
        "- Grishchenko, I., Ablavatski, A., Kartynnik, Y., Raveendran, K., & Grundmann, M. (2020). *Attention Mesh: High-fidelity face mesh prediction in real-time* (arXiv:2006.10962). arXiv. https://doi.org/10.48550/arXiv.2006.10962\n"
        "- Kartynnik, Y., Ablavatski, A., Grishchenko, I., & Grundmann, M. (2019). *Real-time facial surface geometry from monocular video on mobile GPUs* (arXiv:1907.06724). arXiv. https://doi.org/10.48550/arXiv.1907.06724\n"
        "- Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C.-L., Yong, M. G., Lee, J., Chang, W.-T., Hua, W., Georg, M., & Grundmann, M. (2019). *MediaPipe: A framework for building perception pipelines* (arXiv:1906.08172). arXiv. https://doi.org/10.48550/arXiv.1906.08172\n"
        "- Sagonas, C., Antonakos, E., Tzimiropoulos, G., Zafeiriou, S., & Pantic, M. (2016). 300 Faces In-The-Wild Challenge: Database and results. *Image and Vision Computing, 47*, 3–18. https://doi.org/10.1016/j.imavis.2016.01.002 (the NME and failure-rate conventions)\n"
        "- Upstream model cards: MediaPipe Face Mesh V2 (https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf), MediaPipe Blendshape V2 (https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf), BlazeFace short range (https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf)\n"
        "- Upstream source: https://github.com/google-ai-edge/mediapipe (Apache-2.0)\n"
        "- Repository README: https://github.com/kurtvalcorza/mediapipe-face-landmarker-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/mediapipe-face-landmarker-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights notes: https://github.com/kurtvalcorza/mediapipe-face-landmarker-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- uv (the installer that builds the isolated environment): https://docs.astral.sh/uv/\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.2 (in the ml-worker repository)\n"
    ),
}
