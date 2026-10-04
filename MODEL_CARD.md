---
license: apache-2.0
model_card_spec: "1.2"
pipeline_tag: keypoint-detection
task: "Facial landmark estimation (478 3D landmarks), 52 blendshape scores and a facial transformation matrix"
base_model: mediapipe-models/face_landmarker/face_landmarker
base_model_version: "float16/1"
date_published: "2023-05-03"
date_published_source: "Google Cloud Storage object generation `1683136941916318` of `face_landmarker.task` (Last-Modified 2023-05-03T18:02:21Z), the earliest date this repository can establish from a public source. The date of the public announcement of the MediaPipe Face Landmarker task is not established here."
---

# MediaPipe Face Landmarker — task bundle float16/1 (478 3D landmarks, 52 blendshapes)

[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-google--ai--edge%2Fmediapipe-181717?style=flat&logo=github&logoColor=white)](https://github.com/google-ai-edge/mediapipe)
[![Upstream model card](https://img.shields.io/badge/Upstream%20model%20card-Face%20Mesh%20V2-4285F4?style=flat)](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)
[![arXiv Paper](https://img.shields.io/badge/arXiv-1907.06724-b31b1b.svg)](https://arxiv.org/abs/1907.06724)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** The task bundle is used unmodified under its upstream Apache-2.0 licence; the accompanying code and notebook are Apache-2.0. Nothing here is validated for production, and no benchmark result is claimed.

> [!IMPORTANT]
> **This model does not recognise or identify people.** It returns face geometry and expression coefficients, not an identity, and this repository never compares faces with each other. Faces are personal data, and landmarks and blendshape scores derived from them can be biometric data. Blendshape scores are not calibrated probabilities, and the pipeline applies no threshold to them.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run, self-contained Google Colab notebook. It carries the repository's code in its own cells and runs end to end without cloning the repository. It installs nothing into the notebook kernel: every stage runs in an isolated environment built from a committed hash lock, so `Run all` needs no runtime restart. A CPU runtime is enough.

- **Guided Face Landmark Task-Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/mediapipe-face-landmarker-pipeline/blob/main/tutorials/mediapipe_face_landmarker_colab.ipynb) [`mediapipe_face_landmarker_colab.ipynb`](https://github.com/kurtvalcorza/mediapipe-face-landmarker-pipeline/blob/main/tutorials/mediapipe_face_landmarker_colab.ipynb)  
  *The pinned task bundle staged and verified member by member; 214 digest-pinned public face photographs validated, with refusal probes; the full output contract on one photograph; landmark NME on 102 annotated faces beside two mean-shape baselines and a self-reported-group breakdown; robustness under rotation, downscaling, blur, brightness and JPEG compression; a paired neutral/smiling blendshape sanity check; ten new composite faces exported and a VIDEO-mode frame sequence; one provenance record; optional BYOD and a change-one-thing activity.*

---

#### Description

The packaged model is Google's MediaPipe Face Landmarker task bundle `face_landmarker.task`, published as `mediapipe-models/face_landmarker/face_landmarker`, version `float16/1`. This repository pins Google Cloud Storage object generation `1683136941916318` of that file (3,758,596 bytes, SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`). Its architecture source is `google-ai-edge/mediapipe`.

The bundle is a stored, uncompressed zip of three small convolutional or MLP networks in TFLite flatbuffer format and one binary protobuf. A BlazeFace short-range face detector (`face_detector.tflite`) finds each face and its in-plane rotation (Bazarevsky et al., 2019). A face-mesh network (`face_landmarks_detector.tflite`) predicts 478 landmarks on an upright crop of each face: 468 surface points and 10 iris points (Kartynnik et al., 2019; Grishchenko et al., 2020). Each landmark has `x` and `y` as fractions of the image size and a relative depth `z`. A blendshape network (`face_blendshapes.tflite`) maps 146 of those landmarks to 52 named scores in [0, 1]: `_neutral` and 51 expression coefficients. The canonical-face geometry (`geometry_pipeline_metadata_landmarks.binarypb`) yields a 4 × 4 facial transformation matrix per face.

Inference is a fixed forward pass through the `mediapipe` 1.0.0 Tasks API (`FaceLandmarker`) on the CPU. In VIDEO mode the previous frame's landmarks place the next frame's face region. No adaptation takes place: the weights are never trained, fine-tuned or conditioned in context.

This repository adds the following, all as code a reader runs:

- pinning and verification of the bundle by object generation, byte size and SHA-256, and of each zip member by name, size and SHA-256 (`stage_bundle`, `verify_bundle`);
- input validation before the model runs (`validate_image`, `validate_num_faces`);
- a wrapper that fixes the configuration and turns results into identified CSV/JSON rows and overlays (`FaceLandmarkerPipeline`, `landmark_rows`, `blendshape_rows`, `matrix_records`, `draw_overlay`);
- an evaluation path against annotated public faces, with two mean-shape baselines, controlled perturbations and a paired blendshape sanity check (`metrics.py`, `samples.py`, `tools/tutorial_stages.py`);
- a standalone tutorial notebook that runs all of it in a hash-locked environment.

#### Intended Use and Limitations

The repository was built to teach and check face-landmark estimation on still images and short frame sequences, not to serve a product.

###### Primary Intended Uses

The task is facial landmark estimation. The input is an RGB image (or frames in timestamp order) with up to 10 faces. The output per face is 478 landmarks with `x`, `y`, `z` and pixel coordinates, 52 named blendshape scores, and a 4 × 4 facial transformation matrix, each row keyed by `image_id`, `face_id` and `landmark_id` or `blendshape_index`.

The application domains envisioned are the ones the upstream model was designed for, plus teaching:

- teaching how a pretrained landmark model is validated, evaluated against annotations and baselines, and stress-tested;
- avatar animation and augmented-reality effects driven by blendshape scores or the transformation matrix, from a front-facing camera;
- geometry pre-processing for a reader's own pipeline, such as face alignment, cropping or measuring head roll before a downstream model.

In a larger system the intended role is a fixed, zero-configuration geometry extractor whose output a downstream component consumes. The repository code runs locally and embeds in a reader's own application; it sets no threshold and makes no decision.

###### Primary Intended Users

The intended users are machine learning engineers, computer vision students and teaching staff, and application developers building research prototypes. The envisioned deployment settings are teaching, research, and self-hosted prototypes running on a reader's own machine or hosted notebook.

The pipeline assumes that its users:

- know that normalised image coordinates scale with the image width and height, and that `z` is a relative depth, not a distance;
- know that blendshape scores are rig coefficients, not probabilities of an expression;
- understand that a returned face can carry misplaced landmarks, and check outputs on their own data before relying on them;
- know the legal and ethical constraints on processing images of people, including consent and biometric-data rules in their jurisdiction.

The pipeline is robust to malformed files and out-of-range sizes, which it refuses. It is not robust to content outside its design range (see `Out-of-scope use cases`), and it raises no warning when that happens.

###### Out-of-scope use cases

Capability boundaries:

- **Not for face recognition, identification, verification or re-identification.** The model produces no identity embedding, and this repository never compares faces.
- **Not for emotion, age, gender, ethnicity or health inference.** Blendshape scores describe facial deformation, not internal states or attributes. The self-reported attributes of the sample are used only to break down the error.
- **Not for fine-tuning or adaptation.** The bundle is a fixed edge model, and the repository exposes no training path.
- **Not for hands, body pose or full-body landmarks.** Those are separate MediaPipe tasks that this repository does not package.

Input boundaries:

- **Not for images outside 64..8192 px per side, above 40,000,000 pixels, or above 50 MB.** `validate_image` refuses them.
- **Not for more than 10 faces per image.** `validate_num_faces` refuses `num_faces` outside 1..10.
- **Not for faces rotated beyond about 90° in the image plane.** In the local run, rotation by 120° and 180° produced detections whose landmarks were misplaced (failure rate 0.8 and 0.9 on 20 faces).
- **Not for strongly turned, partly hidden or very small faces.** The upstream model card states the mesh is not suited to faces looking more than 80° away, less than 50 % visible, or too small to rescale to the model input. This repository did not measure these cases.

Decision boundaries:

- **Not for any automated decision about a person** — access, employment, education, credit, insurance, policing, border control or health — with or without human oversight.
- **Not for attention, engagement, honesty or emotion scoring** of students, employees, drivers or customers.

#### Factors

The model's behaviour varies with the person in the image, the camera and the capture conditions.

###### Groups

The pipeline is human-centric: every input is a photograph of a person's face.

The evaluation data is the Face Research Lab London Set (DeBruine & Jones, 2017): 102 adults photographed once each, with self-reported gender (49 female, 53 male), ethnicity (69 white, 13 black, 10 West Asian, 9 East Asian, 1 East Asian/white) and age (42 aged 18–24, 41 aged 25–34, 17 aged 35+, 2 not reported). The local run observed these mean NMEs:

| Group | n | mean NME |
|---|---|---|
| female | 49 | 0.0180 |
| male | 53 | 0.0206 |
| black | 13 | 0.0224 |
| East Asian | 9 | 0.0192 |
| West Asian | 10 | 0.0167 |
| white | 69 | 0.0193 |

These are observations from one run on small groups, with no repeated measurement. They do not support a fairness conclusion, and the cause of a difference cannot be attributed to the attribute: lighting, contrast, facial hair, the annotators and the point definitions also vary between faces. Skin tone was not recorded.

The upstream pretraining data is not disclosed in detail. The upstream Face Mesh V2 model card reports its own evaluation on 1,700 images over 17 geographic subregions, two perceived genders and six Fitzpatrick skin types; this repository does not reproduce it. An operator must therefore audit the model on their own population: measure detection rate and landmark error per relevant group (for example skin tone, age, gender presentation, facial hair, eyewear and head coverings) on annotated images from their own setting before relying on the output.

###### Instrumentation

The evaluation photographs were taken in one studio in London in April 2012, at 1350 × 1350 px, full colour, front-facing, under constant lighting, as described by the dataset authors. The annotations are the authors' 189-point WebMorph templates; 14 points are matched to MediaPipe indices (`LANDMARK_MAP` in `samples.py`). The composite faces are pixel averages of four London Set photographs (DeBruine, 2016).

The upstream model card states that the mesh training images came from a diverse set of smartphone front- and back-facing cameras in real-world conditions, and that the blendshape model was trained on lab-captured multi-view data with reconstructed 3D meshes.

Instrument characteristics propagate as follows. Lower resolution, defocus, sensor noise and compression reduce landmark precision; the local run measured this only on synthetic downscaling, blur and JPEG re-encoding of studio photographs. Lens distortion, unusual focal lengths and mirrored front-camera images change the geometry the transformation matrix assumes. The pipeline cannot detect any of these defects: it reports no image-quality or per-landmark confidence.

###### Environment

Operating environment: the pipeline runs on Linux x86_64 with CPython 3.12 and the locked `mediapipe` 1.0.0, `numpy` 2.5.3 and `pillow` 12.3.0. Inference uses the CPU through TFLite's XNNPACK delegate in the bundle's float16 weights. No GPU or other delegate is used or supported by this repository. `mediapipe` 1.0.1 was not chosen because its wheel links `libEGL.so.1` and `libGLESv2.so.2`, which minimal Linux images lack; this was observed when loading it in a minimal container.

Data environment: the reported behaviour holds for upright, front-facing, evenly lit faces that fill a large part of the image, like the evaluation photographs. The upstream design target is a front-facing phone camera at arm's length. Accuracy is expected to degrade with large rotations, profile views, occlusion, motion blur, low light and small faces. Degradation can take the form of confident, misplaced landmarks rather than a missing detection, as the local run showed at 120° and 180°.

#### Metrics

The measures were chosen because landmarks have point-wise ground truth on annotated faces, while blendshape scores have none in the sample.

###### Performance Measures

The pipeline reports these measures, named as the code reports them:

- `nme_mean`, `nme_median`, `nme_std`, `nme_p90`, `nme_max` — the mean 2D distance between the 14 predicted and annotated points, divided by the annotated inter-ocular distance (eye centres as midpoints of the eye corners). This is the upstream model card's normalisation and the common landmark-literature convention (Sagonas et al., 2016).
- `detection_rate` — the share of images in which a face was returned.
- `failure_rate_at_0.10` — the share of faces whose NME exceeds 0.10.
- `nme_mean_bootstrap_95ci` — a percentile bootstrap interval of the mean NME over faces.
- `mean_error_iod` per point — the per-landmark error, which separates definition offsets from general error.
- `consistency_nme_mean` and `annotation_nme_mean` per perturbation — stability of the output, and correctness against the annotation, under rotation, downscaling, blur, brightness and JPEG compression.
- For blendshapes: `fraction_expressive_higher`, `median_paired_difference`, `rank_auc` and `sign_test_p` on neutral/smiling pairs. These are labelled a sanity check, not a performance measure.

NME alone would hide missed faces, so detection rate is reported beside it. Detection rate alone would hide misplaced landmarks, so the failure rate is reported too; the 180° rotation row shows why both are needed (detection 0.9, failure 0.9). Consistency without annotation error would reward a model that is stably wrong.

Observed in the local CPU run (102 neutral photographs, one pass):

| Measure | model | mean shape in detector box | mean shape in image |
|---|---|---|---|
| `detection_rate` | 1.0 | 1.0 | 1.0 |
| `nme_mean` | 0.01937 | 0.0752 | 0.1532 |
| `nme_median` | 0.0194 | 0.06602 | 0.12684 |
| `nme_p90` | 0.02449 | 0.1147 | 0.25363 |
| `failure_rate_at_0.10` | 0.0 | 0.1961 | 0.7059 |

The model's `nme_mean_bootstrap_95ci` was [0.01868, 0.02006]. The largest per-point errors were at `chin_bottom_centre` (0.0389) and `subnasale` (0.0295); the smallest were at the lip midlines (0.0136–0.0138). The smile score was higher on the smiling photograph in 102 of 102 pairs (`rank_auc` 0.948).

The pipeline reports no performance measure for the blendshape scores, the `z` coordinate or the transformation matrix. Evaluating them needs data the sample does not have: expression-coded video with per-frame coefficient annotation, depth ground truth, or measured head pose.

###### Decision thresholds

The pipeline applies these thresholds:

- `min_face_detection_confidence`, `min_face_presence_confidence` and `min_tracking_confidence` are `0.5`, the upstream defaults. A face below them is not returned. They were not tuned on any data in this repository.
- `num_faces` caps the faces returned per image (1 by default, 1..10 accepted). MediaPipe returns the faces in its own order, and `face_id` is the index in that order.
- Evaluation counts a face as a failure when its NME exceeds `FAILURE_NME` = 0.10, a common reporting convention. It is not an acceptance threshold for any use.

No threshold is applied to blendshape scores. They are uncalibrated rig coefficients, so a fixed cut-off such as 0.5 would carry no common meaning across people, cameras or expressions. Calibrating one is the deployer's responsibility. A deployment that turns a score into a decision should choose the cut-off on labelled data from its own setting and weigh the cost of a false positive against a false negative. For example, an avatar effect tolerates false triggers, whereas any use that affects a person does not and is out of scope here.

No acceptance threshold on NME was set during development.

###### Approaches to uncertainty and variability

Every reported number comes from a single pass over the stated images, with no repeated runs.

- **Estimation procedure.** All 102 neutral photographs are evaluated with no selection. The baselines are leave-one-out: each face is excluded from its own mean shape. Perturbations use the first 20 faces by id. Blendshape pairs use all 102 people.
- **Dispersion.** The standard deviation and 90th percentile of NME over faces are reported. Only the mean NME carries an interval: a 2,000-resample percentile bootstrap over faces with seed `0`. Group means, perturbation rows and blendshape statistics carry no interval.
- **Run-to-run variability.** The landmarker is deterministic for a given input on a given machine; two local executions of the notebook gave identical values. Floating-point results may differ slightly between CPU types and library builds. Bootstrap resampling is seeded. Nothing else in the pipeline is random.
- **Confidence outputs.** The pipeline exports no per-landmark confidence. The detection and presence scores are used only as internal gates. Blendshape scores are not calibrated probabilities; a caller who needs a probability that an expression is present must calibrate on labelled data from the intended setting.
- **Sample effects.** One studio setting, small groups and possible overlap with the upstream training data (which cannot be ruled out) limit what these numbers generalise to.

#### Ethical considerations and biases

Faces are personal data, and landmarks and blendshape scores derived from them can be biometric data. No external board reviewed this repository, and the pipeline was not cleared through testing with any specific group.

###### Data

The upstream training data is described only at the level of the upstream model cards: real-world smartphone images annotated by 11 annotators for the face mesh, and lab-captured multi-view facial images with reconstructed 3D meshes for the blendshape model. The images, their subjects, consent terms and composition are not disclosed. Whether they include sensitive personal data is unknown, and it is likely that they contain faces of identifiable people.

This repository distributes code, the bundle manifest, the sample manifest and documentation. It does not distribute the task bundle, which the code downloads from Google Cloud Storage, or any photograph. The sample photographs are downloaded at run time from the public repository `debruine/webmorphR.stim` at commit `fa8b78fda2d659bb74ce62fcd99c4407551d2a77`. They are published under CC BY 4.0 by the dataset authors, whose participants gave signed consent for use in lab-based and web-based studies and to illustrate research. The participants' self-reported age, gender and ethnicity are used only to break down the error.

An operator who supplies images is responsible for having a lawful basis and, where required, consent to process the faces in them. The pipeline performs no audit of personal, sensitive or proprietary content. The notebook's BYOD path keeps uploaded images and outputs inside the runtime and sends nothing to any service.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing, education or any other domain central to human life.

It has not been validated for any such use by anyone. The upstream model cards state that the models are not intended for human life-critical decisions, and this repository's evaluation covers only 102 studio photographs.

Some uses are foreseeable but not intended: driver-drowsiness monitoring from eye blendshapes, clinical facial-palsy assessment, and facial-expression analysis in psychology studies. Each would need independent domain validation on the target population and capture conditions, human oversight of every outcome, and any regulatory clearance the domain requires. This repository provides none of these.

###### Mitigations

The mitigations implemented in this repository are:

- **Supply-chain integrity.** `stage_bundle` downloads only from the URL pinned to object generation `1683136941916318`, writes the file only after its size and SHA-256 match, and never falls back to another source. `verify_bundle` re-checks the bundle's size and SHA-256 and each member's name, size and SHA-256 against `weights/mediapipe-face-landmarker-float16-v1/dimer-base-manifest.json` before every load. `read_manifest` refuses a manifest whose identity differs from the constants in `pipeline.py` or that has duplicate keys. `inspect_bundle` refuses absolute, `..`, directory and symlink members and extracts nothing.
- **Sample integrity.** `fetch_samples` refuses any sample file whose size or SHA-256 differs from `sample_manifest.json`, and replaces a corrupted cache entry.
- **Input integrity.** `validate_image` refuses undecodable files, sides outside 64..8192 px, more than 40,000,000 pixels and files above 50 MB. It reports every colour conversion and EXIF transpose instead of applying them silently. `validate_num_faces` and `validate_confidence` refuse out-of-range settings. The BYOD stage refuses zips with unsafe members, more than 20 files or more than 200 MB.
- **Statistical mitigations.** The evaluation reports two baselines, a failure rate beside the detection rate, per-point errors and group sizes `n`. It states one blendshape expectation before running and reports the other scores descriptively.
- **Reproducibility.** Every runtime dependency is pinned in `pyproject.toml` and locked with hashes in `tutorials/requirements-colab.lock.txt`. The notebook installs the lock with `--require-hashes --only-binary :all:` into an isolated CPython 3.12.12 environment. The bootstrap is seeded, and `outputs/mediapipe_face_landmarker_result.json` records the model identity, member digests, runtime versions, configuration and sample-manifest digest.
- **Refusals.** The repository exposes no face comparison, no identity embedding, no attribute or emotion classifier, no threshold on blendshape scores and no training path. None of these exists in the code.

###### Risks and harms

| Failure mode | Who bears the harm | Conditions and likelihood | Magnitude |
|---|---|---|---|
| Misplaced landmarks returned as a valid face, with no quality flag | the operator, and the person depicted if a decision follows | large in-plane rotation (observed at 120° and 180°), and expected for profile views, occlusion and small faces; likely outside the design range | high if the output drives anything consequential; low for an avatar effect |
| Higher error or missed detection for some groups | the people in under-served groups | unmeasured on the operator's population; the local run saw group means differing by a few thousandths on small groups | moderate to high in any deployment affecting people |
| Over-reading blendshape scores as emotions or probabilities (automation bias) | the person depicted | likely when scores are displayed as percentages or thresholded without calibration | high: attributing states such as deception, inattention or distress to a person |
| Function creep from AR or teaching into surveillance or profiling | the people recorded | possible once a camera pipeline exists | high: biometric processing without consent |
| Leakage of biometric data | the people depicted | landmark CSVs and overlays are written to disk and can be shared or retained beyond their purpose | moderate to high, depending on jurisdiction and the data's spread |
| Over-confidence from studio-only evidence | the operator | likely if the local numbers are taken as field accuracy | moderate: a deployment fails where it was expected to work |

###### Use cases

The following uses are unacceptable even where the model would work:

- **Surveillance and biometric profiling.** Tracking, monitoring or profiling people in public or private spaces, in workplaces or schools, or online; building face-geometry databases of people who have not consented.
- **Identification by combination.** Feeding these outputs into a face-recognition, re-identification or matching system.
- **Demographic or emotion inference.** Inferring ethnicity, gender, age, sexual orientation, health, emotion, honesty or personality from landmarks or blendshape scores, and any social scoring.
- **Unlawful discrimination.** Using any output in decisions about employment, housing, credit, insurance, education or access to healthcare.
- **Deceptive or manipulative uses.** Driving non-consensual impersonation, deepfakes or face-swaps of real people, or covertly measuring reactions to manipulate behaviour.
- **Prohibited uses.** Any use prohibited by law in the deployer's jurisdiction, including rules on biometric data. The Apache-2.0 licence of the bundle adds no use restrictions; the restrictions above are this repository's own.

---

## Immutable provenance

| Item | Value |
|---|---|
| Model id / version | `mediapipe-models/face_landmarker/face_landmarker` / `float16/1` |
| Download URL | `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task?generation=1683136941916318` |
| Object generation (revision) | `1683136941916318` |
| Bundle | `face_landmarker.task`, 3,758,596 bytes, SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |
| `face_detector.tflite` | 229,746 bytes, SHA-256 `b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f` |
| `face_landmarks_detector.tflite` | 2,553,590 bytes, SHA-256 `c7d54204ce0448474c7f3fa9af494787c0965cbdd6f20fc72867e43046bd43d5` |
| `face_blendshapes.tflite` | 955,312 bytes, SHA-256 `4f36dded049db18d76048567439b2a7f58f1daabc00d78bfe8f3ad396a2d2082` |
| `geometry_pipeline_metadata_landmarks.binarypb` | 19,376 bytes, SHA-256 `bdbcda96dfcb7da883da124aaa2c55dee49770d934f0fcc71747f8c21bdc75b4` |
| Architecture source | `google-ai-edge/mediapipe` (Apache-2.0) |
| Runtime | `mediapipe==1.0.0`, `numpy==2.5.3`, `pillow==12.3.0`, CPython 3.12.12 |
| Sample faces | `debruine/webmorphR.stim` at `fa8b78fda2d659bb74ce62fcd99c4407551d2a77`, 327 files, 102,754,576 bytes, each pinned in `src/mediapipe_face_landmarker_pipeline/sample_manifest.json` |

`docs/WEIGHTS.md` records the serialization format of the `.task` file and how each member is verified.

## Input/output contract

- **Input:** one image Pillow can decode, 64..8192 px per side, at most 40,000,000 pixels and 50 MB; converted to 8-bit RGB with any conversion reported; `num_faces` 1..10.
- **Landmarks:** `image_id, face_id, landmark_id, x, y, z, x_px, y_px, z_px`; 478 rows per face. `z_px` uses the image width as its scale.
- **Blendshapes:** `image_id, face_id, blendshape_index, blendshape, score`; 52 rows per face, `_neutral` at index 0.
- **Matrices:** a JSON list of `{image_id, face_id, matrix}` with a 4 × 4 row-major matrix.
- **No face** is an empty result, not an error.

## Verification records

- **Date:** 2026-10-04
- **Subject:** `tutorials/mediapipe_face_landmarker_colab.ipynb`, git blob `9b8a54dcc45294998578e00dcb7d67e0a27b6203`, generated from commit `e49deff733c8aa7a91cd99d12f9fb4980a181935`
- **Runtime:** CPU-only Linux x86_64 container, 4 CPUs, no GPU; kernel CPython 3.11.15; stages in the notebook's isolated environment: CPython 3.12.12 downloaded fresh by `uv` 0.12.15, `mediapipe` 1.0.0, `numpy` 2.5.3, `pillow` 12.3.0
- **Procedure:** a copy of the notebook executed top to bottom with `tools/execute_notebook.py` (`jupyter nbconvert --execute`) from an empty working directory, defaults unchanged; then two further copies with `USE_BYOD = True` and `RUN_ACTIVITY = True`: a zip of two composite faces, and a zip with a `../` member
- **Observed result:** the default path completed 13 of 13 code cells in one pass in 120 s including the environment build, with the values in `Performance Measures`; the compatible BYOD zip produced all outputs; the incompatible one stopped with `BYOD zip has an unsafe member path '../escape.jpg'`. Executed copies are in `docs/execution-evidence/2026-10-04/`
- **Caveats:** this is a local container, not the clean hosted Google Colab runtime that release requires; a hosted run has not been recorded. One pass, no repeated runs

## References

- Bazarevsky, V., Kartynnik, Y., Vakunov, A., Raveendran, K., & Grundmann, M. (2019). *BlazeFace: Sub-millisecond neural face detection on mobile GPUs* (arXiv:1907.05047). https://doi.org/10.48550/arXiv.1907.05047
- DeBruine, L. (2016). *Young adult composite faces* [Data set]. figshare. https://doi.org/10.6084/m9.figshare.4055130.v1
- DeBruine, L., & Jones, B. (2017). *Face Research Lab London Set* (Version 5) [Data set]. figshare. https://doi.org/10.6084/m9.figshare.5047666.v5
- Grishchenko, I., Ablavatski, A., Kartynnik, Y., Raveendran, K., & Grundmann, M. (2020). *Attention Mesh: High-fidelity face mesh prediction in real-time* (arXiv:2006.10962). https://doi.org/10.48550/arXiv.2006.10962
- Kartynnik, Y., Ablavatski, A., Grishchenko, I., & Grundmann, M. (2019). *Real-time facial surface geometry from monocular video on mobile GPUs* (arXiv:1907.06724). https://doi.org/10.48550/arXiv.1907.06724
- Sagonas, C., Antonakos, E., Tzimiropoulos, G., Zafeiriou, S., & Pantic, M. (2016). 300 Faces In-The-Wild Challenge: Database and results. *Image and Vision Computing, 47*, 3–18. https://doi.org/10.1016/j.imavis.2016.01.002
- Upstream model cards: [Face Mesh V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf), [Blendshape V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf), [BlazeFace short range](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf)
