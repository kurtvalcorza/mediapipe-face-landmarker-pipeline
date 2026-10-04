# Weight provenance, the pinned task bundle and DIMER hosting

This repository pins **one** model file: Google's MediaPipe Face Landmarker task bundle. Its manifest is
`weights/mediapipe-face-landmarker-float16-v1/dimer-base-manifest.json` (format `dimer_model_snapshot` v1,
`modelAssetSpec` 1.1). `src/mediapipe_face_landmarker_pipeline/pipeline.py` stages and verifies it; the bundle itself is
git-ignored and never committed.

## 1. Identity

- Upstream id: `mediapipe-models/face_landmarker/face_landmarker`, version `float16/1` (the path Google publishes on
  Cloud Storage), architecture source `google-ai-edge/mediapipe`.
- Mutable name: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`.
  An object name in Cloud Storage can be overwritten, so it does not define identity on its own (MODEL_ASSET_SPEC ID5/ID6).
- Immutable revision: Cloud Storage **object generation** `1683136941916318`. A generation number always refers to the
  same bytes; `stage_bundle` downloads `…/face_landmarker.task?generation=1683136941916318` and nothing else. The
  storage-reported upload time of that generation is 2023-05-03T18:02:21Z.
- Digest: `face_landmarker.task`, 3,758,596 bytes, SHA-256
  `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` (storage MD5 `sOcnSQehZEQE/vZrKN1thQ==`, recorded
  for cross-checking only; it is not trusted for verification).
- The `latest` alias (`…/float16/latest/face_landmarker.task`) served a different object generation (`1683136941468629`)
  with the same size when it was checked on 2026-10-04. It is never used.
- Licence: **Apache-2.0** (upstream model cards and the `google-ai-edge/mediapipe` repository). Redistribution is
  permitted; this repository nevertheless downloads the bundle from Google at run time instead of committing it.

## 2. What the bundle contains

`face_landmarker.task` is a zip archive whose members are **stored** (uncompressed). `inspect_bundle` reads it in memory,
refuses absolute, `..`, directory and symlink members, and extracts nothing. `verify_bundle` checks each member against
the manifest:

| Member | Bytes | SHA-256 | Role |
|---|---|---|---|
| `face_detector.tflite` | 229,746 | `b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f` | BlazeFace short-range face detector: boxes and in-plane rotation |
| `face_landmarks_detector.tflite` | 2,553,590 | `c7d54204ce0448474c7f3fa9af494787c0965cbdd6f20fc72867e43046bd43d5` | face mesh: 478 landmarks (468 surface + 10 iris) and a face-presence score |
| `face_blendshapes.tflite` | 955,312 | `4f36dded049db18d76048567439b2a7f58f1daabc00d78bfe8f3ad396a2d2082` | blendshape network: 146 landmarks in, 52 scores out |
| `geometry_pipeline_metadata_landmarks.binarypb` | 19,376 | `bdbcda96dfcb7da883da124aaa2c55dee49770d934f0fcc71747f8c21bdc75b4` | canonical face model used for the facial transformation matrix |

The three `.tflite` members are TFLite flatbuffers: serialized graphs and float16 weights read by the TFLite interpreter
inside `mediapipe`. They contain no Python and no pickle, and loading them executes no code from the file; the operators
they reference are compiled into the `mediapipe` wheel. The `.binarypb` member is a serialized protocol buffer. No remote
code is required (`remoteCodeRequired: false`).

`detect_face_boxes` reads `face_detector.tflite` from the verified zip in memory and runs it alone through the
`mediapipe` `FaceDetector` task; it is used only by the detector-box mean-shape baseline.

## 3. Serialization format and the DIMER upload format

MODEL_ASSET_SPEC §11.1 prefers non-code-executing tensor formats such as safetensors; §11 treats pickle-based formats
(`.pt`, `.pth`, `.pkl`) as trusted executable serialization. The fleet inventory's upload-format rule lists
`.safetensors`, `.pt`, `.pth` and `.h5` as accepted DIMER uploads "among others", and the inventory row for this model
already plans a `.task` upload.

`.task` is none of the listed formats. The decision recorded here:

- **The DIMER upload is the upstream `face_landmarker.task` unchanged** (3,758,596 bytes, SHA-256 above). It is the only
  form the MediaPipe Tasks runtime loads; no conversion is performed or claimed.
- **Conversion to safetensors is not technically valid** for this model. The networks are TFLite graphs with fixed
  operators, pre- and post-processing and a geometry protobuf, executed by the MediaPipe graph; a safetensors file of the
  weights alone could not be loaded by any MediaPipe API and would not reproduce the output. MODEL_ASSET_SPEC §27 (derived
  assets) therefore does not apply.
- **Security classification:** a zip of TFLite flatbuffers and a protobuf is a data format, not a code-capable one like
  pickle. The residual risk is the TFLite and MediaPipe parsers themselves; it is bounded by pinning the exact bytes and
  verifying every member before loading.
- **Open item for the maintainer:** confirm that the DIMER upload path accepts a `.task` file. Until that is confirmed,
  the format is a recorded deviation from the formats the inventory lists, not an accepted one.

## 4. Runtime facts

- Runtime: `mediapipe==1.0.0` (Tasks API, `FaceLandmarker`, IMAGE and VIDEO running modes), `numpy==2.5.3`,
  `pillow==12.3.0`, CPython 3.12. The tutorial installs them, hash-locked, into an isolated environment.
- Precision and device: the bundle's float16 weights on the CPU through TFLite's XNNPACK delegate. No GPU delegate is used.
- Why 1.0.0 and not 1.0.1: the 1.0.1 wheel's `libmediapipe.so` links `libEGL.so.1` and `libGLESv2.so.2`, which a minimal
  Linux image lacks (`OSError: libEGL.so.1` when the landmarker is created); 0.10.33 and 0.10.35 do the same. 1.0.0,
  0.10.30–0.10.32 and 0.10.21 have no such dependency. Observed in a minimal Linux container on 2026-10-04.
- Configuration: `num_faces` 1 by default (1..10 accepted), `min_face_detection_confidence`, `min_face_presence_confidence`
  and `min_tracking_confidence` 0.5 (upstream defaults), blendshapes and transformation matrices on.
- CPU cost: about 20–80 ms per 1350 × 1350 photograph in the local 4-CPU container (one measurement environment, not a
  benchmark).

## 5. Sample data

The tutorial's sample faces are not model weights and are never committed. `src/mediapipe_face_landmarker_pipeline/sample_manifest.json`
pins 327 files (102,754,576 bytes) from `debruine/webmorphR.stim` at commit `fa8b78fda2d659bb74ce62fcd99c4407551d2a77`,
each by size and SHA-256; `tools/pin_samples.py` regenerates it. Sets and licences: Face Research Lab London Set
(neutral and smiling photographs, annotations, participant information; CC BY 4.0, DeBruine & Jones, 2017,
https://doi.org/10.6084/m9.figshare.5047666.v5) and Young adult composite faces (CC BY 4.0, DeBruine, 2016,
https://doi.org/10.6084/m9.figshare.4055130.v1). The participants gave signed consent for their images to be used in
lab-based and web-based studies and to illustrate research.

## 6. Re-pinning

`python tools/pin_bundle.py --generation <N>` downloads one object generation and rewrites the manifest with the bundle
and member digests; `MODEL_REVISION`, `MODEL_SHA256` and `MODEL_BYTES` in `pipeline.py` must then be updated to match,
and `tools/validate_release_assets.py` checks that the manifest, `pipeline.py`, `README.md`, `MODEL_CARD.md` and this
file agree.
