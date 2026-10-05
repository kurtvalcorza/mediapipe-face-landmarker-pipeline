"""Stage runner for the standalone MediaPipe Face Landmarker tutorial (NOTEBOOK_SPEC 2.2 §25.13 isolated environment).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/``) and runs every stage with the interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --weights WEIGHTS_DIR --stage evaluate

Nothing is installed into the notebook kernel. Each stage is a separate process that starts from files only: the
verified bundle under ``--weights``, the digest-verified sample cache under ``--weights``, and the JSON records of
earlier stages. Learner-facing exports go to ``RUN_DIR/outputs``; hand-off state goes to ``RUN_DIR/state``. On failure a
stage writes ``RUN_DIR/state/<stage>.error.json`` with the exception type and message, which the notebook re-raises.

Stages: weights -> prepare -> inference -> evaluate -> robustness -> blendshapes -> newdata -> export, then the
optional ``byod`` and ``activity``.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

STEM = "mediapipe_face_landmarker"
SAMPLE_CACHE = "face-samples"
WALKTHROUGH_ID = "neutral/001_03"
COLLAGE_IDS = ("composite/f_white", "composite/m_african")
VIDEO_ID = "composite/f_multi"
ROBUSTNESS_FACES = 20
PERTURBATIONS = (
    ("rotate", 15.0), ("rotate", 30.0), ("rotate", 45.0), ("rotate", 90.0), ("rotate", 180.0),
    ("scale", 0.5), ("scale", 0.25), ("scale", 0.12),
    ("blur", 2.0), ("blur", 6.0), ("blur", 12.0),
    ("brightness", 0.35), ("brightness", 1.8),
    ("jpeg", 20.0), ("jpeg", 5.0),
)  # fmt: skip
VIDEO_FRAMES = 30
VIDEO_FPS = 30
BOOTSTRAP_SEED = 0
BYOD_MAX_IMAGES = 20
BYOD_MAX_ZIP_BYTES = 200 * 1024 * 1024
BYOD_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff")
PAIRED_HYPOTHESIS = ("mouthSmileLeft", "mouthSmileRight")
DESCRIPTIVE_BLENDSHAPES = ("cheekSquintLeft", "cheekSquintRight", "eyeSquintLeft", "eyeSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "jawOpen", "mouthClose")


# --------------------------------------------------------------------------------------------------
# run context and small helpers
# --------------------------------------------------------------------------------------------------


class Run:
    """Paths of one run: carried sources and state under ``root``, bundle and sample cache under ``weights``."""

    def __init__(self, root: Path, weights: Path, options: argparse.Namespace) -> None:
        self.root = root
        self.weights = weights
        self.options = options
        self.out = root / "outputs"
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    @property
    def bundle_dir(self) -> Path:
        from mediapipe_face_landmarker_pipeline import MODEL_KEY

        return self.weights / MODEL_KEY

    @property
    def sample_dir(self) -> Path:
        return self.weights / SAMPLE_CACHE

    def write_state(self, name: str, value: Any) -> Path:
        from mediapipe_face_landmarker_pipeline import write_json

        return write_json(value, self.state / name)

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from the top)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_output(self, name: str, value: Any) -> Path:
        from mediapipe_face_landmarker_pipeline import write_json

        return write_json(value, self.out / name)

    def read_output(self, name: str, needed_by: str) -> Any:
        path = self.out / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from the top)")
        return json.loads(path.read_text(encoding="utf-8"))


def runtime_versions() -> dict[str, Any]:
    import numpy
    import PIL

    try:
        import importlib.metadata as metadata

        mediapipe_version = metadata.version("mediapipe")
    except Exception:  # noqa: BLE001 - a stub run has no mediapipe distribution
        mediapipe_version = "not installed"
    return {
        "python": platform.python_version(),
        "mediapipe": mediapipe_version,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "delegate": "CPU (TFLite XNNPACK)",
    }


def r(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


# --------------------------------------------------------------------------------------------------
# model and data factories (the offline stage test replaces these with stubs)
# --------------------------------------------------------------------------------------------------


def make_landmarker(run: Run, **kwargs: Any) -> Any:
    from mediapipe_face_landmarker_pipeline import FaceLandmarkerPipeline

    return FaceLandmarkerPipeline.from_weights(run.bundle_dir, **kwargs)


def face_boxes(run: Run, image: Any) -> list[dict[str, Any]]:
    from mediapipe_face_landmarker_pipeline import detect_face_boxes

    return detect_face_boxes(run.bundle_dir, image)


def fetch(run: Run) -> dict[str, Any]:
    from mediapipe_face_landmarker_pipeline import fetch_samples

    return fetch_samples(run.sample_dir)


def stage_files(run: Run) -> dict[str, Any]:
    """Install the carried manifest, fetch the bundle from the pinned generation if absent, verify it."""
    from mediapipe_face_landmarker_pipeline import MANIFEST_NAME, stage_bundle, verify_bundle

    carried = run.root / "weights" / run.bundle_dir.name / MANIFEST_NAME
    run.bundle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(carried, run.bundle_dir / MANIFEST_NAME)
    fetched = stage_bundle(run.bundle_dir, allow_download=True)
    return {"fetched": fetched, **verify_bundle(run.bundle_dir)}


# --------------------------------------------------------------------------------------------------
# shared steps
# --------------------------------------------------------------------------------------------------


def load_records(run: Run, stage: str) -> dict[str, list[dict[str, Any]]]:
    """The sample records recorded by `prepare`; refuses if the pinned sample manifest changed since."""
    from mediapipe_face_landmarker_pipeline import manifest_digest, sample_records

    data = run.read_state("data.json", stage)
    if manifest_digest() != data["sample_manifest_sha256"]:
        raise RuntimeError("the sample manifest changed since 'prepare'; re-run from Section 4")
    return sample_records(run.sample_dir)


def by_id(records: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    return {rec["id"]: rec for group in records.values() for rec in group}


def predict_subset(faces: list[dict[str, Any]], width: int, height: int) -> Any:
    """The 14 evaluated MediaPipe points of the first face, in pixels (None when no face was found)."""
    from mediapipe_face_landmarker_pipeline import MEDIAPIPE_INDICES, pixel_coordinates

    if not faces:
        return None
    return pixel_coordinates(faces[0]["landmarks"], width, height)[list(MEDIAPIPE_INDICES), :2]


def export_faces(prefix: Path, items: list[tuple[str, Any, list[dict[str, Any]]]]) -> dict[str, str]:
    """Write the output contract for (image_id, image, faces) items: landmarks CSV, blendshapes CSV, matrices JSON."""
    from mediapipe_face_landmarker_pipeline import (
        BLENDSHAPE_FIELDS,
        LANDMARK_FIELDS,
        blendshape_rows,
        landmark_rows,
        matrix_records,
        write_csv,
        write_json,
    )

    landmarks, blendshapes, matrices = [], [], []
    for image_id, image, faces in items:
        landmarks += landmark_rows(image_id, faces, *image.size)
        blendshapes += blendshape_rows(image_id, faces)
        matrices += matrix_records(image_id, faces)
    paths = {
        "landmarks_csv": write_csv(landmarks, prefix.with_name(prefix.name + "_landmarks.csv"), LANDMARK_FIELDS),
        "blendshapes_csv": write_csv(blendshapes, prefix.with_name(prefix.name + "_blendshapes.csv"), BLENDSHAPE_FIELDS),
        "matrices_json": write_json(matrices, prefix.with_name(prefix.name + "_matrices.json")),
    }
    return {k: str(v) for k, v in paths.items()}


def perturbation_table(run: Run, landmarker: Any, faces_records: list[dict[str, Any]], settings: list[tuple[str, float]]) -> list[dict[str, Any]]:
    """Detection rate, landmark consistency and annotation NME for each perturbation setting on the same faces."""
    import numpy as np

    from mediapipe_face_landmarker_pipeline import (
        EYE_CORNERS,
        FAILURE_NME,
        apply_affine,
        inter_ocular_distance,
        invert_affine,
        nme_iod,
        perturb,
        read_template,
        template_subset,
        validate_image,
    )

    base = []
    for rec in faces_records:
        image, _ = validate_image(rec["image_path"], image_id=rec["id"])
        truth = template_subset(read_template(rec["template_path"]))
        reference = predict_subset(landmarker.detect(image), *image.size)
        base.append((image, truth, inter_ocular_distance(truth, EYE_CORNERS), reference))
    rows = []
    for kind, value in settings:
        detected, failures, consistency, annotation = 0, 0, [], []
        for image, truth, iod, reference in base:
            changed, forward = perturb(image, kind, value)
            predicted = predict_subset(landmarker.detect(changed), *changed.size)
            if predicted is None:
                continue
            detected += 1
            back = apply_affine(invert_affine(forward), predicted)  # into original pixel coordinates
            annotation.append(nme_iod(back, truth, iod))
            failures += annotation[-1] > FAILURE_NME
            if reference is not None:
                consistency.append(nme_iod(back, reference, iod))
        rows.append(
            {
                "perturbation": kind,
                "value": value,
                "faces": len(base),
                "detection_rate": r(detected / len(base)),
                "failure_rate_at_0.10": r(failures / len(base)),
                "consistency_nme_mean": r(np.mean(consistency)) if consistency else None,
                "annotation_nme_mean": r(np.mean(annotation)) if annotation else None,
            }
        )
    return rows


# --------------------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------------------


def stage_weights(run: Run) -> None:
    """Section 3: install the carried manifest, fetch the bundle from the pinned generation if absent, verify it."""
    record = stage_files(run)
    print({"model_id": record["model_id"], "version": record["version"], "revision (object generation)": record["revision"], "bytes": record["bytes"], "sha256": record["sha256"], "fetched": record["fetched"]}, flush=True)
    for member in record["members"]:
        print({"bundle_member": member["path"], "bytes": member["bytes"], "compression": member["compression"], "sha256": member["sha256"][:16] + "…", "verified": True}, flush=True)
    run.write_output("weights.json", record)


def stage_prepare(run: Run) -> None:
    """Section 4: fetch and digest-verify the samples, validate every image, demonstrate the refusals."""
    from PIL import Image

    from mediapipe_face_landmarker_pipeline import (
        INPUT_SCHEMA,
        age_band,
        load_manifest,
        manifest_digest,
        read_template,
        sample_records,
        validate_image,
        validate_num_faces,
        write_csv,
    )

    manifest = load_manifest()
    print({"ceilings": INPUT_SCHEMA}, flush=True)
    fetched = fetch(run)
    print({"samples": fetched, "source": manifest["base_url"], "licence": "CC BY 4.0 (Face Research Lab London Set; Young adult composite faces)"}, flush=True)
    records = sample_records(run.sample_dir, manifest)
    rows = []
    for group in ("neutral", "smiling", "composite"):
        for rec in records[group]:
            _, report = validate_image(rec["image_path"], image_id=rec["id"])
            if "template_path" in rec:
                read_template(rec["template_path"])  # 189 finite points or a refusal naming the file
            info = rec.get("info", {})
            rows.append({"id": rec["id"], "set": group, "sha256": rec["sha256"], "width": report["width"], "height": report["height"], "source_mode": report["source_mode"], "changes": "; ".join(report["changes"]) or "none", "annotated": "template_path" in rec, "gender": info.get("gender", ""), "ethnicity": info.get("ethnicity", ""), "age_band": age_band(info.get("age")) if info else ""})
    counts = {group: len(records[group]) for group in records}
    sizes = sorted({(row["width"], row["height"]) for row in rows})
    print({"validated": len(rows), "per_set": counts, "image_sizes": sizes[:5], "annotated_with_templates": sum(row["annotated"] for row in rows), "conversions": sorted({row["changes"] for row in rows})}, flush=True)
    write_csv(rows, run.out / f"{STEM}_sample_manifest.csv", list(rows[0]))
    probes = {}
    tiny = Image.new("RGB", (40, 40), (128, 128, 128))
    grey = Image.new("L", (256, 256), 128)
    for name, attempt in (
        ("text bytes named .jpg", lambda: validate_image(b"this is not an image", image_id="notes.jpg")),
        ("40 x 40 image", lambda: validate_image(tiny, image_id="tiny.png")),
        ("9000 x 100 image", lambda: validate_image(Image.new("RGB", (9000, 100)), image_id="banner.png")),
        ("num_faces = 0", lambda: validate_num_faces(0)),
    ):
        try:
            attempt()
            probes[name] = "accepted (unexpected)"
        except ValueError as exc:
            probes[name] = f"rejected: {exc}"
    _, grey_report = validate_image(grey, image_id="greyscale.png")
    probes["greyscale 256 x 256 image"] = f"accepted with a reported change: {grey_report['changes']}"
    for name, outcome in probes.items():
        print({"probe": name, "outcome": outcome}, flush=True)
    if any(outcome.startswith("accepted (unexpected)") for outcome in probes.values()):
        raise RuntimeError("a refusal probe was accepted; the validator is broken")
    run.write_state("data.json", {"sample_manifest_sha256": manifest_digest(), "counts": counts})
    run.write_output("prepare.json", {"samples": fetched, "counts": counts, "probes": probes})


def stage_inference(run: Run) -> None:
    """Section 5: the task on one photograph, the full output contract, then num_faces and a no-face image."""
    import numpy as np
    from PIL import Image

    from mediapipe_face_landmarker_pipeline import (
        MEDIAPIPE_INDICES,
        NUM_BLENDSHAPES,
        SCORE_SEMANTICS,
        draw_overlay,
        face_summary,
        landmark_rows,
        validate_image,
    )

    recs = by_id(load_records(run, "inference"))
    rec = recs[WALKTHROUGH_ID]
    image, report = validate_image(rec["image_path"], image_id=rec["id"])
    print({"input": report}, flush=True)
    with make_landmarker(run, num_faces=1) as landmarker:
        started = time.perf_counter()
        faces = landmarker.detect(image)
        seconds = time.perf_counter() - started
        config = landmarker.config
    print({"config": config, "faces_found": len(faces), "inference_ms": round(seconds * 1000, 1)}, flush=True)
    if not faces:
        raise RuntimeError("no face found on the walkthrough photograph; the bundle or runtime is not working")
    face = faces[0]
    print({"landmarks_shape": list(face["landmarks"].shape), "blendshapes": len(face["blendshapes"]), "matrix_shape": list(face["transformation_matrix"].shape)}, flush=True)
    for row in landmark_rows(rec["id"], faces, *image.size)[:3]:
        print({"landmark_row": row}, flush=True)
    print({"summary": face_summary(faces, *image.size)}, flush=True)
    print({"transformation_matrix": np.round(face["transformation_matrix"], 3).tolist()}, flush=True)
    print({"score_semantics": SCORE_SEMANTICS}, flush=True)
    paths = export_faces(run.out / f"{STEM}_walkthrough", [(rec["id"], image, faces)])
    draw_overlay(image, faces, highlight=MEDIAPIPE_INDICES).save(run.out / f"{STEM}_walkthrough_overlay.jpg", quality=90)
    print({"written": paths, "overlay": str(run.out / f"{STEM}_walkthrough_overlay.jpg")}, flush=True)
    # num_faces: a collage of two composite faces, then a blank image
    left, _ = validate_image(recs[COLLAGE_IDS[0]]["image_path"])
    right, _ = validate_image(recs[COLLAGE_IDS[1]]["image_path"])
    height = 600
    left = left.resize((round(left.size[0] * height / left.size[1]), height))
    right = right.resize((round(right.size[0] * height / right.size[1]), height))
    collage = Image.new("RGB", (left.size[0] + right.size[0], height))
    collage.paste(left, (0, 0))
    collage.paste(right, (left.size[0], 0))
    counts = {}
    for requested in (1, 2):
        with make_landmarker(run, num_faces=requested) as landmarker:
            found = landmarker.detect(collage)
        counts[f"num_faces={requested}"] = len(found)
        if requested == 2:
            draw_overlay(collage, found).save(run.out / f"{STEM}_two_faces_overlay.jpg", quality=90)
    with make_landmarker(run, num_faces=1) as landmarker:
        counts["blank grey image"] = len(landmarker.detect(Image.new("RGB", (640, 480), (128, 128, 128))))
    print({"faces_returned": counts, "collage": f"{COLLAGE_IDS[0]} + {COLLAGE_IDS[1]}"}, flush=True)
    run.write_output("inference.json", {"input": report, "config": config, "faces_found": len(faces), "inference_ms": round(seconds * 1000, 1), "num_faces_demo": counts, "blendshapes_per_face": NUM_BLENDSHAPES, "outputs": paths})


def stage_evaluate(run: Run) -> None:
    """Section 6: NME against the annotated templates of 102 neutral photographs, two baselines, groups."""
    import numpy as np

    from mediapipe_face_landmarker_pipeline import (
        EYE_CORNERS,
        LANDMARK_MAP,
        MEDIAPIPE_INDICES,
        age_band,
        bootstrap_ci,
        contact_sheet,
        draw_overlay,
        inter_ocular_distance,
        mean_shape_box,
        mean_shape_image,
        nme_iod,
        point_errors,
        read_template,
        summarize_nme,
        template_subset,
        validate_image,
        write_csv,
    )

    records = load_records(run, "evaluate")["neutral"]
    truths, iods, preds, boxes, faces_kept, images = [], [], [], [], [], []
    started = time.perf_counter()
    with make_landmarker(run, num_faces=1) as landmarker:
        for rec in records:
            image, _ = validate_image(rec["image_path"], image_id=rec["id"])
            truth = template_subset(read_template(rec["template_path"]))
            faces = landmarker.detect(image)
            truths.append(truth)
            iods.append(inter_ocular_distance(truth, EYE_CORNERS))
            preds.append(predict_subset(faces, *image.size))
            found = face_boxes(run, image)
            boxes.append(max(found, key=lambda b: b["score"]) if found else None)
            faces_kept.append(faces)
            images.append(image.size)
    seconds = time.perf_counter() - started
    model_nme = [None if p is None else nme_iod(p, t, d) for p, t, d in zip(preds, truths, iods, strict=True)]
    image_baseline = [nme_iod(p, t, d) for p, t, d in zip(mean_shape_image(truths), truths, iods, strict=True)]
    box_baseline = [None if p is None else nme_iod(p, t, d) for p, t, d in zip(mean_shape_box(truths, boxes), truths, iods, strict=True)]
    summary = {"model": summarize_nme(model_nme), "baseline_mean_shape_image": summarize_nme(image_baseline), "baseline_mean_shape_detector_box": summarize_nme(box_baseline)}
    detected_nme = [v for v in model_nme if v is not None]
    summary["model"]["nme_mean_bootstrap_95ci"] = bootstrap_ci(detected_nme, seed=BOOTSTRAP_SEED)
    for name, values in summary.items():
        print({name: values}, flush=True)
    errors = np.array([point_errors(p, t) / d for p, t, d in zip(preds, truths, iods, strict=True) if p is not None])
    per_point = [{"point": name, "template_index": t, "mediapipe_index": m, "mean_error_iod": r(errors[:, i].mean())} for i, (name, t, m) in enumerate(LANDMARK_MAP)]
    for row in sorted(per_point, key=lambda x: -x["mean_error_iod"]):
        print({"per_point": row}, flush=True)
    groups: dict[str, dict[str, list[float]]] = {"gender": {}, "ethnicity": {}, "age_band": {}}
    for rec, value in zip(records, model_nme, strict=True):
        if value is None:
            continue
        info = rec["info"]
        groups["gender"].setdefault(info["gender"], []).append(value)
        groups["ethnicity"].setdefault(info["ethnicity"], []).append(value)
        groups["age_band"].setdefault(age_band(info["age"]), []).append(value)
    group_table = {factor: {g: {"n": len(v), "nme_mean": r(np.mean(v)), "nme_median": r(np.median(v))} for g, v in sorted(table.items())} for factor, table in groups.items()}
    for factor, table in group_table.items():
        print({"group_factor": factor, "groups": table}, flush=True)
    rows = []
    for rec, value, img_b, box_b, d in zip(records, model_nme, image_baseline, box_baseline, iods, strict=True):
        rows.append({"id": rec["id"], "detected": value is not None, "nme_model": "" if value is None else r(value, 5), "nme_baseline_image": r(img_b, 5), "nme_baseline_box": "" if box_b is None else r(box_b, 5), "iod_px": r(d, 2), "gender": rec["info"]["gender"], "ethnicity": rec["info"]["ethnicity"], "age_band": age_band(rec["info"]["age"])})
    write_csv(rows, run.out / f"{STEM}_evaluation_per_image.csv", list(rows[0]))
    # figure: cumulative error distribution of the model and the two baselines
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, values in (("MediaPipe Face Landmarker", detected_nme), ("mean shape in detector box", [v for v in box_baseline if v is not None]), ("mean shape in image", image_baseline)):
        xs = np.sort(values)
        ax.step(xs, np.arange(1, xs.size + 1) / len(records), where="post", label=label)
    ax.set_xlim(0, 0.25)
    ax.set_xlabel("NME (mean point error / inter-ocular distance)")
    ax.set_ylabel("fraction of the 102 faces")
    ax.set_title("Cumulative error distribution, 14 annotated points")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run.out / f"{STEM}_ced.png", dpi=110)
    plt.close(fig)
    # figure: three typical faces and the three largest errors, prediction (orange rings) and annotation (cyan crosses)
    order = np.argsort([np.inf if v is None else v for v in model_nme])
    finite = [i for i in order if model_nme[i] is not None]
    chosen = finite[len(finite) // 2 - 1 : len(finite) // 2 + 2] + finite[-3:]
    tiles, labels = [], []
    for i in chosen:
        image, _ = validate_image(records[i]["image_path"])
        tiles.append(draw_overlay(image, faces_kept[i], highlight=MEDIAPIPE_INDICES, ground_truth=truths[i], max_side=600).crop((120, 160, 480, 520)))
        labels.append(f"{records[i]['face_id']} NME {model_nme[i]:.3f}")
    contact_sheet(tiles, columns=3, tile=320, labels=labels).save(run.out / f"{STEM}_evaluation_examples.jpg", quality=90)
    record = {"protocol": "every neutral London Set photograph, one pass, no selection; NME over 14 annotated points, normalised by the annotated inter-ocular distance (eye-corner midpoints); baselines leave each face out of its own mean", "faces": len(records), "summary": summary, "per_point": per_point, "groups": group_table, "seconds": round(seconds, 1), "image_sizes": sorted({tuple(s) for s in images})}
    print({"seconds": round(seconds, 1), "written": [f"{STEM}_evaluation_per_image.csv", f"{STEM}_ced.png", f"{STEM}_evaluation_examples.jpg"]}, flush=True)
    run.write_output("evaluation.json", record)


def stage_robustness(run: Run) -> None:
    """Section 7: controlled perturbations on 20 annotated faces - detection, consistency and annotation NME."""
    from mediapipe_face_landmarker_pipeline import (
        MEDIAPIPE_INDICES,
        contact_sheet,
        draw_overlay,
        perturb,
        validate_image,
        write_csv,
    )

    subset = load_records(run, "robustness")["neutral"][:ROBUSTNESS_FACES]
    settings = [("none", 0.0), *PERTURBATIONS]
    with make_landmarker(run, num_faces=1) as landmarker:
        table = perturbation_table(run, landmarker, subset, settings)
        tiles, labels = [], []
        image, _ = validate_image(subset[0]["image_path"])
        for kind, value in (("rotate", 45.0), ("scale", 0.12), ("blur", 12.0), ("brightness", 0.35)):
            changed, _ = perturb(image, kind, value)
            tiles.append(draw_overlay(changed, landmarker.detect(changed), highlight=MEDIAPIPE_INDICES, max_side=480))
            labels.append(f"{kind} {value:g}")
    for row in table:
        print(row, flush=True)
    write_csv(table, run.out / f"{STEM}_robustness.csv", list(table[0]))
    contact_sheet(tiles, columns=4, tile=300, labels=labels).save(run.out / f"{STEM}_robustness_examples.jpg", quality=90)
    run.write_output("robustness.json", {"faces": [rec["id"] for rec in subset], "table": table, "definition": "consistency NME: prediction on the perturbed image mapped back to original pixels vs the prediction on the original, divided by the annotated IOD; annotation NME: the mapped-back prediction vs the annotation"})


def stage_blendshapes(run: Run) -> None:
    """Section 8: paired neutral vs smiling photographs of the same 102 people - a sanity check of blendshape scores."""
    import numpy as np

    from mediapipe_face_landmarker_pipeline import BLENDSHAPE_NAMES, SCORE_SEMANTICS, paired_sanity, validate_image, write_csv

    records = load_records(run, "blendshapes")
    smiling = {rec["face_id"]: rec for rec in records["smiling"]}
    pairs = [(rec, smiling[rec["face_id"]]) for rec in records["neutral"] if rec["face_id"] in smiling]
    scores: dict[str, list[list[float] | None]] = {"neutral": [], "smiling": []}
    with make_landmarker(run, num_faces=1) as landmarker:
        for neutral, smile in pairs:
            for label, rec in (("neutral", neutral), ("smiling", smile)):
                image, _ = validate_image(rec["image_path"], image_id=rec["id"])
                faces = landmarker.detect(image)
                scores[label].append(faces[0]["blendshapes"] if faces else None)
    usable = [i for i in range(len(pairs)) if scores["neutral"][i] is not None and scores["smiling"][i] is not None]
    index = {name: BLENDSHAPE_NAMES.index(name) for name in (*PAIRED_HYPOTHESIS, *DESCRIPTIVE_BLENDSHAPES)}

    def series(label: str, names: tuple[str, ...]) -> list[float]:
        return [float(np.mean([scores[label][i][index[n]] for n in names])) for i in usable]

    hypothesis = paired_sanity(series("neutral", PAIRED_HYPOTHESIS), series("smiling", PAIRED_HYPOTHESIS))
    print({"pairs_with_a_face_in_both": len(usable), "of": len(pairs)}, flush=True)
    print({"stated_expectation": "mean(mouthSmileLeft, mouthSmileRight) is higher on the smiling photograph than on the neutral one", **hypothesis}, flush=True)
    descriptive = {}
    for left, right in (("cheekSquintLeft", "cheekSquintRight"), ("eyeSquintLeft", "eyeSquintRight"), ("eyeBlinkLeft", "eyeBlinkRight")):
        descriptive[left.replace("Left", "")] = paired_sanity(series("neutral", (left, right)), series("smiling", (left, right)))
    for single in ("jawOpen", "mouthClose"):
        descriptive[single] = paired_sanity(series("neutral", (single,)), series("smiling", (single,)))
    for name, values in descriptive.items():
        print({"descriptive": name, "median_neutral": values["median_neutral"], "median_smiling": values["median_expressive"], "fraction_smiling_higher": values["fraction_expressive_higher"], "rank_auc": values["rank_auc"]}, flush=True)
    print({"score_semantics": SCORE_SEMANTICS}, flush=True)
    rows = []
    for i in usable:
        neutral, smile = pairs[i]
        row = {"face_id": neutral["face_id"]}
        for label in ("neutral", "smiling"):
            for name, j in index.items():
                row[f"{label}_{name}"] = r(scores[label][i][j])
        rows.append(row)
    write_csv(rows, run.out / f"{STEM}_blendshapes_paired.csv", list(rows[0]))
    import matplotlib.pyplot as plt

    neutral_values, smiling_values = series("neutral", PAIRED_HYPOTHESIS), series("smiling", PAIRED_HYPOTHESIS)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for a, b in zip(neutral_values, smiling_values, strict=True):
        ax.plot([0, 1], [a, b], color="0.6" if b > a else "tab:red", alpha=0.5, linewidth=0.8)
    ax.scatter(np.zeros(len(neutral_values)), neutral_values, s=10, color="tab:blue", zorder=3)
    ax.scatter(np.ones(len(smiling_values)), smiling_values, s=10, color="tab:orange", zorder=3)
    ax.set_xticks([0, 1], ["neutral", "smiling"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("mean of mouthSmileLeft / mouthSmileRight")
    ax.set_title("Same person, two photographs (red: score fell)")
    fig.tight_layout()
    fig.savefig(run.out / f"{STEM}_smile_pairs.png", dpi=110)
    plt.close(fig)
    run.write_output("blendshapes.json", {"pairs": len(pairs), "usable_pairs": len(usable), "hypothesis": {"blendshapes": list(PAIRED_HYPOTHESIS), **hypothesis}, "descriptive": descriptive, "score_semantics": SCORE_SEMANTICS})


def stage_newdata(run: Run) -> None:
    """Section 9: new images (ten composite faces), the full export, and VIDEO mode on a short frame sequence."""
    import numpy as np

    from mediapipe_face_landmarker_pipeline import (
        EYE_CORNERS,
        MEDIAPIPE_INDICES,
        apply_affine,
        contact_sheet,
        draw_overlay,
        face_summary,
        inter_ocular_distance,
        nme_iod,
        perturb,
        read_template,
        summarize_nme,
        template_subset,
        validate_image,
    )

    records = load_records(run, "newdata")
    composites = records["composite"]
    items, tiles, labels, nmes = [], [], [], []
    with make_landmarker(run, num_faces=1) as landmarker:
        for rec in composites:
            image, _ = validate_image(rec["image_path"], image_id=rec["id"])
            faces = landmarker.detect(image)
            items.append((rec["id"], image, faces))
            truth = template_subset(read_template(rec["template_path"]))
            predicted = predict_subset(faces, *image.size)
            nmes.append(None if predicted is None else nme_iod(predicted, truth, inter_ocular_distance(truth, EYE_CORNERS)))
            tiles.append(draw_overlay(image, faces, highlight=MEDIAPIPE_INDICES, max_side=500))
            labels.append(rec["face_id"])
            summary = face_summary(faces, *image.size, top=3)
            print({"image_id": rec["id"], "faces": len(faces), "top_blendshapes": summary[0]["top_blendshapes"] if summary else {}, "nme_vs_template": None if nmes[-1] is None else r(nmes[-1])}, flush=True)
    paths = export_faces(run.out / f"{STEM}_newdata", items)
    contact_sheet(tiles, columns=5, tile=260, labels=labels).save(run.out / f"{STEM}_newdata_overlays.jpg", quality=90)
    composite_summary = summarize_nme(nmes)
    print({"composite_nme": composite_summary, "written": paths}, flush=True)
    # VIDEO mode: a 30-frame sequence made by rotating and shifting one composite face; the true motion is known
    rec = by_id(records)[VIDEO_ID]
    image, _ = validate_image(rec["image_path"])
    # the composite's own annotation (14 mapped points), scaled to the 600 x 600 frame: the reference both modes are scored against
    truth = template_subset(read_template(rec["template_path"])) * np.array([600 / image.size[0], 600 / image.size[1]])
    image = image.resize((600, 600))
    with make_landmarker(run, num_faces=1) as still:
        still_prediction = predict_subset(still.detect(image), *image.size)
    if still_prediction is None:
        raise RuntimeError("no face on the VIDEO-mode source image")
    iod = inter_ocular_distance(truth, EYE_CORNERS)
    frames = []
    for k in range(VIDEO_FRAMES):
        angle = 8.0 * np.sin(2 * np.pi * k / VIDEO_FRAMES)
        frames.append(perturb(image, "roll", float(angle)))
    video = {}
    for mode in ("IMAGE", "VIDEO"):
        vs_motion, vs_still, residuals = [], [], []
        with make_landmarker(run, num_faces=1, running_mode=mode) as landmarker:
            for k, (frame, forward) in enumerate(frames):
                faces = landmarker.detect(frame) if mode == "IMAGE" else landmarker.detect_frame(frame, int(k * 1000 / VIDEO_FPS) + 1)
                predicted = predict_subset(faces, *frame.size)
                if predicted is None:
                    residuals.append(None)
                    continue
                expected = apply_affine(forward, truth)  # the annotation moved by the true (known) motion
                vs_motion.append(nme_iod(predicted, expected, iod))
                vs_still.append(nme_iod(predicted, apply_affine(forward, still_prediction), iod))
                residuals.append(predicted - expected)
        # jitter: frame-to-frame displacement of each landmark after the known motion is removed, / IOD (consecutive tracked frames only)
        steps = [float(np.linalg.norm(b - a, axis=1).mean() / iod) for a, b in zip(residuals, residuals[1:], strict=False) if a is not None and b is not None]
        video[mode] = {
            "frames": VIDEO_FRAMES,
            "tracked": len(vs_motion),
            "nme_vs_true_motion_mean": r(np.mean(vs_motion)) if vs_motion else None,
            "jitter_after_motion_removed": r(np.mean(steps)) if steps else None,
            "self_consistency_nme_mean": r(np.mean(vs_still)) if vs_still else None,
        }
    definitions = {
        "nme_vs_true_motion_mean": "prediction vs the composite's annotation moved by the known rotation, / annotated IOD (the true motion)",
        "jitter_after_motion_removed": "mean frame-to-frame change of each landmark's error vector (prediction minus moved annotation), / annotated IOD",
        "self_consistency_nme_mean": "prediction vs the IMAGE-mode still prediction moved by the known rotation, / annotated IOD (IMAGE mode is compared with itself, so it favours IMAGE mode by construction)",
        "still_prediction_nme_vs_annotation": r(nme_iod(still_prediction, truth, iod)),
    }
    print({"video_mode_demo": video, "motion": "in-plane rotation of ±8° over 30 frames at 30 fps", "reference": "the composite's own annotation moved by the known rotation", "still_prediction_nme_vs_annotation": definitions["still_prediction_nme_vs_annotation"]}, flush=True)
    run.write_output("newdata.json", {"images": [rec["id"] for rec in composites], "composite_nme": composite_summary, "per_image": [{"image_id": i[0], "faces": len(i[2]), "nme_vs_template": None if n is None else r(n, 5)} for i, n in zip(items, nmes, strict=True)], "outputs": paths, "video_mode": video, "video_mode_definitions": definitions})


def stage_export(run: Run) -> None:
    """Section 10: one provenance record for the run, and the list of every file written."""
    from mediapipe_face_landmarker_pipeline import (
        ARCHITECTURE_SOURCE,
        MODEL_ID,
        MODEL_LICENSE,
        MODEL_REVISION,
        MODEL_SHA256,
        MODEL_VERSION,
        bundle_url,
        load_manifest,
        manifest_digest,
    )
    from mediapipe_face_landmarker_pipeline.pipeline import sha256_file

    weights = run.read_output("weights.json", "export")
    evaluation = run.read_output("evaluation.json", "export")
    robustness = run.read_output("robustness.json", "export")
    blendshapes = run.read_output("blendshapes.json", "export")
    newdata = run.read_output("newdata.json", "export")
    inference = run.read_output("inference.json", "export")
    source = json.loads((run.root / "source.json").read_text(encoding="utf-8")) if (run.root / "source.json").is_file() else {}
    manifest = load_manifest()
    result = {
        "model": {"id": MODEL_ID, "version": MODEL_VERSION, "revision": MODEL_REVISION, "sha256": MODEL_SHA256, "url": bundle_url(), "architecture_source": ARCHITECTURE_SOURCE, "license": MODEL_LICENSE, "bundle_members": weights["members"], "serialization": "MediaPipe task bundle (stored zip of TFLite flatbuffers + binary protobuf); nothing unpickled", "remote_code_executed": False},
        "notebook_source": source,
        "runtime": runtime_versions(),
        "inference_config": inference["config"],
        "samples": {"repository": manifest["repository"], "commit": manifest["commit"], "sample_manifest_sha256": manifest_digest(), "sets": manifest["sets"]},
        "evaluation": {"faces": evaluation["faces"], "summary": evaluation["summary"], "per_point": evaluation["per_point"], "groups": evaluation["groups"]},
        "robustness": robustness["table"],
        "blendshape_sanity": {"hypothesis": blendshapes["hypothesis"], "descriptive": blendshapes["descriptive"], "score_semantics": blendshapes["score_semantics"]},
        "new_data": {"composite_nme": newdata["composite_nme"], "video_mode": newdata["video_mode"], "video_mode_definitions": newdata["video_mode_definitions"]},
        "evidence_label": "tutorial / sanity evidence on public sample faces; not a benchmark and not a fairness audit",
        "run_id": run.root.name,
        "stage_seconds": stage_timings(run),
        "optional_branches": {"byod": (run.out / "byod" / "byod_result.json").is_file(), "activity": (run.out / "activity.json").is_file()},
    }
    name = f"{STEM}_result.json"
    # every file the run wrote to outputs/ (except this record), with its full SHA-256: the record binds the files it describes
    result["files"] = [{"file": item.relative_to(run.out).as_posix(), "bytes": item.stat().st_size, "sha256": sha256_file(item)} for item in sorted(run.out.rglob("*")) if item.is_file() and item.relative_to(run.out).as_posix() != name]
    path = run.write_output(name, result)
    print({"result": str(path), "run_id": result["run_id"], "model": result["model"]["id"], "revision": MODEL_REVISION, "sha256": MODEL_SHA256[:16] + "…", "runtime": result["runtime"]}, flush=True)
    for row in result["files"]:
        print({**row, "sha256": row["sha256"][:16] + "…"}, flush=True)


def stage_timings(run: Run) -> dict[str, float]:
    """Seconds per completed stage of this run (written by ``main`` to ``state/timings.json``)."""
    path = run.state / "timings.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _byod_inputs(path: Path, folder: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    """``([(input name, file)], [skipped zip members])`` for a single image, or a zip of images (flat or nested; unsafe
    members refused; members without an image extension skipped and reported; nothing executable is read).

    Zip members are extracted into ``folder``, a scratch directory outside ``outputs/``; the input name is the member
    path inside the archive, so every message names the file the user put in the zip."""
    if not path.is_file():
        raise ValueError(f"BYOD_PATH {path} does not exist; upload a file or set the path to an image or a zip already in the runtime")
    if path.suffix.lower() != ".zip":
        return [(path.name, path)], []
    if path.stat().st_size > BYOD_MAX_ZIP_BYTES:
        raise ValueError(f"BYOD zip is {path.stat().st_size:,} bytes, above {BYOD_MAX_ZIP_BYTES:,}; split it")
    out, skipped = [], []
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        for member in members:
            name = member.filename
            if name.startswith(("/", "\\")) or ".." in Path(name).parts or ":" in name:
                raise ValueError(f"BYOD zip has an unsafe member path {name!r}; refusing the archive")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"BYOD zip has a symlink member {name!r}; refusing the archive")
        visible = [m for m in members if not Path(m.filename).name.startswith(".") and "__MACOSX" not in m.filename]
        images = [m for m in visible if Path(m.filename).suffix.lower() in BYOD_IMAGE_SUFFIXES]
        skipped = [m.filename for m in visible if m not in images]
        if not images:
            raise ValueError(f"BYOD zip {path.name} holds no image files (looked for {', '.join(BYOD_IMAGE_SUFFIXES)}; skipped {skipped or 'nothing'})")
        if len(images) > BYOD_MAX_IMAGES:
            raise ValueError(f"BYOD zip holds {len(images)} image files; at most {BYOD_MAX_IMAGES} images are accepted per run")
        if sum(m.file_size for m in images) > BYOD_MAX_ZIP_BYTES:
            raise ValueError("BYOD zip expands beyond the size ceiling; split it")
        for k, member in enumerate(images):
            target = folder / f"{k:02d}{Path(member.filename).suffix.lower()}"
            target.write_bytes(archive.read(member))
            out.append((member.filename, target))
    return out, skipped


def stage_byod(run: Run) -> None:
    """Section 11 (optional): the same validation, inference and output contract on the reader's own image(s)."""
    from mediapipe_face_landmarker_pipeline import (
        INPUT_SCHEMA,
        MEDIAPIPE_INDICES,
        MODEL_REVISION,
        MODEL_SHA256,
        contact_sheet,
        draw_overlay,
        face_summary,
        validate_image,
        validate_num_faces,
    )

    num_faces = validate_num_faces(run.options.num_faces)
    folder = run.out / "byod"
    scratch = run.root / "byod_inputs"  # extracted zip members: outside outputs/, so outputs/byod holds only results
    for directory in (folder, scratch):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    inputs, skipped = _byod_inputs(Path(run.options.byod), scratch)
    print({"contract": INPUT_SCHEMA, "num_faces": num_faces, "inputs": len(inputs), "skipped_non_image_members": skipped}, flush=True)
    validated = []
    for name, path in inputs:
        image, report = validate_image(path, image_id=name)  # a refusal stops the stage with the rule that failed, naming the user's file
        validated.append((name, image, report))
        print({"validated": report}, flush=True)
    items, tiles, labels = [], [], []
    with make_landmarker(run, num_faces=num_faces) as landmarker:
        for name, image, _report in validated:
            faces = landmarker.detect(image)
            items.append((name, image, faces))
            tiles.append(draw_overlay(image, faces, highlight=MEDIAPIPE_INDICES, max_side=600))
            labels.append(name[:30])
            print({"image_id": name, "faces": len(faces), "summary": face_summary(faces, *image.size, top=3)}, flush=True)
    paths = export_faces(folder / "byod", items)
    contact_sheet(tiles, columns=min(4, len(tiles)), tile=360, labels=labels).save(folder / "byod_overlays.jpg", quality=90)
    record = {"run_id": run.root.name, "num_faces": num_faces, "inputs": [report for _, _, report in validated], "skipped_non_image_members": skipped, "faces_per_image": {name: len(faces) for name, _, faces in items}, "outputs": paths, "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256, "runtime": runtime_versions(), "data_left_runtime": False}
    run.write_output("byod/byod_result.json", record)
    print({"written": [*paths.values(), str(folder / "byod_overlays.jpg"), str(folder / "byod_result.json")], "data_left_runtime": False}, flush=True)


def stage_activity(run: Run) -> None:
    """Section 12 (optional): change one thing - the rotation angle - and compare with Section 7."""
    angle = float(run.options.rotation)
    if not -180.0 <= angle <= 180.0:
        raise ValueError("ACTIVITY_ROTATION must be within -180..180 degrees")
    subset = load_records(run, "activity")["neutral"][:ROBUSTNESS_FACES]
    previous = {(row["perturbation"], row["value"]): row for row in run.read_output("robustness.json", "activity")["table"]}
    with make_landmarker(run, num_faces=1) as landmarker:
        row = perturbation_table(run, landmarker, subset, [("rotate", angle)])[0]
    print({"unperturbed (Section 7)": previous[("none", 0.0)]}, flush=True)
    for value in (15.0, 30.0, 45.0, 90.0, 180.0):
        print({f"rotate {value:g} (Section 7)": previous[("rotate", value)]}, flush=True)
    print({f"rotate {angle:g} (your run)": row}, flush=True)
    run.write_output("activity.json", {"rotation": angle, "row": row})


STAGES = {
    "weights": stage_weights,
    "prepare": stage_prepare,
    "inference": stage_inference,
    "evaluate": stage_evaluate,
    "robustness": stage_robustness,
    "blendshapes": stage_blendshapes,
    "newdata": stage_newdata,
    "export": stage_export,
    "byod": stage_byod,
    "activity": stage_activity,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="run directory holding the carried sources")
    parser.add_argument("--weights", type=Path, required=True, help="directory holding the bundle and the sample cache")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--byod", default="", help="byod: an image or a zip of images")
    parser.add_argument("--num-faces", type=int, default=1, help="byod: faces to return per image (1..10)")
    parser.add_argument("--rotation", type=float, default=120.0, help="activity: rotation angle in degrees")
    parser.add_argument("--keep-stderr", action="store_true", help="do not move native log lines to logs/<stage>.native.log")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    root = options.root.resolve()
    carried_src = root / "src"
    if carried_src.is_dir() and str(carried_src) not in sys.path:
        sys.path.insert(0, str(carried_src))
    os.environ.setdefault("MPLBACKEND", "Agg")
    run = Run(root, options.weights.resolve(), options)
    error_file = run.state / f"{options.stage}.error.json"
    error_file.unlink(missing_ok=True)
    # MediaPipe and TFLite write informational native log lines to stderr (for example "Created TensorFlow Lite XNNPACK
    # delegate for CPU"); they go to logs/<stage>.native.log so the learner-facing output stays readable.
    if not options.keep_stderr:
        native_log = root / "logs" / f"{options.stage}.native.log"
        native_log.parent.mkdir(parents=True, exist_ok=True)
        with native_log.open("w", encoding="utf-8") as handle:
            sys.stderr.flush()
            os.dup2(handle.fileno(), 2)
    started = time.perf_counter()
    try:
        STAGES[options.stage](run)
    except Exception as exc:  # the notebook re-raises this message in the kernel
        traceback.print_exc(file=sys.stdout)
        message = str(exc) or repr(exc)
        error_file.write_text(json.dumps({"stage": options.stage, "type": type(exc).__name__, "message": message}), encoding="utf-8")
        print(f"STAGE FAILED ({options.stage}): {type(exc).__name__}: {message}", flush=True)
        return 2
    seconds = round(time.perf_counter() - started, 1)
    timings = stage_timings(run)
    timings[options.stage] = seconds
    run.write_state("timings.json", timings)
    print({"stage": options.stage, "status": "ok", "seconds": seconds}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
