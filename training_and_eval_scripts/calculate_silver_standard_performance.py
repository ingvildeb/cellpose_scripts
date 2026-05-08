"""
Evaluate silver-standard predictions by comparing prediction npy files to
human-corrected npy files, with atlas-aware and animal-aware outputs.

This script:
- does not run Cellpose inference
- compares prediction masks directly to corrected masks using centroid matching
- records atlas region and animal ID for every TP / FP / FN
- calculates IoU for TP pairs only
- reports chunks without corrected masks as unquantifiable

Config usage:
- Copy `training_and_eval_scripts/configs/calculate_silver_standard_performance_config_template.toml`
  to `training_and_eval_scripts/configs/calculate_silver_standard_performance_config_local.toml`.
- Edit `_local.toml` to your preferred settings and run the script.
- If `_local.toml` is missing, the script falls back to `_template.toml`.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import atlaslevels
from atlaslevels.exceptions import ResolutionError

from utils.io_helpers import (
    load_script_config,
    normalize_user_path,
    require_dir,
)
from utils.utils import (
    calculate_iou,
    centroid_inside_gt_matching,
    instance_centroid,
    label_to_random_color,
)


# -------------------------
# CONFIG LOADING
# -------------------------
test_mode = False
cfg = load_script_config(Path(__file__), "calculate_silver_standard_performance_config", test_mode=test_mode)


# -------------------------
# CONFIG PARAMETERS
# -------------------------
prediction_dir = require_dir(normalize_user_path(cfg["prediction_dir"]), "Prediction directory").resolve()
corrected_dir = require_dir(normalize_user_path(cfg["corrected_dir"]), "Corrected directory").resolve()
atlas_dir = require_dir(normalize_user_path(cfg["atlas_dir"]), "Atlas directory").resolve()

output_dir_cfg = str(cfg.get("output_dir", "")).strip()
make_example_plots = bool(cfg.get("make_example_plots", False))
max_example_plots = int(cfg.get("max_example_plots", 5))
plot_file_type = str(cfg.get("plot_file_type", "svg")).strip()
unquantifiable_fraction_threshold = float(cfg.get("unquantifiable_fraction_threshold", 0.2))

advanced_cfg = cfg.get("advanced", {})
if not isinstance(advanced_cfg, dict):
    raise ValueError("Config field 'advanced' must be a TOML table.")

animal_id_index = int(advanced_cfg.get("animal_id_index", 0))
filename_delimiter = str(advanced_cfg.get("filename_delimiter", "_"))
atlaslevels_preset = str(advanced_cfg.get("atlaslevels_preset", "allen_ccfv3")).strip()
atlaslevels_id_map_preset = str(
    advanced_cfg.get("atlaslevels_id_map_preset", "allen_ccfv3_allen_to_kimlab16bit")
).strip()

if max_example_plots < 0:
    raise ValueError("max_example_plots must be >= 0.")
if not (0.0 <= unquantifiable_fraction_threshold <= 1.0):
    raise ValueError("unquantifiable_fraction_threshold must be between 0 and 1.")


# -------------------------
# HELPERS
# -------------------------
BACKGROUND_ATLAS_ID = 0
BACKGROUND_ATLAS_NAME = "background"


def safe_mean(values: list[float]) -> float:
    """Return the mean, or 0.0 for an empty list."""
    return float(np.mean(values)) if values else 0.0


def calculate_classification_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Calculate precision, recall, and F1 from TP / FP / FN counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def mask_rgb(rgb_img: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """Mask an RGB image by a binary mask."""
    out = rgb_img.copy()
    out[~binary_mask.astype(bool)] = 0
    return out


def load_npy_payload(npy_path: Path) -> dict:
    """Load a Cellpose-style npy file and return its dict payload."""
    data = np.load(npy_path, allow_pickle=True).item()
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict-like npy contents in: {npy_path}")
    return data


def load_mask_array(npy_path: Path) -> np.ndarray:
    """Load a Cellpose-style npy file and return its masks array."""
    data = load_npy_payload(npy_path)
    if "masks" not in data:
        raise ValueError(f"Missing 'masks' key in: {npy_path}")
    masks = np.asarray(data["masks"])
    if masks.ndim != 2:
        raise ValueError(f"Expected 2D masks in: {npy_path}, found shape {masks.shape}")
    return masks


def load_raw_image_for_prediction(prediction_file: Path) -> np.ndarray | None:
    """Load the raw tif matching a prediction file by removing the `_seg` suffix."""
    chunk_stem = prediction_file_to_chunk_stem(prediction_file)
    candidate_paths = [
        prediction_dir / f"{chunk_stem}.tif",
        corrected_dir / f"{chunk_stem}.tif",
    ]

    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file():
            image = tifffile.imread(candidate)
            if image.ndim == 2:
                return image
            return np.squeeze(image)

    return None


def prediction_file_to_chunk_stem(prediction_file: Path) -> str:
    """Convert `*_seg.npy` prediction file name to chunk stem without `_seg`."""
    if prediction_file.suffix != ".npy" or not prediction_file.name.endswith("_seg.npy"):
        raise ValueError(f"Prediction file must end with '_seg.npy': {prediction_file.name}")
    return prediction_file.name[: -len("_seg.npy")]


def expected_corrected_path(prediction_file: Path) -> Path:
    """Return corrected file path expected for a prediction file."""
    return corrected_dir / prediction_file.name


def expected_atlas_path(prediction_file: Path) -> Path:
    """Return atlas tif path expected for a prediction file."""
    chunk_stem = prediction_file_to_chunk_stem(prediction_file)
    return atlas_dir / f"{chunk_stem}_atlas.tif"


def parse_animal_id(prediction_file: Path) -> str:
    """Parse animal ID from the chunk stem using delimiter and user-specified index."""
    chunk_stem = prediction_file_to_chunk_stem(prediction_file)
    parts = chunk_stem.split(filename_delimiter)
    if animal_id_index < 0 or animal_id_index >= len(parts):
        raise ValueError(
            "animal_id_index is out of range for file stem.\n"
            f"Stem: {chunk_stem}\n"
            f"Delimiter: {filename_delimiter!r}\n"
            f"Parts: {parts}\n"
            f"animal_id_index: {animal_id_index}"
        )
    return parts[animal_id_index]


def load_atlas_resources_from_atlaslevels(
    preset_name: str,
    id_map_preset_name: str,
) -> tuple[atlaslevels.AtlasOntology, atlaslevels.AtlasIdMap]:
    """Load Allen ontology plus KimLab->Allen ID map from atlaslevels presets."""
    ontology = atlaslevels.load_preset_ontology(preset_name)
    kimlab_to_allen_id_map = atlaslevels.load_preset_id_map(id_map_preset_name).invert()
    if kimlab_to_allen_id_map.atlas_name != ontology.metadata.atlas_name:
        raise ValueError(
            "atlaslevels ontology preset and ID map preset refer to different atlases.\n"
            f"Ontology atlas: {ontology.metadata.atlas_name}\n"
            f"ID map atlas: {kimlab_to_allen_id_map.atlas_name}"
        )
    return ontology, kimlab_to_allen_id_map


def resolve_atlas_label(
    kimlab_atlas_id: int,
    atlas_ontology: atlaslevels.AtlasOntology,
    kimlab_to_allen_id_map: atlaslevels.AtlasIdMap,
) -> tuple[int, int, str]:
    """Resolve a KimLab atlas ID to Allen ontology ID plus region name."""
    if kimlab_atlas_id == BACKGROUND_ATLAS_ID:
        return BACKGROUND_ATLAS_ID, BACKGROUND_ATLAS_ID, BACKGROUND_ATLAS_NAME

    try:
        allen_atlas_id = int(kimlab_to_allen_id_map.convert(kimlab_atlas_id, strict=True))
    except ResolutionError as exc:
        raise ResolutionError(
            f"No KimLab->Allen atlas ID mapping found for atlas chunk value {kimlab_atlas_id}."
        ) from exc

    atlas_name = atlas_ontology.get_name(allen_atlas_id)
    return kimlab_atlas_id, allen_atlas_id, atlas_name


def get_centroid_lookup(label_img: np.ndarray) -> dict[int, tuple[int, int]]:
    """Return instance_id -> centroid coordinate lookup."""
    ids = [int(i) for i in np.unique(label_img) if i != 0]
    return {instance_id: instance_centroid(label_img == instance_id) for instance_id in ids}


def clip_coordinate(row: int, col: int, shape: tuple[int, int]) -> tuple[int, int]:
    """Clip a coordinate into image bounds."""
    height, width = shape
    row = max(0, min(height - 1, int(row)))
    col = max(0, min(width - 1, int(col)))
    return row, col


def atlas_info_from_coordinate(
    atlas_chunk: np.ndarray,
    row: int,
    col: int,
    atlas_ontology: atlaslevels.AtlasOntology,
    kimlab_to_allen_id_map: atlaslevels.AtlasIdMap,
) -> tuple[int, int, str, int, int]:
    """Return atlas info for a coordinate, after clipping into bounds."""
    row, col = clip_coordinate(row, col, atlas_chunk.shape)
    kimlab_atlas_id = int(atlas_chunk[row, col])
    kimlab_atlas_id, allen_atlas_id, atlas_name = resolve_atlas_label(
        kimlab_atlas_id,
        atlas_ontology,
        kimlab_to_allen_id_map,
    )
    return kimlab_atlas_id, allen_atlas_id, atlas_name, row, col


def count_atlas_pixels(
    atlas_chunk: np.ndarray,
    atlas_ontology: atlaslevels.AtlasOntology,
    kimlab_to_allen_id_map: atlaslevels.AtlasIdMap,
) -> list[dict[str, object]]:
    """Count atlas pixels per region for a chunk."""
    ids, counts = np.unique(atlas_chunk, return_counts=True)
    rows: list[dict[str, object]] = []
    for kimlab_atlas_id, pixel_count in zip(ids.tolist(), counts.tolist()):
        kimlab_atlas_id = int(kimlab_atlas_id)
        kimlab_atlas_id, allen_atlas_id, atlas_name = resolve_atlas_label(
            kimlab_atlas_id,
            atlas_ontology,
            kimlab_to_allen_id_map,
        )
        rows.append(
            {
                "atlas_kimlab_id": kimlab_atlas_id,
                "atlas_allen_id": allen_atlas_id,
                "atlas_name": atlas_name,
                "pixels": int(pixel_count),
            }
        )
    return rows


def validate_prediction_set(prediction_files: list[Path]) -> None:
    """Run preflight validation before processing any chunks."""
    if len(prediction_files) == 0:
        raise ValueError(f"No prediction files ending with '_seg.npy' found in: {prediction_dir}")

    missing_atlas_paths: list[Path] = []
    for prediction_file in prediction_files:
        _ = parse_animal_id(prediction_file)
        atlas_path = expected_atlas_path(prediction_file)
        if not atlas_path.exists():
            missing_atlas_paths.append(atlas_path)

    if missing_atlas_paths:
        lines = "\n".join(str(path) for path in missing_atlas_paths)
        raise FileNotFoundError(
            "Atlas preflight validation failed. Missing atlas chunk files:\n"
            f"{lines}"
        )


def make_output_dir() -> Path:
    """Create and return the output directory for this run."""
    if output_dir_cfg:
        out_dir = normalize_user_path(output_dir_cfg).resolve()
    else:
        out_dir = prediction_dir.parent / "silver_standard_eval" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def make_example_plot(
    raw_image: np.ndarray | None,
    prediction_masks: np.ndarray,
    corrected_masks: np.ndarray,
    matches: list[tuple[int, int, tuple[int, int]]],
    fp_ids: list[int],
    fn_ids: list[int],
    out_path: Path,
) -> None:
    """Create a simple diagnostic plot for one chunk."""
    pred_rgb, _ = label_to_random_color(prediction_masks, seed=123)
    corrected_rgb, _ = label_to_random_color(corrected_masks, seed=456)
    corrected_centroids = get_centroid_lookup(corrected_masks)
    pred_centroids = get_centroid_lookup(prediction_masks)

    tp_array = np.zeros_like(prediction_masks, dtype=np.uint8)
    fp_array = np.zeros_like(prediction_masks, dtype=np.uint8)
    fn_array = np.zeros_like(corrected_masks, dtype=np.uint8)

    for _corrected_id, pred_id, _coord in matches:
        tp_array |= (prediction_masks == pred_id).astype(np.uint8)
    for pred_id in fp_ids:
        fp_array |= (prediction_masks == pred_id).astype(np.uint8)
    for corrected_id in fn_ids:
        fn_array |= (corrected_masks == corrected_id).astype(np.uint8)

    tp_rgb = mask_rgb(pred_rgb, tp_array > 0)
    fp_rgb = mask_rgb(pred_rgb, fp_array > 0)
    fn_rgb = mask_rgb(corrected_rgb, fn_array > 0)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.title("Raw image")
    if raw_image is not None:
        plt.imshow(raw_image, cmap="gray")
    else:
        plt.imshow(np.zeros_like(prediction_masks), cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Corrected masks (+ centroids)")
    plt.imshow(corrected_rgb)
    if corrected_centroids:
        rr = [point[0] for point in corrected_centroids.values()]
        cc = [point[1] for point in corrected_centroids.values()]
        plt.scatter(cc, rr, s=18, marker="o", linewidths=0.8)
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("Predicted masks (+ centroids)")
    plt.imshow(pred_rgb)
    if pred_centroids:
        rr = [point[0] for point in pred_centroids.values()]
        cc = [point[1] for point in pred_centroids.values()]
        plt.scatter(cc, rr, s=18, marker="o", linewidths=0.8)
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("TP")
    if raw_image is not None:
        plt.imshow(raw_image, cmap="gray")
    plt.imshow(tp_rgb, alpha=0.7)
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("FP")
    if raw_image is not None:
        plt.imshow(raw_image, cmap="gray")
    plt.imshow(fp_rgb, alpha=0.7)
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.title("FN")
    if raw_image is not None:
        plt.imshow(raw_image, cmap="gray")
    plt.imshow(fn_rgb, alpha=0.7)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -------------------------
# SETUP
# -------------------------
prediction_files = sorted(prediction_dir.glob("*_seg.npy"))
validate_prediction_set(prediction_files)
atlas_ontology, kimlab_to_allen_id_map = load_atlas_resources_from_atlaslevels(
    atlaslevels_preset,
    atlaslevels_id_map_preset,
)

out_dir = make_output_dir()
plots_dir = out_dir / "example_plots"
unquantifiable_dir = out_dir / "unquantifiable_chunks"
unquantifiable_dir.mkdir(parents=True, exist_ok=True)
if make_example_plots:
    plots_dir.mkdir(parents=True, exist_ok=True)

object_events_path = out_dir / "object_events.csv"
metrics_per_chunk_path = out_dir / "metrics_per_chunk.csv"
metrics_per_animal_path = out_dir / "metrics_per_animal.csv"
metrics_overall_path = out_dir / "metrics_overall.csv"
metrics_per_region_path = out_dir / "metrics_per_region.csv"
unquantifiable_region_summary_path = out_dir / "unquantifiable_region_summary.csv"
run_metadata_path = out_dir / "run_metadata.csv"


# -------------------------
# RUN EVALUATION
# -------------------------
object_events: list[dict[str, object]] = []
metrics_per_chunk: list[dict[str, object]] = []
region_validation_pixels: dict[int, int] = {}
region_unquantifiable_pixels: dict[int, int] = {}
example_plots_made = 0

total_corrected_objects = 0
total_predicted_objects = 0
total_tp = 0
total_fp = 0
total_fn = 0
sum_iou_tp = 0.0
n_tp_for_iou = 0

chunk_precision_list: list[float] = []
chunk_recall_list: list[float] = []
chunk_f1_list: list[float] = []
chunk_mean_iou_list: list[float] = []


for prediction_file in prediction_files:
    chunk_name = prediction_file_to_chunk_stem(prediction_file)
    animal_id = parse_animal_id(prediction_file)
    corrected_path = expected_corrected_path(prediction_file)
    atlas_path = expected_atlas_path(prediction_file)

    atlas_chunk = tifffile.imread(atlas_path)
    if atlas_chunk.ndim != 2:
        raise ValueError(f"Expected 2D atlas chunk in: {atlas_path}, found shape {atlas_chunk.shape}")

    atlas_pixel_rows = count_atlas_pixels(atlas_chunk, atlas_ontology, kimlab_to_allen_id_map)
    for row in atlas_pixel_rows:
        allen_atlas_id = int(row["atlas_allen_id"])
        region_validation_pixels[allen_atlas_id] = region_validation_pixels.get(allen_atlas_id, 0) + int(row["pixels"])

    if not corrected_path.exists():
        chunk_report_rows: list[dict[str, object]] = []
        for row in atlas_pixel_rows:
            allen_atlas_id = int(row["atlas_allen_id"])
            pixels = int(row["pixels"])
            region_unquantifiable_pixels[allen_atlas_id] = region_unquantifiable_pixels.get(allen_atlas_id, 0) + pixels
            chunk_report_rows.append(
                {
                    "chunk_name": chunk_name,
                    "animal_id": animal_id,
                    "atlas_kimlab_id": int(row["atlas_kimlab_id"]),
                    "atlas_allen_id": allen_atlas_id,
                    "atlas_name": row["atlas_name"],
                    "pixels_in_chunk_region": pixels,
                }
            )

        chunk_report_path = unquantifiable_dir / f"{chunk_name}_unquantifiable_regions.csv"
        pd.DataFrame(chunk_report_rows).to_csv(chunk_report_path, index=False)
        print(f"Skipped unquantifiable chunk (missing corrected file): {prediction_file.name}")
        continue

    prediction_payload = load_npy_payload(prediction_file)
    corrected_payload = load_npy_payload(corrected_path)
    if "masks" not in prediction_payload:
        raise ValueError(f"Missing 'masks' key in: {prediction_file}")
    if "masks" not in corrected_payload:
        raise ValueError(f"Missing 'masks' key in: {corrected_path}")

    prediction_masks = np.asarray(prediction_payload["masks"])
    corrected_masks = np.asarray(corrected_payload["masks"])
    if prediction_masks.ndim != 2:
        raise ValueError(f"Expected 2D masks in: {prediction_file}, found shape {prediction_masks.shape}")
    if corrected_masks.ndim != 2:
        raise ValueError(f"Expected 2D masks in: {corrected_path}, found shape {corrected_masks.shape}")

    raw_image = load_raw_image_for_prediction(prediction_file)

    if prediction_masks.shape != corrected_masks.shape:
        raise ValueError(
            "Prediction and corrected masks must have identical shape.\n"
            f"Prediction: {prediction_file} -> {prediction_masks.shape}\n"
            f"Corrected: {corrected_path} -> {corrected_masks.shape}"
        )
    if prediction_masks.shape != atlas_chunk.shape:
        raise ValueError(
            "Prediction/corrected mask shape must match atlas chunk shape.\n"
            f"Prediction: {prediction_file} -> {prediction_masks.shape}\n"
            f"Atlas: {atlas_path} -> {atlas_chunk.shape}"
        )

    centroid_matches, centroid_fp_ids, centroid_fn_ids = centroid_inside_gt_matching(
        manual_masks=corrected_masks,
        predicted_masks=prediction_masks,
    )

    pred_centroids = get_centroid_lookup(prediction_masks)
    corrected_centroids = get_centroid_lookup(corrected_masks)

    tp = len(centroid_matches)
    fp = len(centroid_fp_ids)
    fn = len(centroid_fn_ids)
    precision, recall, f1 = calculate_classification_metrics(tp, fp, fn)

    tp_ious: list[float] = []

    for corrected_id, pred_id, pred_coord in centroid_matches:
        corrected_row, corrected_col = corrected_centroids[corrected_id]
        pred_row, pred_col = clip_coordinate(pred_coord[0], pred_coord[1], prediction_masks.shape)
        atlas_kimlab_id, atlas_allen_id, atlas_name, atlas_row, atlas_col = atlas_info_from_coordinate(
            atlas_chunk,
            corrected_row,
            corrected_col,
            atlas_ontology,
            kimlab_to_allen_id_map,
        )
        pred_mask = prediction_masks == pred_id
        corrected_mask = corrected_masks == corrected_id
        iou = calculate_iou(pred_mask, corrected_mask)
        tp_ious.append(iou)

        object_events.append(
            {
                "chunk_name": chunk_name,
                "prediction_file": prediction_file.name,
                "corrected_file": corrected_path.name,
                "atlas_file": atlas_path.name,
                "animal_id": animal_id,
                "event_type": "TP",
                "pred_instance_id": int(pred_id),
                "corrected_instance_id": int(corrected_id),
                "pred_centroid_row": int(pred_row),
                "pred_centroid_col": int(pred_col),
                "corrected_centroid_row": int(corrected_row),
                "corrected_centroid_col": int(corrected_col),
                "atlas_row": int(atlas_row),
                "atlas_col": int(atlas_col),
                "atlas_kimlab_id": int(atlas_kimlab_id),
                "atlas_allen_id": int(atlas_allen_id),
                "atlas_name": atlas_name,
                "iou": float(iou),
                "pred_area_pixels": int(np.sum(pred_mask)),
                "corrected_area_pixels": int(np.sum(corrected_mask)),
            }
        )

    for pred_id in centroid_fp_ids:
        pred_row, pred_col = pred_centroids[pred_id]
        atlas_kimlab_id, atlas_allen_id, atlas_name, atlas_row, atlas_col = atlas_info_from_coordinate(
            atlas_chunk,
            pred_row,
            pred_col,
            atlas_ontology,
            kimlab_to_allen_id_map,
        )
        pred_mask = prediction_masks == pred_id

        object_events.append(
            {
                "chunk_name": chunk_name,
                "prediction_file": prediction_file.name,
                "corrected_file": corrected_path.name,
                "atlas_file": atlas_path.name,
                "animal_id": animal_id,
                "event_type": "FP",
                "pred_instance_id": int(pred_id),
                "corrected_instance_id": "",
                "pred_centroid_row": int(pred_row),
                "pred_centroid_col": int(pred_col),
                "corrected_centroid_row": "",
                "corrected_centroid_col": "",
                "atlas_row": int(atlas_row),
                "atlas_col": int(atlas_col),
                "atlas_kimlab_id": int(atlas_kimlab_id),
                "atlas_allen_id": int(atlas_allen_id),
                "atlas_name": atlas_name,
                "iou": "",
                "pred_area_pixels": int(np.sum(pred_mask)),
                "corrected_area_pixels": "",
            }
        )

    for corrected_id in centroid_fn_ids:
        corrected_row, corrected_col = corrected_centroids[corrected_id]
        atlas_kimlab_id, atlas_allen_id, atlas_name, atlas_row, atlas_col = atlas_info_from_coordinate(
            atlas_chunk,
            corrected_row,
            corrected_col,
            atlas_ontology,
            kimlab_to_allen_id_map,
        )
        corrected_mask = corrected_masks == corrected_id

        object_events.append(
            {
                "chunk_name": chunk_name,
                "prediction_file": prediction_file.name,
                "corrected_file": corrected_path.name,
                "atlas_file": atlas_path.name,
                "animal_id": animal_id,
                "event_type": "FN",
                "pred_instance_id": "",
                "corrected_instance_id": int(corrected_id),
                "pred_centroid_row": "",
                "pred_centroid_col": "",
                "corrected_centroid_row": int(corrected_row),
                "corrected_centroid_col": int(corrected_col),
                "atlas_row": int(atlas_row),
                "atlas_col": int(atlas_col),
                "atlas_kimlab_id": int(atlas_kimlab_id),
                "atlas_allen_id": int(atlas_allen_id),
                "atlas_name": atlas_name,
                "iou": "",
                "pred_area_pixels": "",
                "corrected_area_pixels": int(np.sum(corrected_mask)),
            }
        )

    chunk_mean_iou = safe_mean(tp_ious)
    n_corrected_objects = int(len(np.unique(corrected_masks)) - 1)
    n_predicted_objects = int(len(np.unique(prediction_masks)) - 1)

    metrics_per_chunk.append(
        {
            "chunk_name": chunk_name,
            "animal_id": animal_id,
            "n_corrected_objects": n_corrected_objects,
            "n_predicted_objects": n_predicted_objects,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou_tp": chunk_mean_iou,
        }
    )

    total_corrected_objects += n_corrected_objects
    total_predicted_objects += n_predicted_objects
    total_tp += tp
    total_fp += fp
    total_fn += fn
    sum_iou_tp += float(np.sum(tp_ious))
    n_tp_for_iou += len(tp_ious)

    chunk_precision_list.append(precision)
    chunk_recall_list.append(recall)
    chunk_f1_list.append(f1)
    chunk_mean_iou_list.append(chunk_mean_iou)

    if make_example_plots and example_plots_made < max_example_plots:
        plot_path = plots_dir / f"{chunk_name}.{plot_file_type}"
        make_example_plot(
            raw_image=raw_image,
            prediction_masks=prediction_masks,
            corrected_masks=corrected_masks,
            matches=centroid_matches,
            fp_ids=centroid_fp_ids,
            fn_ids=centroid_fn_ids,
            out_path=plot_path,
        )
        example_plots_made += 1

    print(
        f"Evaluated {prediction_file.name}: "
        f"TP={tp} FP={fp} FN={fn} "
        f"P={precision:.4f} R={recall:.4f} F1={f1:.4f}"
    )


# -------------------------
# WRITE OUTPUTS
# -------------------------
object_events_df = pd.DataFrame(object_events)
if object_events_df.empty:
    object_events_df = pd.DataFrame(
        columns=[
            "chunk_name",
            "prediction_file",
            "corrected_file",
            "atlas_file",
            "animal_id",
            "event_type",
            "pred_instance_id",
            "corrected_instance_id",
            "pred_centroid_row",
            "pred_centroid_col",
            "corrected_centroid_row",
            "corrected_centroid_col",
            "atlas_row",
            "atlas_col",
            "atlas_kimlab_id",
            "atlas_allen_id",
            "atlas_name",
            "iou",
            "pred_area_pixels",
            "corrected_area_pixels",
        ]
    )
object_events_df.to_csv(object_events_path, index=False)

metrics_per_chunk_df = pd.DataFrame(metrics_per_chunk)
if metrics_per_chunk_df.empty:
    metrics_per_chunk_df = pd.DataFrame(
        columns=[
            "chunk_name",
            "animal_id",
            "n_corrected_objects",
            "n_predicted_objects",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "mean_iou_tp",
        ]
    )
metrics_per_chunk_df.to_csv(metrics_per_chunk_path, index=False)

if not metrics_per_chunk_df.empty:
    metrics_per_animal_df = (
        metrics_per_chunk_df.groupby("animal_id", dropna=False)
        .agg(
            n_chunks=("chunk_name", "count"),
            n_corrected_objects=("n_corrected_objects", "sum"),
            n_predicted_objects=("n_predicted_objects", "sum"),
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
            overall_mean_iou_tp=("mean_iou_tp", "mean"),
        )
        .reset_index()
    )

    precision_vals: list[float] = []
    recall_vals: list[float] = []
    f1_vals: list[float] = []
    overall_mean_iou_vals: list[float] = []

    for _, row in metrics_per_animal_df.iterrows():
        precision, recall, f1 = calculate_classification_metrics(int(row["tp"]), int(row["fp"]), int(row["fn"]))
        precision_vals.append(precision)
        recall_vals.append(recall)
        f1_vals.append(f1)
        overall_mean_iou_vals.append(float(row["overall_mean_iou_tp"]))

    metrics_per_animal_df["precision"] = precision_vals
    metrics_per_animal_df["recall"] = recall_vals
    metrics_per_animal_df["f1"] = f1_vals
    metrics_per_animal_df["overall_mean_iou_tp"] = overall_mean_iou_vals
else:
    metrics_per_animal_df = pd.DataFrame(
        columns=[
            "animal_id",
            "n_chunks",
            "n_corrected_objects",
            "n_predicted_objects",
            "tp",
            "fp",
            "fn",
            "overall_mean_iou_tp",
            "precision",
            "recall",
            "f1",
        ]
    )
metrics_per_animal_df.to_csv(metrics_per_animal_path, index=False)

overall_precision, overall_recall, overall_f1 = calculate_classification_metrics(total_tp, total_fp, total_fn)
overall_mean_iou_tp = (sum_iou_tp / n_tp_for_iou) if n_tp_for_iou > 0 else 0.0

metrics_overall_rows = [
    {
        "summary_type": "overall_summed_counts",
        "n_prediction_chunks": len(prediction_files),
        "n_quantifiable_chunks": int(len(metrics_per_chunk_df)),
        "n_unquantifiable_chunks": int(len(prediction_files) - len(metrics_per_chunk_df)),
        "n_corrected_objects": total_corrected_objects,
        "n_predicted_objects": total_predicted_objects,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "mean_iou_tp": overall_mean_iou_tp,
    },
    {
        "summary_type": "average_across_chunks",
        "n_prediction_chunks": len(prediction_files),
        "n_quantifiable_chunks": int(len(metrics_per_chunk_df)),
        "n_unquantifiable_chunks": int(len(prediction_files) - len(metrics_per_chunk_df)),
        "n_corrected_objects": "",
        "n_predicted_objects": "",
        "tp": "",
        "fp": "",
        "fn": "",
        "precision": safe_mean(chunk_precision_list),
        "recall": safe_mean(chunk_recall_list),
        "f1": safe_mean(chunk_f1_list),
        "mean_iou_tp": safe_mean(chunk_mean_iou_list),
    },
]
metrics_overall_df = pd.DataFrame(metrics_overall_rows)
metrics_overall_df.to_csv(metrics_overall_path, index=False)

if not object_events_df.empty:
    object_events_no_background = object_events_df[object_events_df["atlas_allen_id"] != BACKGROUND_ATLAS_ID].copy()
    if object_events_no_background.empty:
        region_summary_df = pd.DataFrame(
            columns=[
                "atlas_allen_id",
                "atlas_name",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "mean_iou_tp",
                "n_events",
            ]
        )
    else:
        region_rows: list[dict[str, object]] = []
        for (atlas_allen_id, atlas_name), group_df in object_events_no_background.groupby(
            ["atlas_allen_id", "atlas_name"],
            dropna=False,
        ):
            tp = int((group_df["event_type"] == "TP").sum())
            fp = int((group_df["event_type"] == "FP").sum())
            fn = int((group_df["event_type"] == "FN").sum())
            precision, recall, f1 = calculate_classification_metrics(tp, fp, fn)
            tp_ious = pd.to_numeric(group_df.loc[group_df["event_type"] == "TP", "iou"], errors="coerce").dropna()
            region_rows.append(
                {
                    "atlas_allen_id": int(atlas_allen_id),
                    "atlas_name": atlas_name,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "mean_iou_tp": float(tp_ious.mean()) if not tp_ious.empty else 0.0,
                    "n_events": int(len(group_df)),
                }
            )
        region_summary_df = pd.DataFrame(region_rows).sort_values(["atlas_allen_id", "atlas_name"])
else:
    region_summary_df = pd.DataFrame(
        columns=["atlas_allen_id", "atlas_name", "tp", "fp", "fn", "precision", "recall", "f1", "mean_iou_tp", "n_events"]
    )
region_summary_df.to_csv(metrics_per_region_path, index=False)

all_region_ids = sorted(set(region_validation_pixels) | set(region_unquantifiable_pixels))
unquantifiable_region_rows: list[dict[str, object]] = []
for atlas_id in all_region_ids:
    if atlas_id == BACKGROUND_ATLAS_ID:
        continue
    total_pixels = int(region_validation_pixels.get(atlas_id, 0))
    unquantifiable_pixels = int(region_unquantifiable_pixels.get(atlas_id, 0))
    fraction = (unquantifiable_pixels / total_pixels) if total_pixels > 0 else 0.0
    unquantifiable_region_rows.append(
        {
            "atlas_allen_id": atlas_id,
            "atlas_name": atlas_ontology.get_name(atlas_id),
            "total_pixels_in_validation_set": total_pixels,
            "pixels_in_unquantifiable_chunks": unquantifiable_pixels,
            "fraction_unquantifiable": fraction,
            "is_region_flagged_unquantifiable": bool(fraction >= unquantifiable_fraction_threshold),
        }
    )
unquantifiable_region_df = pd.DataFrame(unquantifiable_region_rows)
unquantifiable_region_df.to_csv(unquantifiable_region_summary_path, index=False)

with open(run_metadata_path, mode="w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["key", "value"],
    )
    writer.writeheader()
    for key, value in [
        ("timestamp", datetime.now().isoformat(timespec="seconds")),
        ("prediction_dir", str(prediction_dir)),
        ("corrected_dir", str(corrected_dir)),
        ("atlas_dir", str(atlas_dir)),
        ("output_dir", str(out_dir)),
        ("make_example_plots", make_example_plots),
        ("max_example_plots", max_example_plots),
        ("plot_file_type", plot_file_type),
        ("unquantifiable_fraction_threshold", unquantifiable_fraction_threshold),
        ("animal_id_index", animal_id_index),
        ("filename_delimiter", filename_delimiter),
        ("atlaslevels_preset", atlaslevels_preset),
        ("atlaslevels_id_map_preset", atlaslevels_id_map_preset),
        ("atlas_name", atlas_ontology.metadata.atlas_name),
        ("atlas_version", atlas_ontology.metadata.atlas_version),
        ("atlas_ontology_source_path", atlas_ontology.metadata.ontology_source_path),
        ("atlas_id_map_source_space", kimlab_to_allen_id_map.source_space),
        ("atlas_id_map_target_space", kimlab_to_allen_id_map.target_space),
    ]:
        writer.writerow({"key": key, "value": value})

print("\nSilver-standard evaluation complete.")
print(f"Results written to: {out_dir}")
print(f"Object events: {object_events_path}")
print(f"Per-chunk metrics: {metrics_per_chunk_path}")
print(f"Overall metrics: {metrics_overall_path}")
