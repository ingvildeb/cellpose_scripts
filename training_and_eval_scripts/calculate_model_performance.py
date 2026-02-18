from cellpose import models, io
import numpy as np
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from utils.utils import (
    match_instances_iou,
    label_to_random_color,
    centroid_inside_gt_matching,
    calculate_iou,
    instance_centroid,
)
import pandas as pd
from datetime import datetime

"""
EVALUATE A CELLPOSE MODEL WITH TWO COMPLEMENTARY INSTANCE METRICS (2D)

This script evaluates a Cellpose model on 2D image chunks using:
1) IoU-based 1-to-1 instance matching (segmentation-quality proxy; IoU>=threshold)
2) Centroid-inside-GT 1-to-1 instance matching (detection/localization proxy)

Outputs:
- Per-image CSV: metrics_per_image.csv (per-image metrics only)
- Figures: diagnostic plots per image
- Evaluation log CSV: evaluation_log.csv (one row per evaluation run), written to the SAME directory as training_log_path

Notes:
- training_log_path should point to your training record CSV that contains one row per trained model.
- In that training log, 'model' should match model_path.name and map to a unique 'model_number'.
- This version ALWAYS appends a new row to evaluation_log.csv (no duplicate blocking).
"""

# --------------------
# USER PARAMETERS
# --------------------

# Path to your trained model (folder or file path used by CellposeModel)
model_path = Path(
    r"Z:\Labmembers\Ingvild\Cellpose\NeuN_model\testing\christines_data\2026-02-16_cpsam_Npas4-Cre_v2_71_training_images_2000epochs_wd-0.1_lr-1e-05_normTrue"
)

# Path to validation images (2D .tif chunks) with corresponding *_seg.npy files
validation_path = Path(r"Z:\Labmembers\Ingvild\Cellpose\NeuN_model\testing\christines_data\val_data")

# Path to CSV with training record (one row per model/training run)
training_log_path = Path(
    r"Z:\Labmembers\Ingvild\Cellpose\NeuN_model\testing\christines_data\training_record.csv"
)

# Cellpose inference params
flow_threshold = 0.6
normalize = True

# Choose whether to use GPU. Set False if running on CPU only.
use_gpu = True

# Evaluation params
iou_threshold = 0.5

# Output figure type
file_type = "svg"

# Optional: limit number of images for quick debugging (set to None to run all)
limit_images = None  # e.g. 5


# --------------------
# HELPERS
# --------------------
def mask_rgb(rgb_img: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    """Mask an RGB image by a binary mask (keeps colors, zeros background)."""
    out = rgb_img.copy()
    out[~binary_mask.astype(bool)] = 0
    return out


def get_instance_centroids_from_labels(label_img: np.ndarray) -> dict[int, tuple[int, int]]:
    """Return dict: instance_id -> (row, col) centroid."""
    ids = [int(i) for i in np.unique(label_img) if i != 0]
    out = {}
    for _id in ids:
        out[_id] = instance_centroid(label_img == _id)
    return out


def safe_mean(x):
    return float(np.mean(x)) if len(x) else 0.0


def calculate_classification_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# --------------------
# SETUP
# --------------------
# Get model name, list images to be evaluated and raise errors if user specified paths / files are not found
model_name = model_path.name
validation_path = validation_path.resolve()
tif_images = sorted(validation_path.glob("*.tif"))
if limit_images is not None:
    tif_images = tif_images[: int(limit_images)]

if len(tif_images) == 0:
    raise ValueError(f"No .tif images found in validation_path: {validation_path}")

if not training_log_path.exists():
    raise FileNotFoundError(f"training_log_path not found: {training_log_path}")

training_df = pd.read_csv(training_log_path)

# Check that model and model number columns exist in the training log
if "model" not in training_df.columns:
    raise ValueError(f"training_log_path must contain a 'model' column. Found: {list(training_df.columns)}")
if "model_number" not in training_df.columns:
    raise ValueError(
        f"training_log_path must contain a 'model_number' column (unique ID). Found: {list(training_df.columns)}"
    )

# Check that model name is found in log
model_rows = training_df[training_df["model"] == model_name]
if len(model_rows) == 0:
    raise ValueError(
        f"Model '{model_name}' not found in training log. "
        f"Check that training_record.csv column 'model' matches model_path.name."
    )

# Check that model name is unique
if len(model_rows) > 1:
    raise ValueError(
        f"Model '{model_name}' matches multiple rows in training log. "
        f"Expected unique mapping to model_number."
    )

# Get the model number
model_number = model_rows.iloc[0]["model_number"]

# Set up output directories in same folder as the training log
log_dir = training_log_path.parent.resolve()
evaluation_log_path = log_dir / "evaluation_log.csv"

out_base = log_dir / "f_score_eval"
out_base.mkdir(parents=True, exist_ok=True)

out_name_base = f"results_model-number{model_number}"
out_path = out_base / out_name_base

# --- create a unique out_path if it already exists ---
if out_path.exists():
    k = 1
    while True:
        candidate = out_base / f"{out_name_base}_run-{k:03d}"
        if not candidate.exists():
            out_path = candidate
            break
        k += 1

out_path.mkdir(parents=True, exist_ok=False)

# Per-image CSV path
metrics_csv_path = out_path / "metrics_per_image.csv"

# Load Cellpose model
model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))

# --------------------
# RUN EVALUATION
# --------------------

# Empty list to hold all the metrics for different images
metrics_per_image = []

# Initialize total values used in overall metrics
# Overall metrics are computed as an alternative to just averaging across images, to avoid "unfair" influence of
# poorly predicted chunks
total_cells = 0
total_TP_iou, total_FP_iou, total_FN_iou = 0, 0, 0
total_TP_c, total_FP_c, total_FN_c = 0, 0, 0

sum_iou_TP_iou = 0.0
n_TP_iou = 0

sum_iou_TP_centroid = 0.0
n_TP_centroid = 0

# Initialize lists to hold all precision, recall, f1 and mean iou values for all images
prec_iou_list, rec_iou_list, f1_iou_list, mean_iou_iou_list = [], [], [], []
prec_c_list, rec_c_list, f1_c_list, mean_iou_c_list = [], [], [], []

# Loop through all tif images to perform evaluation
for tif in tif_images:
    
    # --- Image data preparation ---

    # Read the image and create prediction masks using the cellpose model
    img = io.imread(tif)
    predicted_masks, _, _ = model.eval(img, flow_threshold=flow_threshold, normalize=normalize)
    
    # Find the corresponding npy file, raising error if it does not exist
    npy_path = validation_path / f"{tif.stem}_seg.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Missing GT npy for {tif.name}: {npy_path}")

    # Load the npy file and read mask data
    manual_data = np.load(npy_path, allow_pickle=True).item()
    manual_masks = manual_data["masks"]

    # --- IoU matching ---
    # This part uses an IoU threshold to determine whether pairs of objects are matching
    
    # Use iou matching function to get matches, FP ids and FN ids
    # matches_iou is a list of tuples each containing gt_id, pr_id, and iou value for TP
    # FP_ids and FN_ids are lists of the FP and FN ids
    matches_iou, FP_ids_iou, FN_ids_iou = match_instances_iou(
        manual_masks=manual_masks,
        predicted_masks=predicted_masks,
        iou_threshold=iou_threshold,
    )
    
    # Get the number of TP, FP and FN from length of the lists created by match_instances_iou
    TP_iou = len(matches_iou)
    FP_iou = len(FP_ids_iou)
    FN_iou = len(FN_ids_iou)

    # Calculate precision, recall and f1 score based on IoU matched pairs
    precision_iou, recall_iou, f1_iou = calculate_classification_metrics(TP_iou, FP_iou, FN_iou)

    # Extract the iou value from every matched pair in the matches_iou list, and calculate their mean
    # Returns 0 if there is no TP match
    # Used for average metrics
    mean_iou_TP_iouMatching = safe_mean([m[2] for m in matches_iou])

    # Sum all iou values and add to sum_iou_TP_iou
    # Get n of all iou values and add to n_TP_iou
    # Used for overall metrics
    if TP_iou > 0:
        sum_iou_TP_iou += float(np.sum([m[2] for m in matches_iou]))
        n_TP_iou += TP_iou

    # Add all TP, FP and FN instances to respective counters (used for overall metrics)
    total_TP_iou += TP_iou
    total_FP_iou += FP_iou
    total_FN_iou += FN_iou

    # --- Centroid-inside-GT matching ---
    # This part checks whether centroids of predicted objects are within gt objects to determine whether pairs of objects are matching
    
    # Use centroid_inside_gt_matching function to get matches, FP ids and FN ids
    # centroid_matches is a list of tuples each containing gt_id, pr_id, and iou value for TP
    # FP_ids and FN_ids are lists of the FP and FN ids
    
    centroid_matches, centroid_FP_ids, centroid_FN_ids = centroid_inside_gt_matching(
        manual_masks=manual_masks,
        predicted_masks=predicted_masks,
    )

    # Get the number of TP, FP and FN from length of the lists created by centroid_inside_gt_matching
    TP_c = len(centroid_matches)
    FP_c = len(centroid_FP_ids)
    FN_c = len(centroid_FN_ids)

    # Calculate precision, recall and f1 score based on centroid matched pairs
    precision_c, recall_c, f1_c = calculate_classification_metrics(TP_c, FP_c, FN_c)

    # Initialize mean_iou_TP_centroidMatching and calculate iou of all matching pairs
    
    mean_iou_TP_centroidMatching = 0.0
    if TP_c > 0:
        ious_c = []
        for gt_id, pr_id, _rc in centroid_matches:
            gt = (manual_masks == gt_id)
            pr = (predicted_masks == pr_id)
            ious_c.append(calculate_iou(gt, pr))
            
        # Calculate mean IoU in the image, used for average metrics
        mean_iou_TP_centroidMatching = safe_mean(ious_c)

        # Sum all iou values and add to sum_iou_TP_iou
        # Get n of all iou values and add to n_TP_iou
        # Used for overall metrics
        sum_iou_TP_centroid += float(np.sum(ious_c))
        n_TP_centroid += len(ious_c)

    # Add all TP, FP and FN instances to respective counters (used for overall metrics)
    total_TP_c += TP_c
    total_FP_c += FP_c
    total_FN_c += FN_c

    # --- Append image-level metrics ---
    # Get number of cells in ground truth data
    cells_in_image = (len(np.unique(manual_masks)) - 1)
    
    # Add number of cells to total count
    total_cells += cells_in_image

    # Append image-level metrics to respective lists
    prec_iou_list.append(precision_iou)
    rec_iou_list.append(recall_iou)
    f1_iou_list.append(f1_iou)
    mean_iou_iou_list.append(mean_iou_TP_iouMatching)

    prec_c_list.append(precision_c)
    rec_c_list.append(recall_c)
    f1_c_list.append(f1_c)
    mean_iou_c_list.append(mean_iou_TP_centroidMatching)

    # Store image-level metrics as a row
    metrics_per_image.append(
        (
            tif.name,
            precision_iou, recall_iou, f1_iou, mean_iou_TP_iouMatching,
            precision_c, recall_c, f1_c, mean_iou_TP_centroidMatching,
            cells_in_image,
        )
    )

    # --------------------
    # VISUALIZATION
    # --------------------
    
    # --- Construct arrays for iou based analysis ---
    
    # Label gt and prediction images with random color schemes
    pred_rgb, _ = label_to_random_color(predicted_masks, seed=123)
    gt_rgb, _ = label_to_random_color(manual_masks, seed=456)

    # Initialize arrays for visualization
    TP_array = np.zeros_like(manual_masks, dtype=np.uint8)
    FP_array = np.zeros_like(manual_masks, dtype=np.uint8)
    FN_array = np.zeros_like(manual_masks, dtype=np.uint8)

    # Add TP predictions to TP_array
    for gt_id, pr_id, _iou in matches_iou:
        TP_array |= (predicted_masks == pr_id).astype(np.uint8)

    # Add FP predictions to FP arrays
    for pr_id in FP_ids_iou:
        FP_array |= (predicted_masks == pr_id).astype(np.uint8)

    # Add FN grount truth objects to FN array
    for gt_id in FN_ids_iou:
        FN_array |= (manual_masks == gt_id).astype(np.uint8)

    # Mask RGB-converted prediction and GT images by the TP, FP and FN arrays
    # to create color-consistent arrays
    TP_rgb = mask_rgb(pred_rgb, TP_array > 0)
    FP_rgb = mask_rgb(pred_rgb, FP_array > 0)
    FN_rgb = mask_rgb(gt_rgb, FN_array > 0)

    # --- Construct arrays for centroid based analysis ---
    
    # Initialize arrays for visualization
    TPc_array = np.zeros_like(manual_masks, dtype=np.uint8)
    FPc_array = np.zeros_like(manual_masks, dtype=np.uint8)
    FNc_array = np.zeros_like(manual_masks, dtype=np.uint8)

    # Add TP predictions to TP_array
    for _gt_id, pr_id, _rc in centroid_matches:
        TPc_array |= (predicted_masks == pr_id).astype(np.uint8)
        
    # Add FP predictions to FP arrays    
    for pr_id in centroid_FP_ids:
        FPc_array |= (predicted_masks == pr_id).astype(np.uint8)
        
    # Add FN grount truth objects to FN array    
    for gt_id in centroid_FN_ids:
        FNc_array |= (manual_masks == gt_id).astype(np.uint8)
        
    # Mask RGB-converted prediction and GT images by the TP, FP and FN arrays
    # to create color-consistent arrays
    TPc_rgb = mask_rgb(pred_rgb, TPc_array > 0)
    FPc_rgb = mask_rgb(pred_rgb, FPc_array > 0)
    FNc_rgb = mask_rgb(gt_rgb, FNc_array > 0)

    # --- Get centroids to overlay ---
    gt_centroids = get_instance_centroids_from_labels(manual_masks)
    pred_centroids = get_instance_centroids_from_labels(predicted_masks)

    # --- Plot figure ---
    plt.figure(figsize=(15, 15))

    # Raw image subplot
    plt.subplot(3, 3, 1)
    plt.title("Raw image")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    # Ground truth subplot
    plt.subplot(3, 3, 2)
    plt.title("Ground Truth (+ centroids)")
    plt.imshow(gt_rgb)

    if gt_centroids:
        rr = [p[0] for p in gt_centroids.values()]
        cc = [p[1] for p in gt_centroids.values()]
        plt.scatter(cc, rr, s=18, marker="o", linewidths=0.8)

    plt.axis("off")
    
    # Predictions subplot
    plt.subplot(3, 3, 3)
    plt.title("Predictions (+ centroids)")
    plt.imshow(pred_rgb)

    if pred_centroids:
        rr = [p[0] for p in pred_centroids.values()]
        cc = [p[1] for p in pred_centroids.values()]
        plt.scatter(cc, rr, s=18, marker="o", linewidths=0.8)

    plt.axis("off")

    # True positives subplot (IoU)
    plt.subplot(3, 3, 4)
    plt.title("TP (IoU)")
    plt.imshow(img, cmap="gray")
    plt.imshow(TP_rgb, alpha=0.7)
    plt.axis("off")

    # False positives subplot (IoU)
    plt.subplot(3, 3, 5)
    plt.title("FP (IoU)")
    plt.imshow(img, cmap="gray")
    plt.imshow(FP_rgb, alpha=0.7)
    plt.axis("off")

    # False negatives subplot (IoU)
    plt.subplot(3, 3, 6)
    plt.title("FN (IoU)")
    plt.imshow(img, cmap="gray")
    plt.imshow(FN_rgb, alpha=0.7)
    plt.axis("off")

    # True positives subplot (centroid)
    plt.subplot(3, 3, 7)
    plt.title("TP (centroid)")
    plt.imshow(img, cmap="gray")
    plt.imshow(TPc_rgb, alpha=0.7)
    plt.axis("off")

    # False positives subplot (centroid)
    plt.subplot(3, 3, 8)
    plt.title("FP (centroid)")
    plt.imshow(img, cmap="gray")
    plt.imshow(FPc_rgb, alpha=0.7)
    plt.axis("off")

    # False negatives subplot (centroid)
    plt.subplot(3, 3, 9)
    plt.title("FN (centroid)")
    plt.imshow(img, cmap="gray")
    plt.imshow(FNc_rgb, alpha=0.7)
    plt.axis("off")

    # Save and show plot
    plt.tight_layout()
    plt.savefig(out_path / f"plot_{tif.stem}.{file_type}")
    plt.show()
    
    # Print image-level metrics
    print(f"File evaluated: {tif.name}")
    print(
        f"IoU-matching: P={precision_iou:.4f} R={recall_iou:.4f} F1={f1_iou:.4f} "
        f"meanIoU_TP={mean_iou_TP_iouMatching:.3f}"
    )
    print(
        f"Centroid-matching: P={precision_c:.4f} R={recall_c:.4f} F1={f1_c:.4f} "
        f"meanIoU_TP={mean_iou_TP_centroidMatching:.3f}"
    )
    print(f"Number of GT cells: {cells_in_image}")
    print("---")

# --------------------
# OVERALL + AVERAGE METRICS
# --------------------

# --- Calculate overall metrics based on the total counts ---

# IoU based metrics
overall_precision_iou, overall_recall_iou, overall_f1_iou = calculate_classification_metrics(total_TP_iou, 
                                                                                             total_FP_iou, 
                                                                                             total_FN_iou)
overall_mean_iou_TP_iouMatching = (sum_iou_TP_iou / n_TP_iou) if n_TP_iou > 0 else 0.0

# Centroid based metrics
overall_precision_centroid, overall_recall_centroid, overall_f1_centroid = calculate_classification_metrics(total_TP_c, 
                                                                                                            total_FP_c, 
                                                                                                            total_FN_c)
overall_mean_iou_TP_centroidMatching = (sum_iou_TP_centroid / n_TP_centroid) if n_TP_centroid > 0 else 0.0

# Calculate average metrics across all images
average_precision_iou = safe_mean(prec_iou_list)
average_recall_iou = safe_mean(rec_iou_list)
average_f1_iou = safe_mean(f1_iou_list)
average_mean_iou_TP_iouMatching = safe_mean(mean_iou_iou_list)

average_precision_centroid = safe_mean(prec_c_list)
average_recall_centroid = safe_mean(rec_c_list)
average_f1_centroid = safe_mean(f1_c_list)
average_mean_iou_TP_centroidMatching = safe_mean(mean_iou_c_list)

# Print overall metrics
print("\nOverall Metrics (micro across all images):")
print("IoU matching:")
print(f"  Precision: {overall_precision_iou:.4f}  Recall: {overall_recall_iou:.4f}  F1: {overall_f1_iou:.4f}")
print(f"  mean_iou_TP_iouMatching: {overall_mean_iou_TP_iouMatching:.3f}")
print("Centroid matching:")
print(f"  Precision: {overall_precision_centroid:.4f}  Recall: {overall_recall_centroid:.4f}  F1: {overall_f1_centroid:.4f}")
print(f"  mean_iou_TP_centroidMatching: {overall_mean_iou_TP_centroidMatching:.3f}")
print(f"Total GT cells: {total_cells}")
print(f"Images evaluated: {len(tif_images)}")

# --------------------
# WRITE PER-IMAGE CSV (PER-IMAGE ONLY)
# --------------------
with open(metrics_csv_path, mode="w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "Evaluated file",
        "Precision_iouMatch", "Recall_iouMatch", "F1_iouMatch", "mean_iou_TP_iouMatching",
        "Precision_centroid", "Recall_centroid", "F1_centroid", "mean_iou_TP_centroidMatching",
        "Number of GT cells",
    ])
    w.writerows(metrics_per_image)

print(f"\nPer-image metrics written to: {metrics_csv_path}")

# --------------------
# APPEND ONE ROW TO EVALUATION LOG (AVERAGE + OVERALL)
# --------------------
evaluation_log_path = log_dir / "evaluation_log.csv"

eval_cols = [
    "eval_timestamp",
    "model_number",
    "model",
    "model_path",
    "validation_dir",
    "out_dir",
    "n_images",
    "total_gt_cells",
    "iou_threshold",
    "flow_threshold",
    "normalize_apply",

    "overall_precision_iou",
    "overall_recall_iou",
    "overall_f1_iou",
    "overall_mean_iou_TP_iouMatching",

    "average_precision_iou",
    "average_recall_iou",
    "average_f1_iou",
    "average_mean_iou_TP_iouMatching",

    "overall_precision_centroid",
    "overall_recall_centroid",
    "overall_f1_centroid",
    "overall_mean_iou_TP_centroidMatching",

    "average_precision_centroid",
    "average_recall_centroid",
    "average_f1_centroid",
    "average_mean_iou_TP_centroidMatching",
]

eval_row = {
    "eval_timestamp": datetime.now().isoformat(timespec="seconds"),
    "model_number": model_number,
    "model": model_name,
    "model_path": str(model_path),
    "validation_dir": str(validation_path),
    "out_dir": str(out_path),
    
    "n_images": len(tif_images),
    "total_gt_cells": total_cells,
    "iou_threshold": iou_threshold,
    "flow_threshold": flow_threshold,
    "normalize_apply": str(normalize),

    "overall_precision_iou": overall_precision_iou,
    "overall_recall_iou": overall_recall_iou,
    "overall_f1_iou": overall_f1_iou,
    "overall_mean_iou_TP_iouMatching": overall_mean_iou_TP_iouMatching,

    "average_precision_iou": average_precision_iou,
    "average_recall_iou": average_recall_iou,
    "average_f1_iou": average_f1_iou,
    "average_mean_iou_TP_iouMatching": average_mean_iou_TP_iouMatching,

    "overall_precision_centroid": overall_precision_centroid,
    "overall_recall_centroid": overall_recall_centroid,
    "overall_f1_centroid": overall_f1_centroid,
    "overall_mean_iou_TP_centroidMatching": overall_mean_iou_TP_centroidMatching,

    "average_precision_centroid": average_precision_centroid,
    "average_recall_centroid": average_recall_centroid,
    "average_f1_centroid": average_f1_centroid,
    "average_mean_iou_TP_centroidMatching": average_mean_iou_TP_centroidMatching,
}
# Check so that header is written only if this is the first time creating the log file
write_header = not evaluation_log_path.exists()

# Write the eval_row to evaluation log file
with open(evaluation_log_path, mode="a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=eval_cols)
    if write_header:
        w.writeheader()
    w.writerow({k: eval_row.get(k, "") for k in eval_cols})
