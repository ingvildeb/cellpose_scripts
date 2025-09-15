from cellpose import models, io, transforms
import numpy as np
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from utils import calculate_iou
import pandas as pd

## USER PARAMETERS
model_path = Path(r"Z:\Labmembers\Ingvild\Cellpose\Iba1_model\4_train\models\2025-09-08_cpsam_iba1_500epochs_wd-0.1_lr-1e-05_normTrue")
image_path = Path(r"Z:\Labmembers\Ingvild\Cellpose\Iba1_model\5_validation\\")
flow_threshold = 0.4
normalize = True


## MAIN CODE

# Set up paths
log_path = model_path.parent.parent / "training_logs" / "training_record.csv"
iou_threshold = 0.5
out_path = model_path.parent.parent / "training_logs" / "f_score_eval" / f"results_{model_path.name}_iou-{iou_threshold}_flowthreshold-{flow_threshold}_normTest{normalize}"
tif_images = list(image_path.glob('*.tif'))
model_name = model_path.name
out_path.mkdir(parents=True, exist_ok=True)

log_df = pd.read_csv(log_path)

# Find the index where the model matches model_name
row_idx = log_df.index[log_df['model'] == model_name].tolist()

if not row_idx:
    raise ValueError(f"No rows found for model {model_name}. Cannot update metrics.")

idx = row_idx[0]  # Assuming model_name is unique, take the first match

# Check for existing evaluations with the same flow_threshold and normalize_test
existing_eval = log_df[(log_df['model'] == model_name) & 
                       (log_df['normalize_apply'] == normalize)]

if len(existing_eval) > 0:
    raise ValueError(f"Model {model_name} has already been assessed with flow threshold {flow_threshold} and normalization {normalize}.")

# Initialize lists to hold metrics for all files
metrics = []

for tif in tif_images:
    img = io.imread(tif)
    model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
    predicted_masks, _, _ = model.eval(img, flow_threshold=flow_threshold, normalize=normalize)

    npy_path = Path(rf"{image_path}/{tif.name.split('.')[0]}_seg.npy")
    manual_data = np.load(npy_path, allow_pickle=True).item()

    # Access the segmentation masks
    manual_masks = manual_data['masks']
    predicted_masks_list = list(np.unique(predicted_masks))
    predicted_masks_list.remove(0)
    manual_masks_list = list(np.unique(manual_masks))
    manual_masks_list.remove(0)

    TP, FP, FN = 0, 0, 0

    TP_mask = np.zeros_like(manual_masks, dtype=np.uint16)
    FP_mask = np.zeros_like(manual_masks, dtype=np.uint16)
    FN_mask = np.zeros_like(manual_masks, dtype=np.uint16)

    for i in predicted_masks_list:
        pred_mask = (predicted_masks == i)  # Create a mask for the current prediction
        found_iou = False

        for j in manual_masks_list:
            gt_mask = (manual_masks == j)

            iou = calculate_iou(pred_mask, gt_mask)  # Calculate IoU between the prediction and ground truth

            if iou >= iou_threshold:  # If IoU meets or exceeds the threshold
                found_iou = True
                TP += 1
                
                # Create a binary array where pred_mask is 1 where True and 0 otherwise
                TP_mask += pred_mask.astype(np.uint16)
                break

        if not found_iou:
            FP += 1  # Count as FP if there's no valid IoU
            FP_mask += pred_mask.astype(np.uint16)

    for i in manual_masks_list:
        gt_mask = (manual_masks == i)
        found_iou = False

        for j in predicted_masks_list:
            pred_mask = (predicted_masks == j)

            iou = calculate_iou(gt_mask, pred_mask)

            if iou >= iou_threshold:  # If IoU meets or exceeds the threshold
                found_iou = True
                break

        if not found_iou:
            FN += 1
            FN_mask += gt_mask.astype(np.uint16)

    # Calculate precision, recall, and F1 score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    total_cells = TP + FN

    # Store metrics for the current image pair
    metrics.append((tif.name, precision, recall, f1, total_cells))

    # Print metrics for each image pair
    print(f'File evaluated: {tif.name}')
    print(f'True Positives Count: {TP}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('---')  # Separator for better readability

    # Preparing for plotting
    # Plotting Section
    plt.figure(figsize=(15, 10))

    # Ground Truth
    plt.subplot(2, 3, 1)
    plt.title("Image")
    plt.imshow(img, cmap='gray', alpha=1)  # Binarized for plotting
    plt.axis('off')

    # Ground Truth
    plt.subplot(2, 3, 2)
    plt.title("Ground Truth")
    plt.imshow((manual_masks > 0).astype(np.uint8), cmap='gray', alpha=1)  # Binarized for plotting
    plt.axis('off')

    # Predictions
    plt.subplot(2, 3, 3)
    plt.title("Predictions")
    plt.imshow((predicted_masks > 0).astype(np.uint8), cmap='gray', alpha=1)  # Binarized for plotting
    plt.axis('off')

    # True Positives
    plt.subplot(2, 3, 4)
    plt.title("True Positives")
    plt.imshow((TP_mask > 0).astype(np.uint8), cmap='gray', alpha=1)  # Binarized for plotting
    plt.axis('off')

    # False Positives
    plt.subplot(2, 3, 5)
    plt.title("False Positives")
    plt.imshow((FP_mask > 0).astype(np.uint8), cmap='gray', alpha=1)  # Binarized for plotting
    plt.axis('off')

    # False Negatives
    plt.subplot(2, 3, 6)
    plt.title("False Negatives")
    plt.imshow((FN_mask > 0).astype(np.uint8), cmap='gray', alpha=1)  # Binarized for plotting
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(Path(out_path / f"plot_{tif.stem}.svg"))
    plt.show()

# Calculate average metrics
average_precision = np.mean([m[1] for m in metrics])
average_recall = np.mean([m[2] for m in metrics])
average_f1 = np.mean([m[3] for m in metrics])
total_cells = np.sum([m[4] for m in metrics])

# Print average metrics
print('Average Metrics:')
print(f'Average Precision: {average_precision:.4f}')
print(f'Average Recall: {average_recall:.4f}')
print(f'Average F1 Score: {average_f1:.4f}')
print(f'Total number of ground truth cells: {total_cells}')

# Check if the existing flow threshold is "N/A" or a number
existing_row = log_df.loc[idx]
existing_flow_threshold = existing_row['flow_threshold']

if np.isnan(existing_flow_threshold):
    # Overwrite existing row with new metrics since it's the first test
    log_df.at[idx, 'precision'] = average_precision
    log_df.at[idx, 'recall'] = average_recall
    log_df.at[idx, 'F1'] = average_f1
    log_df.at[idx, 'num_ground_truth_cells'] = total_cells
    log_df.at[idx, 'flow_threshold'] = flow_threshold
    log_df.at[idx, 'normalize_apply'] = normalize
    print(f"Metrics updated for model {model_name}, which was not assessed previously.")
    
else:
    # Create a new entry for the log by copying the existing row and adding new metrics
    new_entry = {
        'model_number': existing_row['model_number'],
        'date': existing_row['date'],
        'model': existing_row['model'],
        'epochs': existing_row['epochs'],
        'weight_decay': existing_row['weight_decay'],
        'learning rate': existing_row['learning rate'],
        'normalize_train': existing_row['normalize_train'],
        'num_train_images': existing_row['num_train_images'],
        'num_test_images': existing_row['num_test_images'],
        'train_dir': existing_row['train_dir'],
        'test_dir': existing_row['test_dir'],
        'time_to_train': existing_row['time_to_train'],
        'precision': average_precision,
        'recall': average_recall,
        'F1': average_f1,
        'num_ground_truth_cells': total_cells,
        'flow_threshold': flow_threshold,
        'normalize_apply': normalize
    }

    # Create a DataFrame from the new entry
    new_entry_df = pd.DataFrame([new_entry])

    # Concatenate the new entry DataFrame with the existing log_df
    log_df = pd.concat([log_df, new_entry_df], ignore_index=True)
    print(f"New metrics written to log for model {model_name} with flow threshold {flow_threshold} and normalization {normalize}.")

# Save the updated log_df back to the CSV file
log_df.to_csv(log_path, index=False)
print(f"Log updated and saved to {log_path}.")

# Write metrics per image to a CSV file
csv_file_path = Path(out_path / f"metrics_per_image.csv")

with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Evaluated file', 'Precision', 'Recall', 'F1 Score'])
    writer.writerows(metrics)

    # Write average metrics at the end
    writer.writerow(['Average Metrics', average_precision, average_recall, average_f1, total_cells])

print(f'Metrics per iamge have been written to {csv_file_path}')