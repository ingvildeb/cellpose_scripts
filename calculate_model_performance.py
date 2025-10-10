from cellpose import models, io, transforms
import numpy as np
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from utils import calculate_iou
import pandas as pd

"""
CALCULATE THE PRECISION, RECALL AND F1 SCORE OF A CELLPOSE MODEL

This script allows you to calculate the precision, recall and F1 score of your cellpose model. It requires a manually
labeled validation set to test the model predictions against.

The code will generate a set of outputs in a folder called "f_score_eval" under your training logs folder in your 
training data folder. The metrics per image will be stored to a folder named after the model you're evaluating, with
a csv file summarizing the quantitative data and an image file (choose your preferred in the user settings) to visually
represent the raw image, ground truth labels, predictions, true positives, false positives and false negatives.

The overall precision, recall and F1 score (which may differ slightly from the score averaged across images), will
be saved to the training_log that was created when you trained the model.

"""

## USER PARAMETERS

# Set the path to your model
model_path = Path(r"Z:\Labmembers\Ingvild\Cellpose\MOBP_model\4_train\models\2025-10-09_cpsam_MOBP_1000epochs_wd-0.1_lr-1e-06_normTrue")
#model_path = Path(r"example\path\your_model")

# Set the path to your validation images. These should be manually labelled images.
validation_path = Path(r"Z:\Labmembers\Ingvild\Cellpose\MOBP_model\6_test")
#validation_path = Path(r"example\path\your_validation_path")

# Choose a flow threshold. The default is 0.4. Increasing flow_threshold makes Cellpose more lenient, 
# This can lead to the detection of more cells, especially in crowded or complex images, but may also 
# result in less accurate or ill-shaped masks.
flow_threshold = 0.6

# Set the normalization behavior. Should usually be True.
normalize = True

# Choose the file type for your output. svg is good if you want to incorporate it in a figure for a paper.
file_type = "svg"

final_test = True

## MAIN CODE

# Set IOU threshold. 0.5 is pretty standard in the field and should generally not be changed.
iou_threshold = 0.5

# Get model name and list of validation images
model_name = model_path.name
tif_images = list(validation_path.glob('*.tif'))

# Generate out path name and create it
out_path = model_path.parent.parent / "training_logs" / "f_score_eval" / f"results_{model_name}_iou-{iou_threshold}_flowthreshold-{flow_threshold}_normTest{normalize}"
out_path.mkdir(parents=True, exist_ok=True)

# Read the log file
log_path = model_path.parent.parent / "training_logs" / "training_record.csv"
log_df = pd.read_csv(log_path)

# Get the row of the model to be evaluated by model name
row_idx = log_df.index[log_df['model'] == model_name].tolist()

# Raise error if model is not in log file
if not row_idx:
    raise ValueError(f"No rows found for model {model_name}. Cannot update metrics.")

# Check for existing evaluations with the same parameters
existing_eval = log_df[(log_df['model'] == model_name) & 
                       (log_df['normalize_apply'] == normalize) &
                       (log_df['flow_threshold'] == flow_threshold)]

# Raise error if evaluation has already been done

if final_test:
    if len(existing_eval) == 0:
        raise ValueError(f"Model {model_name} has never been assessed with these parameters. Are you sure this is your final parameters for test set?")

else:
    if len(existing_eval) > 0:
        raise ValueError(f"Model {model_name} has already been assessed with these parameters.")

# Load the cellpose model
model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))

# Initialize lists to hold metrics for all files
metrics = []
total_TP = 0
total_FP = 0
total_FN = 0
total_cells = 0

# Loop through validation images
for tif in tif_images:

    # Read the image and run model on it, generating predicted masks
    img = io.imread(tif)
    predicted_masks, _, _ = model.eval(img, flow_threshold=flow_threshold, normalize=normalize)

    # Find the corresponding npy file and get manual masks
    npy_path = Path(rf"{validation_path}/{tif.name.split('.')[0]}_seg.npy")
    manual_data = np.load(npy_path, allow_pickle=True).item()
    manual_masks = manual_data['masks']

    # List all the unique masks in predictions and manual data, removing 0 which corresponds to the background
    predicted_masks_list = list(np.unique(predicted_masks))
    predicted_masks_list.remove(0)
    manual_masks_list = list(np.unique(manual_masks))
    manual_masks_list.remove(0)

    # Start counters and make empty arrays for the true positive, false positive, and false negatives masks
    TP, FP, FN = 0, 0, 0
    
    TP_array = np.zeros_like(manual_masks, dtype=np.uint16)
    FP_array = np.zeros_like(manual_masks, dtype=np.uint16)
    FN_array = np.zeros_like(manual_masks, dtype=np.uint16)

    # Loop through predicted mask IDs
    for i in predicted_masks_list:

        # Find the mask corresponding to the ID
        pred_mask = (predicted_masks == i)

        # Initialize found_iou as False
        found_iou = False

        # Loop through manual mask IDs
        for j in manual_masks_list:

            # Find the mask corresponding to the ID
            manual_mask = (manual_masks == j)

            # Calculate IoU between the prediction and ground truth mask
            iou = calculate_iou(pred_mask, manual_mask)

            # If IoU meets or exceeds the threshold, set found_IOU to true and count as a true positive
            if iou >= iou_threshold:  
                found_iou = True
                TP += 1
                
                # Add the mask to the array of true positives and stop searching for matching masks
                TP_array += pred_mask.astype(np.uint16)
                break
        
        # If found_iou is still False after searching through all manual masks, count as a false positive
        if not found_iou:
            FP += 1

            # Add the mask to the array of false positives
            FP_array += pred_mask.astype(np.uint16)

    # Loop through all manual mask IDs
    for i in manual_masks_list:

        # Find the mask corresponding to the ID
        manual_mask = (manual_masks == i)

        # Initialize found_iou as False
        found_iou = False

        # Loop through predicted mask IDs
        for j in predicted_masks_list:

            # Find the mask corresponding to the ID
            pred_mask = (predicted_masks == j)

            # Calculate IoU between the ground truth and predicted mask
            iou = calculate_iou(manual_mask, pred_mask)

            # If IoU meets or exceeds the threshold, set found_IOU to true and stop searching for matching masks
            if iou >= iou_threshold:
                found_iou = True
                break
        
        # If found_iou is still False after searching through all manual masks, count as a false negative
        if not found_iou:
            FN += 1

            # Add the mask to the array of false negatives
            FN_array += manual_mask.astype(np.uint16)

    # Calculate precision, recall, and F1 score based on the number of true positives, false positives and false negatives
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0

    # Get the total number of ground truth cells by the number of manual masks
    # Subtract one because ID 0 is the background
    cells_in_image = (len(np.unique(manual_masks)) - 1)

    # Add the number of true positives, false positives and false negatives to the total counts
    total_TP = total_TP + TP
    total_FP = total_FP + FP
    total_FN = total_FN + FN

    # Add the number of cells in the image to the total count
    total_cells = total_cells + cells_in_image

    # Store metrics for the current image to the list of metrics
    metrics.append((tif.name, precision, recall, f1, cells_in_image))

    # Print metrics for the image
    print(f'File evaluated: {tif.name}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Number of cells in image: {cells_in_image}')
    print('---')  # Separator for better readability

    # Preparing for plotting
    plt.figure(figsize=(15, 10))

    # Plot the raw image
    plt.subplot(2, 3, 1)
    plt.title("Image")
    plt.imshow(img, cmap='gray', alpha=1)
    plt.axis('off')

    # Plot the ground truth array
    plt.subplot(2, 3, 2)
    plt.title("Ground Truth")
    plt.imshow((manual_masks > 0).astype(np.uint8), cmap='gray', alpha=1)
    plt.axis('off')

    # Plot the predictions array
    plt.subplot(2, 3, 3)
    plt.title("Predictions")
    plt.imshow((predicted_masks > 0).astype(np.uint8), cmap='gray', alpha=1)
    plt.axis('off')

    # Plot the true positives array
    plt.subplot(2, 3, 4)
    plt.title("True Positives")
    plt.imshow((TP_array > 0).astype(np.uint8), cmap='gray', alpha=1)
    plt.axis('off')

    # Plot the false positives array
    plt.subplot(2, 3, 5)
    plt.title("False Positives")
    plt.imshow((FP_array > 0).astype(np.uint8), cmap='gray', alpha=1)
    plt.axis('off')

    # Plot the false negatives array
    plt.subplot(2, 3, 6)
    plt.title("False Negatives")
    plt.imshow((FN_array > 0).astype(np.uint8), cmap='gray', alpha=1)
    plt.axis('off')

    plt.tight_layout()

    # Save the figure
    if final_test:
        out_path = validation_path / "f_score_eval"
        out_path.mkdir(exist_ok=True)
        plt.savefig(Path(out_path / f"plot_{tif.stem}.{file_type}"))
    else:
        plt.savefig(Path(out_path / f"plot_{tif.stem}.{file_type}"))

    plt.show()

# Calculate the overall precision, recall and F1 score (based on all manual masks across all images)
overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
overall_f1 = 2 * ((overall_precision * overall_recall) / (overall_precision + overall_recall)) if (overall_precision + overall_recall) > 0 else 0

# Print overall metrics
print('Overall Metrics:')
print(f'Overall Precision: {overall_precision:.4f}')
print(f'Overall Recall: {overall_recall:.4f}')
print(f'Overall F1 Score: {overall_f1:.4f}')
print(f'Total number of ground truth cells: {total_cells}')

# Logging the results

# Get the first match from the log for the model name
idx = row_idx[0]

# Check if the existing flow threshold is "N/A" or a number
existing_row = log_df.loc[idx]
existing_flow_threshold = existing_row['flow_threshold']

if np.isnan(existing_flow_threshold):
    # Overwrite existing row with new metrics since it's the first test
    log_df.at[idx, 'precision'] = overall_precision
    log_df.at[idx, 'recall'] = overall_recall
    log_df.at[idx, 'F1'] = overall_f1
    log_df.at[idx, 'num_ground_truth_cells'] = total_cells
    log_df.at[idx, 'flow_threshold'] = flow_threshold
    log_df.at[idx, 'normalize_apply'] = str(normalize)
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
        'precision': overall_precision,
        'recall': overall_recall,
        'F1': overall_f1,
        'num_ground_truth_cells': total_cells,
        'flow_threshold': flow_threshold,
        'normalize_apply': str(normalize)
    }

    # Create a DataFrame from the new entry
    new_entry_df = pd.DataFrame([new_entry])

    # Concatenate the new entry DataFrame with the existing log_df
    log_df = pd.concat([log_df, new_entry_df], ignore_index=True)

# Calculate average metrics across images
# This may be slightly different than the overall metrics
average_precision = np.mean([m[1] for m in metrics])
average_recall = np.mean([m[2] for m in metrics])
average_f1 = np.mean([m[3] for m in metrics])

# Write metrics per image to a CSV file
csv_file_path = Path(out_path / f"metrics_per_image.csv")

with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Evaluated file', 'Precision', 'Recall', 'F1 Score', 'Number of cells'])
    writer.writerows(metrics)

    # Write average metrics at the end
    writer.writerow(['Average Metrics', average_precision, average_recall, average_f1, total_cells])

print(f'Metrics per image have been written to {csv_file_path}')


# Save the updated log_df back to the CSV file
if final_test:
    new_entry_df.to_csv(out_path / "results_test_data.csv", index=False)
    print(f"Overall metrics and model info written to {out_path}")
else:
    log_df.to_csv(log_path, index=False)
    print(f"Log updated and saved to {log_path}.")






