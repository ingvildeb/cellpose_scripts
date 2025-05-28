from cellpose import models, io, transforms
import numpy as np
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from utils import calculate_iou

iou_threshold = 0.5
model_path = Path(r"C:\Users\SmartBrain_32C_TR\.cellpose\models\cpsam")
image_path = Path(r"Z:\Labmembers\Ingvild\RM1\protocol_testing\202503_LargeBatch_AgingCCFBrains\pilot_analysis\561Neun\training_with_MIPs\images_for_validation\\")
out_path = Path(image_path / f"results_{model_path.name}_iou-{iou_threshold}")
tif_images = list(image_path.glob('*.tif'))
out_path.mkdir(exist_ok=True)

# Initialize lists to hold metrics for all files
metrics = []

for tif in tif_images:
    img = io.imread(tif)
    model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
    predicted_masks, _, _ = model.eval(img, flow_threshold=1, normalize={"percentile":[1,99]})

    npy_path = Path(rf"{image_path}/{tif.name.split('.')[0]}_seg.npy")
    manual_data = np.load(npy_path, allow_pickle=True).item()

    # Access the segmentation masks
    manual_masks = manual_data['masks']
    predicted_masks_list = list(np.unique(predicted_masks))
    predicted_masks_list.remove(0)
    manual_masks_list = list(np.unique(manual_masks))
    manual_masks_list.remove(0)

    TP = 0
    FP = 0
    FN = 0

    
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

    # FN = (len(np.unique(manual_masks)) - 1) - TP  # Remaining ground truth objects not matched
    # FN_mask = (manual_masks * (1 - TP_mask.astype(bool))).astype(np.uint16)

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

# Write metrics to a CSV file

csv_file_path = Path(out_path / "metrics_summary.csv")

with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Evaluated file', 'Precision', 'Recall', 'F1 Score'])
    writer.writerows(metrics)

    # Write average metrics at the end
    writer.writerow(['Average Metrics', '', average_precision, average_recall, average_f1])

print(f'Metrics have been written to {csv_file_path}')
