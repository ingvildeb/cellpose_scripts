import numpy as np
from skimage import io
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the ground truth and predictions
ground_truth_path = r'Z:\Labmembers\Ingvild\RM1\protocol_testing\202503_LargeBatch_AgingCCFBrains\pilot_analysis\561Neun\training_with_MIPs\images_for_validation\manual\10_CS0290_MIP_025200_025280_chunk_4608_512_seg.png'
predictions_path = r'Z:\Labmembers\Ingvild\RM1\protocol_testing\202503_LargeBatch_AgingCCFBrains\pilot_analysis\561Neun\training_with_MIPs\images_for_validation\model_predictions\10_CS0290_MIP_025200_025280_chunk_4608_512.tif'

ground_truth = io.imread(ground_truth_path).astype(np.uint8)
predictions = io.imread(predictions_path).astype(np.uint8)

# Ensure binary
ground_truth = (ground_truth > 0).astype(np.uint8)  # Convert to binary (0s and 1s)
predictions = (predictions > 0).astype(np.uint8)  # Convert to binary (0s and 1s)

# Flatten the arrays for sklearn compatibility
ground_truth_flat = ground_truth.flatten()
predictions_flat = predictions.flatten()

# Calculate precision, recall, and F1 score
precision = precision_score(ground_truth_flat, predictions_flat)
recall = recall_score(ground_truth_flat, predictions_flat)
f1 = f1_score(ground_truth_flat, predictions_flat)

# Print metrics
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')