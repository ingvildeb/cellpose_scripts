import numpy as np
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import seaborn as sns  # For better aesthetics in plots
from skimage import io  # Don't forget to import io if you're using it to read images

image_path = Path(r"Z:\Labmembers\Ingvild\RM1\protocol_testing\202503_LargeBatch_AgingCCFBrains\pilot_analysis\561Neun\training_with_MIPs\images_for_validation\\")
tif_images = list(image_path.glob('*.tif'))

# Initialize lists to hold metrics for all files
metrics = []
large_cells_images = []  # To hold images with cells larger than 200 pixels

# Size threshold for masks
size_threshold = 200  # Size threshold in pixels

for tif in tif_images:
    img = io.imread(tif)

    npy_path = Path(rf"{image_path}/{tif.name.split('.')[0]}_seg.npy")
    manual_data = np.load(npy_path, allow_pickle=True).item()

    # Access the segmentation masks
    manual_masks = manual_data['masks']
    manual_masks_list = list(np.unique(manual_masks))
    manual_masks_list.remove(0)  # Assuming 0 is the background class that should be excluded

    # Iterate through each mask and calculate its size
    for i in manual_masks_list:
        manual_mask = (manual_masks == i)
        mask_size = np.sum(manual_mask)  # Sum the True values to get the mask size
        
        # Store the mask size with its corresponding id
        metrics.append({'mask_id': i, 'size': mask_size})

        # Check if the mask size exceeds the size threshold and record the image name
        if mask_size > size_threshold:
            large_cells_images.append(tif.name)

# Extract the sizes from metrics for statistical calculations
sizes = np.array([metric['size'] for metric in metrics])

# Calculate average, min, and max
average_size = np.mean(sizes) if sizes.size > 0 else 0
min_size = np.min(sizes) if sizes.size > 0 else 0
max_size = np.max(sizes) if sizes.size > 0 else 0

# Print the results
print(f"Average mask size: {average_size}")
print(f"Minimum mask size: {min_size}")
print(f"Maximum mask size: {max_size}")

# Print the names of images with cells larger than 200 pixels
if large_cells_images:
    print("Found cells with sizes > 200 pixels in the following images:")
    for image in set(large_cells_images):  # Use set to avoid duplicates
        print(image)
else:
    print("No images found with cells larger than 200 pixels.")

# Plot the distribution of mask sizes
plt.figure(figsize=(10, 6))
sns.histplot(sizes, bins=30, kde=True)  # KDE=True adds a kernel density estimate
plt.title('Distribution of Cell Sizes')
plt.xlabel('Cell Size (Number of Pixels)')
plt.ylabel('Frequency')
plt.grid()
plt.axvline(x=average_size, color='r', linestyle='--', label='Average Size')
plt.axvline(x=min_size, color='g', linestyle='--', label='Minimum Size')
plt.axvline(x=max_size, color='b', linestyle='--', label='Maximum Size')
plt.legend()
plt.show()

# Optionally, you can save the metrics to a CSV file
output_csv_path = image_path / "mask_sizes.csv"
with open(output_csv_path, mode='w', newline='') as csv_file:
    fieldnames = ['mask_id', 'size']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for metric in metrics:
        writer.writerow(metric)

print(f"Mask sizes saved to {output_csv_path}")

# Optionally, you can save the metrics to a CSV file
output_csv_path = image_path / "mask_sizes.csv"
with open(output_csv_path, mode='w', newline='') as csv_file:
    fieldnames = ['mask_id', 'size']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for metric in metrics:
        writer.writerow(metric)

print(f"Mask sizes saved to {output_csv_path}")
