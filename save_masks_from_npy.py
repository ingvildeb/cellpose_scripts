import numpy as np
from cellpose import plot
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imsave

# Set the path for your .npy images
base_path = Path(r"Z:\Labmembers\Ingvild\RM1\protocol_testing\202503_LargeBatch_AgingCCFBrains\pilot_analysis\561Neun\training_with_MIPs\images_for_validation//")
out_path = Path(base_path / "manual")

# Loop through all .npy files in the base path
for data_file in base_path.glob("*.npy"):
    # Load the .npy file
    data = np.load(data_file, allow_pickle=True).item()

    # Access the segmentation masks
    masks = data['masks']

    # Binarize the masks: Convert to binary (0 and 255)
    binary_masks = (masks > 0).astype(np.uint8) * 255  # Non-zero values become 255; zero values remain 0

    # Save the binary masks as a PNG file with the same name as the .npy file
    masks_output_path = Path(out_path / (data_file.name)).with_suffix('.png')
    imsave(masks_output_path, binary_masks)

    # Optionally visualize the binary masks
    plt.imshow(binary_masks, cmap='gray')  # Use gray colormap for binary images
    plt.title(f'Binarized Masks for {data_file.name}')
    plt.show()

