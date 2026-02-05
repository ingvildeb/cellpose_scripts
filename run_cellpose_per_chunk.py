import tifffile as tiff
from pathlib import Path
import numpy as np
import os
from cellpose import models, io
from tqdm import tqdm

## USER SETTINGS
model_path = r"Z:\Labmembers\Ingvild\Cellpose\MOBP_model\4_train\models\2025-10-09_cpsam_MOBP_1000epochs_wd-0.1_lr-1e-06_normTrue"
#model_path = r"example\path\your_model"

chunk_dir = Path(r"Z:\Labmembers\Ingvild\Cellpose\MOBP_model\for_retraining\chunked_images_256by256\MIP_006720_006780\\")
#chunk_dir = Path(r"example\path\your_chunked_image_path")

out_path = Path(r"Z:\Labmembers\Ingvild\GliaMapping\sfn_analysis\test_cellpose")
#out_path = Path(r"example\path\your_output_path")

# Choose whether or not to reconstruct the mask image after running
reconstruct = False
orig_image_path = Path(r"")

# Cellpose parameters

# Choose a flow threshold. The default is 0.4. Increasing flow_threshold makes Cellpose more lenient, 
# This can lead to the detection of more cells, especially in crowded or complex images, but may also 
# result in less accurate or ill-shaped masks.
flow_threshold = 0.6

# Set normalize to True or False. Should normally be True.
normalize = True


## MAIN CODE, do not edit

# List all chunk files in the chunk directory
# Convert to a list for tqdm to work
chunk_files = list(chunk_dir.glob("*.tif"))

if reconstruct:

    image_name = chunk_dir.stem
    orig_image = tiff.imread(orig_image_path)
    expected_shape = orig_image.shape
    out_path.mkdir(exist_ok=True)

    reconstructed_image = np.zeros(expected_shape, dtype=np.uint16)
    # Use tqdm to show progress
    for chunk_file in tqdm(chunk_files, desc="Processing chunks", unit="chunk"):
        img = io.imread(chunk_file)
        model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
        predicted_masks, _, _ = model.eval(img, flow_threshold=flow_threshold, normalize=normalize)

        # Parse the filename to get chunk coordinates
        parts = chunk_file.stem.split('_')  # Use .stem to get filename without extension

        # Extract y (row) and x (column) from filename
        y = int(parts[-2])  # y-position
        x = int(parts[-1])  # x-position without file extension
        
        # Calculate boundaries for placing into the reconstructed image
        start_y = y
        start_x = x
        end_y = start_y + img.shape[0]
        end_x = start_x + img.shape[1]

        # Ensure we do not go out of bounds when placing the chunk
        end_y = min(end_y, reconstructed_image.shape[0])
        end_x = min(end_x, reconstructed_image.shape[1])

        # Check if the chunk can fit in the reconstructed image
        reconstructed_image[start_y:end_y, start_x:end_x] = predicted_masks[:end_y - start_y, :end_x - start_x]


    # Save the reconstructed image
    output_path = os.path.join(out_path, f"reconstructed_{image_name}.tif")
    tiff.imwrite(output_path, reconstructed_image)
    print(f"Reconstructed image saved at {output_path}")

else:
    # Use tqdm to show progress
    for chunk_file in tqdm(chunk_files, desc="Processing chunks", unit="chunk"):
        img = io.imread(chunk_file)
        model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
        predicted_masks, flows, _ = model.eval(img, flow_threshold=flow_threshold, normalize=normalize)

        # generate npy file for the z plane and save to individual z output folder
        tiff.imwrite(chunk_file.parent / f"{chunk_file.stem}_masks.tif", predicted_masks.astype(np.uint8))
        io.masks_flows_to_seg(img, predicted_masks, flows, chunk_file)
