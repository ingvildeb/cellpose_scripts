import numpy as np
from cellpose import plot
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile
from utils import split_z_stack, calculate_z_numbers


### USER PARAMETERS

# Path to your tif and npy images
base_path = Path(r"Z:\Labmembers\Ingvild\Testing_CellPose\test_3d\test_iba1\chunked_images_512by512//")

# Option to not save the first and last z plane of the stack
# Set to True if you did not label these planes
omit_first_and_last = True


### MAIN CODE, do not edit

out_path = Path(base_path / "split_files")

# Create output directory if it doesn't exist
out_path.mkdir(exist_ok=True)

# Loop through all .tif files in the base path
for tif_file in base_path.glob("*.tif"):
    # Generate list of tif and npy planes
    img_z_list, npy_z_list, no_z_planes = split_z_stack(tif_file)

    # Get the start and end numbers for all the MIPs in the z stack
    split_stem = tif_file.stem.split("_")

    if "MIP" in split_stem:
        first_MIP_start_z, first_MIP_end_z = int(split_stem[4]), int(split_stem[5])
        last_MIP_start_z, last_MIP_end_z = int(split_stem[8]), int(split_stem[9])

        MIP_start_numbers = calculate_z_numbers(first_MIP_start_z, last_MIP_start_z, no_z_planes)
        MIP_end_numbers = calculate_z_numbers(first_MIP_end_z, last_MIP_end_z, no_z_planes)

    else:
        start_z = int(split_stem[5])
        end_z = int(split_stem[10])

        section_numbers = calculate_z_numbers(start_z, end_z, no_z_planes)

    subject_id = split_stem[1]
    chunk_info = "_".join(split_stem[-3:])

    # Determine the range of z planes to iterate over
    if omit_first_and_last:
        z_range = range(1, len(img_z_list) - 1)  # Exclude the first and last
    else:
        z_range = range(len(img_z_list))  # Include all planes

    # Loop through z planes to generate individual tif and npy files
    for z in z_range:
        img_z = img_z_list[z]
        npy_z = npy_z_list[z]

        if "MIP" in split_stem:
            out_name = f"{subject_id}_MIP_{MIP_start_numbers[z]}_{MIP_end_numbers[z]}_{chunk_info}"
        else:
            out_name = f"{subject_id}_{section_numbers[z]}_{chunk_info}"

        z_img_filename = out_path / f"{out_name}.tif"
        tifffile.imwrite(z_img_filename, img_z)

        z_npy_filename = out_path / f"{out_name}_seg.npy"
        np.save(z_npy_filename, npy_z)