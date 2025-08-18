import numpy as np
from cellpose import plot
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile
from utils import split_z_stack, calculate_z_numbers

# Set the path for your .npy images
base_path = Path(r"Z:\Labmembers\Ingvild\Testing_CellPose\test_3d\test_iba1\chunked_images_512by512//")
out_path = Path(base_path / "split_files")


## Main code, do not edit
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
    
    # Loop through z planes to generate individual tif and npy files
    z = -1
    for img_z, npy_z in zip(img_z_list, npy_z_list):
        z = z+1

        if "MIP" in split_stem:
            out_name = f"{subject_id}_MIP_{MIP_start_numbers[z]}_{MIP_end_numbers[z]}_{chunk_info}"
        else:
            out_name = f"{subject_id}_{section_numbers[z]}_{chunk_info}"

        z_img_filename = out_path / f"{out_name}.tif"
        tifffile.imwrite(z_img_filename, img_z)

        z_npy_filename = out_path / f"{out_name}_seg.npy"
        np.save(z_npy_filename, npy_z)







