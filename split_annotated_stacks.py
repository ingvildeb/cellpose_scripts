import numpy as np
from cellpose import plot
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile
from utils import split_annotated_z_stack, calculate_z_numbers


### USER PARAMETERS

# Path to your tif and npy images
image_path = Path(r"example\path\your_path")

# Option to not save the first and last z plane of the stack
# Set to True if you did not label these planes
omit_first_and_last = True

# Filename parsing settings for names split by underscores
subject_id_index = 1
chunk_info_num_parts = 3

# Indices for MIP-style filenames
mip_first_start_index = 4
mip_first_end_index = 5
mip_last_start_index = 8
mip_last_end_index = 9

# Indices for non-MIP filenames
section_start_index = 5
section_end_index = 10


### MAIN CODE

out_path = Path(image_path / "split_files")

# Create output directory if it doesn't exist
out_path.mkdir(exist_ok=True)

# Loop through all .tif files in the image path
for tif_file in image_path.glob("*.tif"):

    npy_file = tif_file.parent / f"{tif_file.stem}_seg.npy"

    if npy_file.exists():

        # Generate list of tif and npy planes
        img_z_list, npy_z_list, no_z_planes = split_annotated_z_stack(tif_file)
        print(f"{no_z_planes} z planes found for {tif_file.stem}")

        # Get the start and end numbers for all the MIPs in the z stack
        split_stem = tif_file.stem.split("_")

        # Validate required indices for common filename fields
        if len(split_stem) <= subject_id_index:
            raise ValueError(
                f"Filename '{tif_file.stem}' does not contain subject_id_index={subject_id_index}. "
                f"Split parts: {split_stem}"
            )
        if len(split_stem) < chunk_info_num_parts:
            raise ValueError(
                f"Filename '{tif_file.stem}' has fewer parts than chunk_info_num_parts={chunk_info_num_parts}. "
                f"Split parts: {split_stem}"
            )

        # Get components for file names

        # extract the subject id from the file name
        subject_id = split_stem[subject_id_index]

        # extract the chunk info from the file name
        chunk_info = "_".join(split_stem[-chunk_info_num_parts:])

        # If MIP, get all z numbers based on first and last MIP of first and last z
        if "MIP" in split_stem:
            mip_required_max_index = max(
                mip_first_start_index,
                mip_first_end_index,
                mip_last_start_index,
                mip_last_end_index,
            )
            if len(split_stem) <= mip_required_max_index:
                raise ValueError(
                    f"Filename '{tif_file.stem}' does not match configured MIP indices "
                    f"(max index {mip_required_max_index}). Split parts: {split_stem}"
                )

            first_MIP_start_z = int(split_stem[mip_first_start_index])
            first_MIP_end_z = int(split_stem[mip_first_end_index])
            last_MIP_start_z = int(split_stem[mip_last_start_index])
            last_MIP_end_z = int(split_stem[mip_last_end_index])

            MIP_start_numbers = calculate_z_numbers(first_MIP_start_z, last_MIP_start_z, no_z_planes)
            MIP_end_numbers = calculate_z_numbers(first_MIP_end_z, last_MIP_end_z, no_z_planes)

        # if not MIP, get all z numbers based on first and last z number
        else:
            section_required_max_index = max(section_start_index, section_end_index)
            if len(split_stem) <= section_required_max_index:
                raise ValueError(
                    f"Filename '{tif_file.stem}' does not match configured non-MIP indices "
                    f"(max index {section_required_max_index}). Split parts: {split_stem}"
                )

            start_z = int(split_stem[section_start_index])
            end_z = int(split_stem[section_end_index])

            section_numbers = calculate_z_numbers(start_z, end_z, no_z_planes)

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
    else:
        print(f"No .npy file for {tif_file.stem}. Skipping.")
