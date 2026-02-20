"""
Split annotated z-stacks into per-plane TIFF and `_seg.npy` files with standardized names.

Config usage:
- Copy `z_stack_scripts/configs/split_annotated_stacks_config_template.toml` to
  `z_stack_scripts/configs/split_annotated_stacks_config_local.toml`.
- Edit `_local.toml` to your preferred settings and run the script.
- If `_local.toml` is missing, the script falls back to `_template.toml`.
"""

from pathlib import Path
import numpy as np
import tifffile
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from utils.io_helpers import load_script_config, normalize_user_path, require_dir
from utils.utils import split_annotated_z_stack, calculate_z_numbers

# -------------------------
# CONFIG LOADING (shared helper)
# -------------------------
test_mode = False
cfg = load_script_config(Path(__file__), "split_annotated_stacks_config", test_mode=test_mode)

# -------------------------
# CONFIG PARAMETERS
# -------------------------
image_path = require_dir(normalize_user_path(cfg["image_path"]), "Image path")
omit_first_and_last = cfg["omit_first_and_last"]
mip_mode = cfg["mip_mode"]

subject_id_index = cfg["subject_id_index"]

mip_first_start_index = cfg["mip_first_start_index"]
mip_first_end_index = cfg["mip_first_end_index"]
mip_last_start_index = cfg["mip_last_start_index"]
mip_last_end_index = cfg["mip_last_end_index"]

section_start_index = cfg["section_start_index"]
section_end_index = cfg["section_end_index"]

if mip_mode:
    chunk_info_num_parts = cfg["chunk_info_num_parts_mip"]
else:
    chunk_info_num_parts = cfg["chunk_info_num_parts_section"]

# -------------------------
# MAIN CODE
# -------------------------
out_path = image_path / "split_files"
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

        # Get components for file names
        
        # extract the subject id from the file name
        subject_id = split_stem[subject_id_index]

        # extract the chunk info from the file name
        chunk_info = "_".join(split_stem[chunk_info_num_parts:])
        print(chunk_info)

        # If MIP, get all z numbers based on first and last MIP of first and last z
        if mip_mode:
            first_MIP_start_z, first_MIP_end_z = int(split_stem[mip_first_start_index]), int(split_stem[mip_first_end_index])
            last_MIP_start_z, last_MIP_end_z = int(split_stem[mip_last_start_index]), int(split_stem[mip_last_end_index])

            MIP_start_numbers = calculate_z_numbers(first_MIP_start_z, last_MIP_start_z, no_z_planes)
            MIP_end_numbers = calculate_z_numbers(first_MIP_end_z, last_MIP_end_z, no_z_planes)
            print(MIP_start_numbers, MIP_end_numbers)
        # if not MIP, get all z numbers based on first and last z number
        else:
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

            if mip_mode:
                out_name = f"{subject_id}_MIP_{MIP_start_numbers[z]}_{MIP_end_numbers[z]}_{chunk_info}"
            else:
                out_name = f"{subject_id}_{section_numbers[z]}_{chunk_info}"

            z_img_filename = out_path / f"{out_name}.tif"
            tifffile.imwrite(z_img_filename, img_z)

            z_npy_filename = out_path / f"{out_name}_seg.npy"
            np.save(z_npy_filename, npy_z)
    else:
        print(f"No .npy file for {tif_file.stem}. Skipping.")
