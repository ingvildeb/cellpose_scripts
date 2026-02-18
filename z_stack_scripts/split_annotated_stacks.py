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
cfg = load_script_config(Path(__file__), "split_annotated_stacks_config")

# -------------------------
# CONFIG PARAMETERS
# -------------------------
image_path = require_dir(normalize_user_path(cfg["image_path"]), "Image path")
omit_first_and_last = cfg["omit_first_and_last"]

subject_id_index = cfg["subject_id_index"]
chunk_info_num_parts = cfg["chunk_info_num_parts"]

mip_first_start_index = cfg["mip_first_start_index"]
mip_first_end_index = cfg["mip_first_end_index"]
mip_last_start_index = cfg["mip_last_start_index"]
mip_last_end_index = cfg["mip_last_end_index"]

section_start_index = cfg["section_start_index"]
section_end_index = cfg["section_end_index"]

# -------------------------
# MAIN CODE
# -------------------------
out_path = image_path / "split_files"
out_path.mkdir(exist_ok=True)

for tif_file in sorted(image_path.glob("*.tif")):
    npy_file = tif_file.parent / f"{tif_file.stem}_seg.npy"

    if npy_file.exists():
        img_z_list, npy_z_list, no_z_planes = split_annotated_z_stack(tif_file)
        print(f"{no_z_planes} z planes found for {tif_file.stem}")

        split_stem = tif_file.stem.split("_")

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

        subject_id = split_stem[subject_id_index]
        chunk_info = "_".join(split_stem[-chunk_info_num_parts:])

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

        if omit_first_and_last:
            z_range = range(1, len(img_z_list) - 1)
        else:
            z_range = range(len(img_z_list))

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
