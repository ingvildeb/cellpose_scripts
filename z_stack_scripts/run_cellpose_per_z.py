from pathlib import Path
import random

import numpy as np
import tifffile
from cellpose import io, models
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from utils.io_helpers import load_script_config, normalize_user_path, require_dir, require_file
from utils.utils import generate_cellpose_npy_dict

# -------------------------
# CONFIG LOADING (shared helper)
# -------------------------
cfg = load_script_config(Path(__file__), "run_cellpose_per_z_config")

# -------------------------
# CONFIG PARAMETERS
# -------------------------
model_path = require_file(normalize_user_path(cfg["model_path"]), "Cellpose model path")
stack_dir = require_dir(normalize_user_path(cfg["stack_dir"]), "Stack directory")

flow_threshold = cfg["flow_threshold"]
normalize = cfg["normalize"]
use_gpu = cfg["use_gpu"]

# -------------------------
# MAIN CODE
# -------------------------
print("Loading cellpose model ...")
model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))

out_path = stack_dir / "individual_z_predictions"
out_path.mkdir(exist_ok=True)

for stack in sorted(stack_dir.glob("*.tif")):
    print(f"Processing stack with name {stack.stem} ...")

    all_masks = []
    all_outlines = []

    image_array = tifffile.TiffFile(require_file(stack, "Stack tif")).asarray()

    if len(image_array.shape) != 3:
        print(f"{stack.stem} is not a stack with dimensions z * W * H. Skipping...")
        continue

    num_masks = 0
    for z in range(image_array.shape[0]):
        img_z = image_array[z]
        z_img_out = out_path / f"{stack.stem}_{z}.tif"
        tifffile.imwrite(z_img_out, img_z)

        predicted_masks, flows, _ = model.eval(
            img_z,
            flow_threshold=flow_threshold,
            normalize=normalize,
        )
        num_masks += len(np.unique(predicted_masks)) - 1

        io.masks_flows_to_seg(img_z, predicted_masks, flows, z_img_out)
        npy_file = z_img_out.parent / f"{z_img_out.stem}_seg.npy"

        data = np.load(require_file(npy_file, "Z-plane seg npy"), allow_pickle=True).item()
        masks, outlines = data["masks"], data["outlines"]

        all_masks.append(masks)
        all_outlines.append(outlines)

    composite_masks = np.stack(all_masks, axis=0)
    composite_outlines = np.stack(all_outlines, axis=0)

    all_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(num_masks)]
    npy_dict = generate_cellpose_npy_dict(composite_masks, composite_outlines, all_colors, stack)

    np.save(stack_dir / f"{stack.stem}_seg.npy", npy_dict)
