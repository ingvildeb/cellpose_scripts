from pathlib import Path

import tifffile as tiff
from cellpose import io, models

from io_helpers import load_script_config, normalize_user_path, require_dir, require_file

# -------------------------
# CONFIG LOADING (shared helper)
# -------------------------
cfg = load_script_config(Path(__file__), "run_cellpose_per_image_config")

# -------------------------
# CONFIG PARAMETERS
# -------------------------
model_path = require_file(normalize_user_path(cfg["model_path"]), "Cellpose model path")
input_path = normalize_user_path(cfg["input_path"])
out_path = normalize_user_path(cfg["out_path"])

flow_threshold = cfg["flow_threshold"]
normalize = cfg["normalize"]
use_gpu = cfg["use_gpu"]

# -------------------------
# MAIN CODE
# -------------------------
out_path.mkdir(exist_ok=True, parents=True)
model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))

if input_path.is_dir():
    input_dir = require_dir(input_path, "Input directory")
    for f in sorted(input_dir.glob("*.tif")):
        img = io.imread(f)
        predicted_masks, _, _ = model.eval(img, flow_threshold=flow_threshold, normalize=normalize)
        tiff.imwrite(out_path / f"masks_{f.stem}.tif", predicted_masks)

elif input_path.is_file():
    input_file = require_file(input_path, "Input tif")
    img = io.imread(input_file)
    predicted_masks, _, _ = model.eval(img, flow_threshold=flow_threshold, normalize=normalize)
    tiff.imwrite(out_path / f"predictions_{input_file.stem}.tif", predicted_masks)

else:
    raise RuntimeError("input_path must be a tif file or a directory containing tif files")
