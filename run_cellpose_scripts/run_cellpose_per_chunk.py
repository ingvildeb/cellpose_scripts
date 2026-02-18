from pathlib import Path
import sys
import numpy as np
import tifffile as tiff
from cellpose import io, models
from tqdm import tqdm

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from utils.io_helpers import load_script_config, normalize_user_path, require_dir, require_file

# -------------------------
# CONFIG LOADING (shared helper)
# -------------------------
cfg = load_script_config(Path(__file__), "run_cellpose_per_chunk_config")

# -------------------------
# CONFIG PARAMETERS
# -------------------------
model_path = require_file(normalize_user_path(cfg["model_path"]), "Cellpose model path")
chunk_dir = require_dir(normalize_user_path(cfg["chunk_dir"]), "Chunk directory")
out_path = normalize_user_path(cfg["out_path"])

reconstruct = cfg["reconstruct"]
orig_image_path = normalize_user_path(cfg.get("orig_image_path", "")) if cfg.get("orig_image_path") else None

flow_threshold = cfg["flow_threshold"]
normalize = cfg["normalize"]
use_gpu = cfg["use_gpu"]

# -------------------------
# MAIN CODE
# -------------------------
chunk_files = sorted(chunk_dir.glob("*.tif"))
model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))

if reconstruct:
    if orig_image_path is None:
        raise RuntimeError("orig_image_path must be set when reconstruct=true")

    orig_image = tiff.imread(require_file(orig_image_path, "Original image path"))
    expected_shape = orig_image.shape
    out_path.mkdir(exist_ok=True, parents=True)

    reconstructed_image = np.zeros(expected_shape, dtype=np.uint16)
    for chunk_file in tqdm(chunk_files, desc="Processing chunks", unit="chunk"):
        img = io.imread(chunk_file)
        predicted_masks, _, _ = model.eval(img, flow_threshold=flow_threshold, normalize=normalize)

        parts = chunk_file.stem.split("_")
        y = int(parts[-2])
        x = int(parts[-1])

        end_y = min(y + img.shape[0], reconstructed_image.shape[0])
        end_x = min(x + img.shape[1], reconstructed_image.shape[1])

        reconstructed_image[y:end_y, x:end_x] = predicted_masks[: end_y - y, : end_x - x]

    tiff.imwrite(out_path / f"reconstructed_{chunk_dir.stem}.tif", reconstructed_image)

else:
    for chunk_file in tqdm(chunk_files, desc="Processing chunks", unit="chunk"):
        img = io.imread(chunk_file)
        predicted_masks, flows, _ = model.eval(img, flow_threshold=flow_threshold, normalize=normalize)
        tiff.imwrite(chunk_file.parent / f"{chunk_file.stem}_masks.tif", predicted_masks)
        io.masks_flows_to_seg(img, predicted_masks, flows, chunk_file)
