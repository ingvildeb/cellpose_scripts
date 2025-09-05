from pathlib import Path
import tifffile as tiff
from cellpose import models, io, transforms
import numpy as np

## USER SETTINGS
model_path = r"Z:/Labmembers/Ingvild/Cellpose/NeuN_model/manual_and_human-in-the-loop/train/models/cpsam_neun_100epochs_wd-0.1_lr-1e-06_normTrue"
input = Path(r"Z:\Labmembers\Ingvild\Testing_CellPose\demo_tiling_issues\\")
out_path = Path(r"Z:\Labmembers\Ingvild\Testing_CellPose\demo_tiling_issues\with_python_code\\")

flow_threshold = 0.4
normalize = True

## MAIN CODE, do not edit
out_path.mkdir(exist_ok=True)

model = models.CellposeModel(gpu=True, pretrained_model=model_path)

if input.is_dir():

    flist = input.glob("*.tif")

    for f in flist:
        img = io.imread(f)
        model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
        predicted_masks, _, _ = model.eval(img, flow_threshold=flow_threshold, normalize=normalize)

        fname = f.stem
        tiff.imwrite(out_path / f"predictions_{fname}.tif", predicted_masks.astype(np.uint8))

elif input.is_file():
        
        img = io.imread(input)
        model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
        predicted_masks, _, _ = model.eval(img, flow_threshold=flow_threshold, normalize=normalize)

        fname = input.stem
        tiff.imwrite(out_path / f"predictions_{fname}.tif", predicted_masks.astype(np.uint8))

else:
     print("Input must be either a file or a folder. Check your input setting.")