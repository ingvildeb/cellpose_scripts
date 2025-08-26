from pathlib import Path
import tifffile as tiff
from cellpose import models, io, transforms
import numpy as np

## USER SETTINGS
model_path = r"Z:\Labmembers\Ingvild\Cellpose\NeuN_model\manual_and_human-in-the-loop\train\models\cpsam_neun_100epochs_wd-0.1_lr-1e-06_normTrue"
input = Path(r"Z:\Labmembers\Ingvild\Cellpose\NeuN_model\test_run_and_reconstruct\IEB0029_MIP_027360_027420.tif\\")
out_path = Path(r"Z:\Labmembers\Ingvild\Cellpose\NeuN_model\test_run_and_reconstruct\\")

flow_threshold = 0.4
normalize = True
single_or_folder = "single"


## MAIN CODE, do not edit
out_path.mkdir(exist_ok=True)

model = models.CellposeModel(gpu=True, pretrained_model=model_path)

if single_or_folder == "folder":

    flist = input.glob("*.tif")

    for f in flist:
        img = io.imread(f)
        model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
        predicted_masks, _, _ = model.eval(img, flow_threshold=flow_threshold, normalize=normalize)

        fname = f.stem
        tiff.imwrite(out_path / f"predictions_{fname}.tif", predicted_masks.astype(np.uint8))

elif single_or_folder == "single":
        
        img = io.imread(input)
        model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
        predicted_masks, _, _ = model.eval(img, flow_threshold=flow_threshold, normalize=normalize)

        fname = input.stem
        tiff.imwrite(out_path / f"predictions_{fname}.tif", predicted_masks.astype(np.uint8))

else:
     print("single_or_folder variable must be either single or folder")