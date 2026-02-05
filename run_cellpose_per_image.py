from pathlib import Path
import tifffile as tiff
from cellpose import models, io, transforms
import numpy as np

## USER SETTINGS

# Set the path to your cellpose model.
model_path = r"Z:\Labmembers\Ingvild\Cellpose\Iba1_model\5_train\models\2025-09-20_cpsam_iba1_1000epochs_wd-0.1_lr-1e-05_normTrue"
#model_path = r"example\path\your_model"

# Set the path to the input image(s). Can be a single tif file or a folder with tif images.
input = Path(r"Z:\Labmembers\Ingvild\RM1\HPC_scripts\testing\hpc_vs_local\masks_and_coords\test_MIPS")
#input = Path(r"example\path\your_path")

# Set the path where you want the output to be stored.
out_path = Path(r"Z:\Labmembers\Ingvild\RM1\HPC_scripts\testing\hpc_vs_local\masks_and_coords\masks_32c_pythonIngvild\\")
#out_path = Path(r"example\path\your_out_path")

# Cellpose parameters

# Choose a flow threshold. The default is 0.4. Increasing flow_threshold makes Cellpose more lenient, 
# This can lead to the detection of more cells, especially in crowded or complex images, but may also 
# result in less accurate or ill-shaped masks.
flow_threshold = 0.4

# Set normalize to True or False. Should normally be True.
normalize = True

## MAIN CODE
out_path.mkdir(exist_ok=True)

model = models.CellposeModel(gpu=True, pretrained_model=model_path)

if input.is_dir():

    flist = input.glob("*.tif")

    for f in flist:
        img = io.imread(f)
        #model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
        predicted_masks, _, _ = model.eval(img, flow_threshold=flow_threshold, normalize=normalize)

        fname = f.stem
        tiff.imwrite(out_path / f"masks_{fname}.tif", predicted_masks)

elif input.is_file():
        
        img = io.imread(input)
        #model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
        predicted_masks, _, _ = model.eval(img, flow_threshold=flow_threshold, normalize=normalize)

        fname = input.stem
        tiff.imwrite(out_path / f"predictions_{fname}.tif", predicted_masks)

else:
     print("Input must be either a file or a folder. Check your input setting.")