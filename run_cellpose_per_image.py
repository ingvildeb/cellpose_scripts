from pathlib import Path
import tifffile as tiff
from cellpose import models, io, transforms
import numpy as np

## USER SETTINGS

# Set the path to your cellpose model.
model_path = r"Z:\Labmembers\Ingvild\Cellpose\MOBP_model\4_train\models\2025-10-06_cpsam_MOBP_500epochs_wd-0.1_lr-1e-05_normTrue_84train-images"
#model_path = r"example\path\your_model"

# Set the path to the input image(s). Can be a single tif file or a folder with tif images.
input = Path(r"Z:\Labmembers\Ingvild\Cellpose\MOBP_model\test_application")
#input = Path(r"example\path\your_path")

# Set the path where you want the output to be stored.
out_path = Path(r"Z:\Labmembers\Ingvild\Cellpose\MOBP_model\test_application")
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