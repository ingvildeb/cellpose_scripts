import tifffile
from pathlib import Path
import numpy as np
from cellpose import models, io
import random
from utils import generate_cellpose_npy_dict

## USER SETTINGS
model_path = r"Z:\Labmembers\Ingvild\Cellpose\Iba1_model\4_train\models\2025-09-03_cpsam_iba1_500epochs_wd-0.1_lr-1e-05_normTrue"
stack_dir = Path(r'Z:\Labmembers\Ingvild\Cellpose\Iba1_model\testing_application\split_apply_combine\\')

# Cellpose parameters
flow_threshold = 0.4
normalize = True

## MAIN CODE

# load cellpose model
model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))

# create an out path for individual predictions
out_path = stack_dir / "individual_z_predictions"
out_path.mkdir(exist_ok=True)

# loop through stacks in directory
for stack in stack_dir.glob("*.tif"):
    
    # initialize lists to hold all masks and outlines across z planes
    all_masks = []
    all_outlines = []

    # load array from the tif image
    img = tifffile.TiffFile(stack)
    image_array = img.asarray()

    # check if the image is actually a z stack with the expected number of dimensions
    if len(image_array.shape) != 3:
        print(f"{stack.stem} is not a stack with dimensions z * W * H. Skipping...")

    # continue if dimensions are as expected
    else:
        # initialize counter to hold the number of masks across all z planes
        num_masks = 0

        # loop through z planes in the image
        for z in range(image_array.shape[0]):

            # get image array from the z plane and save to individual z output folder
            img_z = image_array[z]
            z_img_out = out_path / f"{stack.stem}_{z}.tif"
            tifffile.imwrite(z_img_out, img_z)

            # apply model to the z plane image
            predicted_masks, flows, _ = model.eval(img_z, 
                                               flow_threshold=flow_threshold, 
                                               normalize=normalize)
            
            # add number of masks from z planes to count of masks across all planes
            num_masks = num_masks + (len(np.unique(predicted_masks)) - 1)
            
            # generate npy file for the z plane and save to individual z output folder
            io.masks_flows_to_seg(img_z, predicted_masks, flows, z_img_out)
            npy_file = z_img_out.parent / f"{z_img_out.stem}_seg.npy"

            # load the individual z plane npy file to get the masks and outlines
            data = np.load(npy_file, allow_pickle=True).item()
            masks, outlines = data['masks'], data['outlines']

            # append the individual z plane masks and outlines to respective lists
            all_masks.append(masks)
            all_outlines.append(outlines)

        # stack masks and outlines from the respective lists into arrays    
        composite_masks = np.stack(all_masks, axis=0)
        composite_outlines = np.stack(all_outlines, axis=0)

        # generate a list of random colors the same length as the total number of masks
        all_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(num_masks)]

        # generate cellpose compatible npy dictionary
        npy_dict = generate_cellpose_npy_dict(composite_masks, composite_outlines,
                                              all_colors, stack)

        # save z stack npy as _seg.npy file
        npy_out_name = stack_dir / f"{stack.stem}_seg.npy"
        np.save(npy_out_name, npy_dict)

        break
        