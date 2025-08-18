import numpy as np
import tifffile
from pathlib import Path
import random


def calculate_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    iou = intersection / union if union > 0 else 0
    return iou


def split_z_stack(tif_file):
    img = tifffile.TiffFile(tif_file)
    image_array = img.asarray()
    print(f"{image_array.shape[0]} z planes found")

    # Load the corresponding .npy file
    npy_file = tif_file.parent / f"{tif_file.stem}_seg.npy"
    data = np.load(npy_file, allow_pickle=True).item()
    
    img_z_list = []
    npy_z_list = []

    # Loop z planes found in the .tif file
    for z in range(image_array.shape[0]):
        # Get image array from the z plane
        img_z = image_array[z]

        # Extract masks and annotations from the z plane
        z_masks = data['masks'][z]
        z_annos = data['outlines'][z]

        # Calculate the number of masks in the image (subtract one due to background value being 0)
        mask_number = len(np.unique(z_masks)) - 1

        # Create color lists for the number of masks in the z plane
        z_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(mask_number)]

        # Create a dictionary to hold the z-plane data and masks
        npy_z = {
            'masks': z_masks,          # Update with the current z-plane mask
            'outlines': z_annos,       # Update with the current z-plane outlines
            'colors': z_colors,         # Use generated random colors
            'current_channel': data['current_channel'],
            'filename': data['filename'],
            'flows': data['flows'],
            'zdraw': data['zdraw'],
            'model_path': data['model_path'],
            'flow_threshold': data['flow_threshold'],
            'cellprob_threshold': data['cellprob_threshold'],
            'normalize_params': data['normalize_params'],
            'restore': data['restore'],
            'ratio': data['ratio'],
            'diameter': data['diameter']
        }

        img_z_list.append(img_z)
        npy_z_list.append(npy_z)

    return img_z_list, npy_z_list, image_array.shape[0]


def calculate_z_numbers(first_z_number, last_z_number, no_z_planes):
    
    distance = last_z_number - first_z_number
    step = distance / (no_z_planes - 1)
    numbers = [str(int(first_z_number + i * step)).zfill(6) for i in range(no_z_planes)]
    
    return numbers