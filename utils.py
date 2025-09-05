import numpy as np
import tifffile
from pathlib import Path
import random

def calculate_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    iou = intersection / union if union > 0 else 0
    return iou

def generate_cellpose_npy_dict(masks, outlines, colors, filepath):
        
    npy_dict = {'masks': masks,
                'outlines': outlines,
                'colors': colors,
                'filename': str(filepath),
                'flow_threshold': 0.4,
                'cellprob_threshold': 0.0,
                'normalize_params': {'lowhigh': None,
                'percentile': [1.0, 99.0],
                'normalize': True,
                'norm3D': True,
                'sharpen_radius': 0.0,
                'smooth_radius': 0.0,
                'tile_norm_blocksize': 0.0,
                'tile_norm_smooth3D': 0.0,
                'invert': False},
                'restore': None,
                'ratio': 1.0,
                'diameter': None}
    
    return npy_dict

def split_annotated_z_stack(tif_file):
    # open tif image
    img = tifffile.TiffFile(tif_file)
    image_array = img.asarray()   
  
    img_z_list = []
    npy_z_list = []
    
    # Load the corresponding .npy file
    npy_file = tif_file.parent / f"{tif_file.stem}_seg.npy"
    data = np.load(npy_file, allow_pickle=True).item()

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
        npy_z = generate_cellpose_npy_dict(z_masks, z_annos, z_colors, data['filename'])
       
        img_z_list.append(img_z)
        npy_z_list.append(npy_z)

    return img_z_list, npy_z_list, image_array.shape[0]


def calculate_z_numbers(first_z_number, last_z_number, no_z_planes):
    
    distance = last_z_number - first_z_number
    step = distance / (no_z_planes - 1)
    numbers = [str(int(first_z_number + i * step)).zfill(6) for i in range(no_z_planes)]
    
    return numbers


def chunk_image(path_to_image, image_outdir, chunk_size):
    # read the image
    img = tifffile.TiffFile(path_to_image).asarray()
    image_name = path_to_image.stem

    # get the shape of the image
    shape = img.shape
    
    # crop the image and save each chunk
    for i in range(0, shape[0], chunk_size):
        for j in range(0, shape[1], chunk_size):
            chunk = img[i:i+chunk_size, j:j+chunk_size]
            tifffile.imsave("{}/{}_chunk_{}_{}.tif".format(image_outdir,image_name,i, j), chunk)

