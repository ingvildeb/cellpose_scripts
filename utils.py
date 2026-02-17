import numpy as np
import tifffile
from pathlib import Path
import random


def calculate_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU for two boolean masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return (intersection / union) if union > 0 else 0.0


def match_instances_iou(manual_masks: np.ndarray,
                        predicted_masks: np.ndarray,
                        iou_threshold: float = 0.5):
    """
    Greedy 1-to-1 matching by IoU (descending).
    Returns:
      matches: list of (gt_id, pred_id, iou)
      FP_ids: list of unmatched pred ids
      FN_ids: list of unmatched gt ids
    """
    gt_ids = [i for i in np.unique(manual_masks) if i != 0]
    pr_ids = [i for i in np.unique(predicted_masks) if i != 0]

    # Build candidate pairs (only those with nonzero intersection)
    pairs = []
    for gt_id in gt_ids:
        gt = (manual_masks == gt_id)
        for pr_id in pr_ids:
            pr = (predicted_masks == pr_id)

            # Quick skip if no overlap (saves time)
            if not np.any(np.logical_and(gt, pr)):
                continue

            iou = calculate_iou(gt, pr)
            pairs.append((iou, gt_id, pr_id))

    # Sort by IoU descending
    pairs.sort(reverse=True, key=lambda x: x[0])

    matched_gt = set()
    matched_pr = set()
    matches = []

    # Greedy assignment
    for iou, gt_id, pr_id in pairs:
        if iou < iou_threshold:
            break
        if gt_id in matched_gt or pr_id in matched_pr:
            continue
        matched_gt.add(gt_id)
        matched_pr.add(pr_id)
        matches.append((gt_id, pr_id, iou))

    FP_ids = [pr_id for pr_id in pr_ids if pr_id not in matched_pr]
    FN_ids = [gt_id for gt_id in gt_ids if gt_id not in matched_gt]

    return matches, FP_ids, FN_ids


def label_to_random_color(label_img, color_lut=None, seed=0, bg_color=(0, 0, 0)):
    """
    If color_lut is provided: dict {label_id: (r,g,b)} used directly.
    Else: generate random colors with seed and return (rgb, color_lut).
    """
    labels = np.unique(label_img)
    labels = labels[labels != 0]

    if color_lut is None:
        rng = np.random.default_rng(seed)
        color_lut = {int(lab): tuple(rng.integers(0, 255, size=3, dtype=np.uint8).tolist())
                     for lab in labels}
    else:
        # make sure any new labels get a color (rare, but safe)
        missing = [int(l) for l in labels if int(l) not in color_lut]
        if missing:
            rng = np.random.default_rng(seed)
            for lab in missing:
                color_lut[lab] = tuple(rng.integers(0, 255, size=3, dtype=np.uint8).tolist())

    rgb = np.zeros((*label_img.shape, 3), dtype=np.uint8)
    rgb[:] = bg_color
    for lab, col in color_lut.items():
        rgb[label_img == lab] = col

    return rgb, color_lut

def instance_centroid(mask: np.ndarray) -> tuple[int, int]:
    """
    Centroid of a boolean mask in (row, col). Uses mean of pixel coordinates.
    Rounds to nearest integer.
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        return (0, 0)
    r = int(np.round(coords[:, 0].mean()))
    c = int(np.round(coords[:, 1].mean()))
    return (r, c)


def centroid_inside_gt_matching(manual_masks: np.ndarray,
                                predicted_masks: np.ndarray):
    """
    Detection-style matching:
    A predicted instance is TP if its centroid falls inside ANY GT instance.
    1-to-1 enforced: each GT can be matched to at most one predicted instance.

    Returns:
      matches_centroid: list of (gt_id, pred_id, (r, c))
      FP_pred_ids: list of predicted ids not matched
      FN_gt_ids: list of GT ids not matched
    """
    # Get all unique IDs in gt and pred files, excluding 0 as this is background
    gt_ids = [i for i in np.unique(manual_masks) if i != 0]
    pr_ids = [i for i in np.unique(predicted_masks) if i != 0]

    matched_gt = set()
    matches_centroid = []
    FP_pred_ids = []

    H, W = manual_masks.shape

    for pr_id in pr_ids:
        pr = (predicted_masks == pr_id)
        
        # Get row and column value of the pr centroid
        r, c = instance_centroid(pr)

        # Clip to image bounds; ensures centroid is inside the image
        r = max(0, min(H - 1, r))
        c = max(0, min(W - 1, c))

        # Get the value of the corresponding pixel in the gt_image
        gt_id = int(manual_masks[r, c])

        # Check whether gt_id is not 0 (i.e. it is an object) and append to matches_centroids list if it is not already there
        if gt_id > 0 and gt_id not in matched_gt:
            matched_gt.add(gt_id)
            matches_centroid.append((gt_id, pr_id, (r, c)))
            
        # If gt_id on the centroid_coordinate is 0 or if it was already matched to a prediction, append to FP list
        else:
            FP_pred_ids.append(pr_id)
            
    # Append any ids that are not TP or FP to FN
    FN_gt_ids = [gt_id for gt_id in gt_ids if gt_id not in matched_gt]

    return matches_centroid, FP_pred_ids, FN_gt_ids

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

