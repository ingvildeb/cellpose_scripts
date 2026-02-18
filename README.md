# cellpose_scripts

This repository contains scripts to work with cellpose. The scripts were primarily developed to work with Light Sheet Fluorescence Microscopy (LSFM) data from the mouse brain, but may be useful for other purposes as well. The scripts have been made and optimized using chunked images from 2D sections or chunked Z stacks from a range of 2D sections.

## Script to train and evaluate a cellpose model
**- train_model.py:** Script to train a cellpose model on training data, with validation data included. Saves losses to a txt file and creates a csv log file (training_record.csv) that will update every time you train a new model, to help you keep track of your model versions. 

**- plot_model_losses.py:** Script to plot train and test loss for your trained model. Saves an svg file with the plot.

**- calculate_model_performance.py:** Script to calculate the precision, recall and F1 score of your trained model. The user inputs a set of validation images to test the model on. Outputs: (1) an .svg file for each validation image, showing the image, ground truth labels, predicted labels, true positives, false positives and false negatives, (2) a metrics summary report with the precision, recall and number of labels per validation image. The script also saves the average metrics to the training_record.csv file.

## Scripts for running a cellpose model on section images
**- run_cellpose_per_image.py:** Script that allows you to run a cellpose model on tif images. The user can input a single image or a folder of images, and adjust flow threshold and normalization behavior as desired.

**- run_cellpose_per_chunk.py:** Script that runs a cellpose model on chunks from an image and reconstructs them into the full image. This script was created because the latest version of cellpose (v4.0) shows some strange behavior when applying a model trained on image chunks to a whole image (see https://github.com/MouseLand/cellpose/issues/1276#issuecomment-3285468852). In our experience, this behavior is problematic for images with very dense stains (such as NeuN, Sytox) and less prominent with more distributed signals. This code provides a workaround that is incredibly slow, but can be used to assess whether the model performs very different on chunks versus the whole image.

## Scripts to work with Z stacks in cellpose while still training and applying a 2D model
Sometimes, we find that it is useful for the human to work with a Z stack when labelling data for cellpose. This allows the user to navigate between planes to better see if a profile should be labelled or not. However, training and applying a cellpose model in 3D for large datasets like those from the brain is complex and time consuming. Therefore, we still want to train and apply our models in 2D. These scripts allows such a process.

**- split_annotated_stacks.py:** Split an annotated z stack so that individual planes can be used for 2D training.

**- run_cellpose_per_z.py:** Allows you to run a cellpose model on each plane of a z stack individually, and puts them back together into an annotated Z stack. Useful for human-in-the-loop labelling with the Z stack labelling-2D training workflow.

## Getting started
I recommend running these scripts from your cellpose environment and using the GPU. Instructions for cellpose installation can be found in their Github https://github.com/MouseLand/cellpose. Most of the neccessary packages required to run scripts in this repository will be installed when installing cellpose.

## TOML configuration workflow
The scripts use shared config loading from `io_helpers.py`, matching the local/template pattern:

- Scripts load config automatically using:
  - `configs/<script_config_name>_local.toml` (preferred)
  - `configs/<script_config_name>_template.toml` (fallback)
- Keep `_template.toml` files in git.
- Copy them to `_local.toml` for machine-specific paths.

Configured scripts:
- `run_cellpose_per_image.py` → `run_cellpose_per_image_config_local.toml`
- `run_cellpose_per_chunk.py` → `run_cellpose_per_chunk_config_local.toml`
- `run_cellpose_per_z.py` → `run_cellpose_per_z_config_local.toml`
- `split_annotated_stacks.py` → `split_annotated_stacks_config_local.toml`
- `train_model.py` → `train_model_config_local.toml`
- `plot_model_losses.py` → `plot_model_losses_config_local.toml`
- `calculate_model_performance.py` → `calculate_model_performance_config_local.toml`
