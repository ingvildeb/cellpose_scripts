from pathlib import Path
from cellpose import models, io, train
from datetime import date, datetime
import pandas as pd


"""
CELLPOSE MODEL TRAINING AND DOCUMENTATION SCRIPT

This script allows you to train a cellpose model with a train and test set specified.
Simply input your directories and desired hyperparameters under user settings and run the whole script.

The script will generate a set of outputs in a folder called "training_logs" located in your training directory:

images_per_model_logs: folder with csv logs of which files were used for training and validation of the model. Just
for documentation purposes, may or may not be useful later.

model_loss_eval: folder with txt files containing the train and test loss for the model. You can create a graph of
the losses across epochs with this txt file using the plot_model_losses.py script in this repository.

training_log.csv: csv file with metadata about the model training, including hyperparameters, train and test directory.
The training log csv file will have columns for model performance data, which can be acquired using the calculate_model_performance.py
script of this repository. Until you run that script for the trained model, the cells will say "N/A".
"""

# Define directories using pathlib
train_dir = Path(r"example\path\your_train_directory")
test_dir = Path(r"example\path\your_test_directory")

# Specify your hyperparameters

# Specify number of epochs - how many times the model gets to see the training data.
# Recommended setting from cellpose is 100, but longer times may improve performance
n_epochs = 100

# Specify weight decay
weight_decay = 0.1

# Specify learning rate - essentially how quickly the model learns
# Recommended setting from cellpose is 1e-5.
# A lower learning rate may allow you to train for longer without overfitting
learning_rate = 1e-6

# Specify normalization behavior. Should generally be set to True.
normalize = True

# Give a descriptor for your model, typically a name for the signal
model_name = "iba1"


## MAIN CODE, do not edit

# Set up timestamp and starting time
timestamp = str(date.today())
start_time = datetime.now()

# Define output directory and create it if it does not exist
out_dir = train_dir / "training_logs"
out_dir.mkdir(parents=True, exist_ok=True)

# Read log file (if it exists) or create empty dataframe to hold log if not
log_out = out_dir / "training_record.csv"

if log_out.exists():
    log_df = pd.read_csv(log_out)
else:
    log_df = pd.DataFrame()

# Set up for training, load train and test data
io.logger_setup()

# Convert train_dir and test_dir to string
output = io.load_train_test_data(str(train_dir), 
                                 str(test_dir), 
                                 mask_filter="_seg.npy", 
                                 look_one_level_down=False)

images, labels, image_names, test_images, test_labels, image_names_test = output

model = models.CellposeModel(gpu=True)

# Define model path and create it if it does not exist
model_folder = train_dir / "models" 
model_folder.mkdir(exist_ok=True)

# Define the name of the model file with timestamp and hyperparameters included
model_out_name = f"{timestamp}_cpsam_{model_name}_{n_epochs}epochs_wd-{weight_decay}_lr-{learning_rate}_norm{normalize}"
model_path = model_folder / model_out_name

# Train the model
model_path, train_losses, test_losses = train.train_seg(model.net,
                            train_data=images, 
                            train_labels=labels,
                            test_data=test_images, 
                            test_labels=test_labels,
                            weight_decay=weight_decay, 
                            learning_rate=learning_rate,
                            n_epochs=n_epochs, 
                            normalize=normalize,
                            model_name=str(model_path))

# Set end time and calculate elapsed time it took to train the model
end_time = datetime.now()
elapsed_time = end_time - start_time

# Save txt file with the losses

# Specify the directory and create if it does not exist
loss_dir = out_dir / "model_loss_eval"
loss_dir.mkdir(exist_ok=True, parents=True)
filename = loss_dir / f"{model_out_name}_trainAndTestLosses.txt"

# Open the file and write losses
with filename.open('w') as f:
    # Write the header
    f.write("Epoch,Training Loss,Test Loss\n")
    # Write the data
    for epoch in range(len(train_losses)):
        f.write(f"{epoch},{train_losses[epoch]},{test_losses[epoch]}\n")

print(f"Losses saved to {filename}")

# Save all the training information in the log file

# Assign a unique ascending model_number
if log_df.empty:
    model_number = 1
else:
    model_number = log_df['model_number'].max() + 1

# Create a new row dictionary with values you want to log
new_row = {
    "model_number": model_number,
    "date": timestamp,
    "model": model_out_name,
    "epochs": n_epochs,
    "weight_decay": weight_decay,
    "learning rate": learning_rate,
    "normalize_train": normalize,
    "num_train_images": len(images),
    "num_test_images": len(test_images),
    "train_dir": train_dir,
    "test_dir": test_dir,
    "time_to_train": elapsed_time,
    "precision": "N/A",
    "recall": "N/A",
    "F1": "N/A",
    "num_ground_truth_cells": "N/A",
    "flow_threshold": "N/A",
    "normalize_apply": "N/A"
}

log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
log_df.to_csv(log_out, index=False)

# Create log files to store the names of training and test images
# Prepare the test_images_list as a DataFrame with the column named after model_number
img_logs_out = out_dir / "images_per_model_logs"
img_logs_out.mkdir(parents=True, exist_ok=True)

train_images_list = [i.split("\\")[-1] for i in image_names]
train_images_df = pd.DataFrame({f"Train images for model {model_number}": train_images_list})
train_images_out = img_logs_out / f"train_images_model{model_number}_{model_out_name}.csv"

test_images_list = [i.split("\\")[-1] for i in image_names_test]
test_images_df = pd.DataFrame({f"Test images for model {model_number}": test_images_list})
test_images_out = img_logs_out / f"test_images_model{model_number}_{model_out_name}.csv"

# Save log_df and test_images_list to separate sheets in one Excel workbook
train_images_df.to_csv(train_images_out, index=False)
test_images_df.to_csv(test_images_out, index=False)

