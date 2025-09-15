from pathlib import Path
from cellpose import models, io, train
from datetime import date, datetime
import pandas as pd

# Define directories using pathlib
train_dir = Path(r"Z:\Labmembers\Ingvild\Cellpose\Iba1_model\4_train")
test_dir = Path(r"Z:\Labmembers\Ingvild\Cellpose\Iba1_model\5_validation")

# Specify your hyperparameters
n_epochs = 200
weight_decay = 0.1
learning_rate = 1e-6
normalize = True

# Give a descriptor for your model, typically a name for the signal
model_name = "iba1"

## MAIN CODE, do not edit

timestamp = str(date.today())
start_time = datetime.now()

# Define output directory
out_dir = train_dir / "training_logs"
out_dir.mkdir(parents=True, exist_ok=True)  # Make sure the output directory exists
log_out = out_dir / "training_record.csv"

if log_out.exists():
    log_df = pd.read_csv(log_out)
else:
    log_df = pd.DataFrame()


io.logger_setup()

# Convert train_dir and test_dir to string
output = io.load_train_test_data(str(train_dir), str(test_dir), 
                                  mask_filter="_seg.npy", look_one_level_down=False)

images, labels, image_names, test_images, test_labels, image_names_test = output

model = models.CellposeModel(gpu=True)

# Define model path
model_folder = train_dir / "models" 
model_folder.mkdir(exist_ok=True)

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

end_time = datetime.now()
elapsed_time = end_time - start_time

# SAVE TXT FILE WITH LOSSES

# Specify the filename
filename = out_dir / "model_loss_eval" / f"{model_out_name}_trainAndTestLosses.txt"

# Open the file in write mode
with filename.open('w') as f:
    # Write the header
    f.write("Epoch,Training Loss,Test Loss\n")
    # Write the data
    for epoch in range(len(train_losses)):
        f.write(f"{epoch},{train_losses[epoch]},{test_losses[epoch]}\n")

print(f"Losses saved to {filename}")

# SAVE ALL THE TRAINING INFO TO TRAINING RECORD

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

# SAVE LOG FILES LISTING THE TRAIN AND TEST UIMAGES
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

