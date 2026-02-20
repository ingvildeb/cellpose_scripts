"""
Plot training and test loss curves for a trained Cellpose model.

Config usage:
- Copy `training_and_eval_scripts/configs/plot_model_losses_config_template.toml` to
  `training_and_eval_scripts/configs/plot_model_losses_config_local.toml`.
- Edit `_local.toml` to your preferred settings and run the script.
- If `_local.toml` is missing, the script falls back to `_template.toml`.
"""

import matplotlib.pyplot as plt
from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from utils.io_helpers import load_script_config, normalize_user_path

# -------------------------
# CONFIG LOADING (shared helper)
# -------------------------
test_mode = False
cfg = load_script_config(Path(__file__), "plot_model_losses_config", test_mode=test_mode)

# -------------------------
# CONFIG PARAMETERS
# -------------------------
model_path = normalize_user_path(cfg["model_path"])

# Choose what interval of epoch numbers to plot along the x axis.
# Set to higher interval to avoid very crowded axis with high number of epochs
plot_every = cfg["plot_every"]

# Choose the colors of your train and test loss lines
train_color = cfg["train_color"]
test_color = cfg["test_color"]



### MAIN CODE
loss_dir = model_path.parent.parent / "training_logs" / "model_loss_eval"

# Edit with caution if you changed the folder structure after running train_model.py
loss_file =  loss_dir / f"{model_path.name}_trainAndTestLosses.txt"

# Plot the train and test losses
train_losses = []
test_losses_dict = {}

# Open the file in read mode
with loss_file.open('r') as f:
    next(f)  # Skip the header line
    for line in f:
        epoch, train_loss, test_loss = line.strip().split(',')
        epoch = int(epoch)  # Ensure epoch is an integer
        train_losses.append(float(train_loss))

        if float(test_loss) != 0:
            test_losses_dict[epoch] = float(test_loss)

# Prepare data for plotting
test_epochs = list(test_losses_dict.keys())
test_losses = list(test_losses_dict.values())

# Epochs for training loss
epochs = range(len(train_losses))

# Plotting
plt.figure(figsize=(10, 5))

# Plot Training Loss
plt.plot(epochs, train_losses, label='Training Loss', marker='o', linestyle='-', color=train_color)

# Plot Test Loss; connect test loss points using the dictionary
plt.plot(test_epochs, test_losses, label='Test Loss', marker='o', linestyle='-', color=test_color)

# Adding labels and title
plt.title(f'Model: {model_path.name}')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Set x-ticks to show every 5 epochs (modify as needed)
xticks = range(0, len(train_losses), plot_every)  # Every 5 epochs
plt.xticks(xticks)  # Set x-ticks to correspond with the specified epochs

plt.legend()
plt.grid(True)  # Adds a grid for better readability

# Save the plot
plt.savefig(loss_dir / f"{model_path.name}_trainAndTestLosses.svg")

# Show the plot
plt.show()
