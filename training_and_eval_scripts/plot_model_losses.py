import matplotlib.pyplot as plt
from pathlib import Path

"""
PLOT CELLPOSE MODEL LOSSES TO A GRAPH

This script allows you to plot the losses of a trained model. The script uses the loss file saved when running the
train_model.py script.

Please note that you will have problems if you changed your folder structure after running train_model.py.
If needed, you can specify the path of your loss file directly on line 27. However, I recommend keeping the folder
structure and letting the script fetch it automatically to avoid any mismatching of model name and losses.

"""

# Set the path to your model
model_path = Path(r"example\path\your_model")

# Choose what interval of epoch numbers to plot along the x axis.
# Set to higher interval to avoid very crowded axis with high number of epochs
plot_every = 50

# Choose the colors of your train and test loss lines
train_color = 'blue'
test_color = 'orange'



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