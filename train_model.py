from pathlib import Path
from cellpose import models, io, train
import matplotlib.pyplot as plt

# Define directories using pathlib
train_dir = Path(r"Z:\Labmembers\Ingvild\Cellpose\NeuN_model\manual_and_human-in-the-loop\train")
test_dir = Path(r"Z:\Labmembers\Ingvild\Cellpose\NeuN_model\manual_and_human-in-the-loop\validation")

# Specify your hyperparameters
n_epochs = 100
weight_decay = 0.1
learning_rate = 1e-6
normalize = True

# Define output directory
out_dir = train_dir / "training_logs"
out_dir.mkdir(parents=True, exist_ok=True)  # Make sure the output directory exists

io.logger_setup()

# Convert train_dir and test_dir to string
output = io.load_train_test_data(str(train_dir), str(test_dir), 
                                  mask_filter="_seg.npy", look_one_level_down=False)

images, labels, image_names, test_images, test_labels, image_names_test = output

model = models.CellposeModel(gpu=True)

# Define model path
model_path = train_dir / "models" / f"cpsam_neun_{n_epochs}epochs_wd-{weight_decay}_lr-{learning_rate}_norm{normalize}"

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

# Specify the filename
filename = out_dir / f"cpsam_neun_{n_epochs}epochs_wd-{weight_decay}_lr-{learning_rate}_norm{normalize}_trainAndTestLosses.txt"

# Open the file in write mode
with filename.open('w') as f:
    # Write the header
    f.write("Epoch,Training Loss,Test Loss\n")
    # Write the data
    for epoch in range(len(train_losses)):
        f.write(f"{epoch},{train_losses[epoch]},{test_losses[epoch]}\n")

print(f"Losses saved to {filename}")

# Plot the train and test losses
train_losses = []
test_losses_dict = {}

# Open the file in read mode
with filename.open('r') as f:
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
plt.plot(epochs, train_losses, label='Training Loss', marker='o', linestyle='-')

# Plot Test Loss; connect test loss points using the dictionary
plt.plot(test_epochs, test_losses, label='Test Loss', marker='o', linestyle='-', color='orange')

# Adding labels and title
plt.title(f'Model: cpsam_neun_{n_epochs}epochs_wd-{weight_decay}_lr-{learning_rate}_norm{normalize}')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Set x-ticks to show every 5 epochs (modify as needed)
xticks = range(0, len(train_losses), 5)  # Every 5 epochs
plt.xticks(xticks)  # Set x-ticks to correspond with the specified epochs

plt.legend()
plt.grid(True)  # Adds a grid for better readability

# Save the plot
plt.savefig(out_dir / f"cpsam_neun_{n_epochs}epochs_wd-{weight_decay}_lr-{learning_rate}_norm{normalize}_trainAndTestLosses.svg")

# Show the plot
plt.show()
