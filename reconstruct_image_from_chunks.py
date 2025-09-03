import os
import numpy as np
import tifffile
from pathlib import Path

def reconstruct_image(chunk_dir, output_dir, image_name, expected_shape):
    reconstructed_image = np.zeros(expected_shape, dtype=np.uint16)  # Adjust dtype as needed

    # List all chunk files in the chunk directory
    chunk_files = [f for f in os.listdir(chunk_dir) if f.endswith('.tif')]

    for chunk_file in chunk_files:
        # Parse the filename to get chunk coordinates
        parts = chunk_file.split('_')

        # Extract y (row) and x (column) from filename
        y = int(parts[-2])  # y-position
        x = int(parts[-1].split('.')[0])  # x-position

        # Load the chunk
        chunk_path = chunk_dir / chunk_file
        chunk = tifffile.imread(chunk_path)
        
        # Calculate boundaries for placing into the reconstructed image
        start_y = y
        start_x = x
        end_y = start_y + chunk.shape[0]
        end_x = start_x + chunk.shape[1]

        # Ensure we do not go out of bounds when placing the chunk
        end_y = min(end_y, reconstructed_image.shape[0])
        end_x = min(end_x, reconstructed_image.shape[1])

        # Check if the chunk can fit in the reconstructed image
        reconstructed_image[start_y:end_y, start_x:end_x] = chunk[:end_y - start_y, :end_x - start_x]

    # Save the reconstructed image
    output_path = os.path.join(output_dir, f"reconstructed_{image_name}.tif")
    tifffile.imwrite(output_path, reconstructed_image)
    print(f"Reconstructed image saved at {output_path}")



# Example usage
chunk_dir = Path(r'Z:\Labmembers\Ingvild\Cellpose\NeuN_model\model0_training_sections\chunked_images\NB058_MIP_021000_021080\\')
image_name = chunk_dir.stem
orig_image_path = chunk_dir.parent.parent / f"{image_name}.tif"
orig_image = tifffile.imread(orig_image_path)
expected_shape = orig_image.shape

reconstruct_image(chunk_dir=chunk_dir, 
                  output_dir=chunk_dir.parent, 
                  image_name=image_name,
                  expected_shape=expected_shape)
