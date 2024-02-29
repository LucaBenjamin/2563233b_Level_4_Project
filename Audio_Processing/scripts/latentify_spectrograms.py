import os
import torch
import numpy as np
from HFAutoencoder import ImageAutoencoder  # Assuming the class is saved in image_autoencoder.py

def convert_images_to_latent_and_save(input_directory, output_directory, image_size=512):
    # Initialize the ImageAutoencoder
    autoencoder = ImageAutoencoder()

    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process each file in the input directory
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Construct full file paths
            file_path = os.path.join(input_directory, filename)
            output_file = os.path.join(output_directory, f"latent_{filename.split('.')[0]}.npy")

            # Encode the image and get latent representation
            latent_dist = autoencoder.encode(file_path, image_size)
  
            # Convert latent distribution to numpy array
            latent_array = latent_dist.sample().numpy() * 0.18215

            print(latent_array.shape)
            # Save the latent representation as a numpy file
            np.save(output_file, latent_array)

# Example usage
input_dir = 'Audio_Processing//midi_spectograms'
output_dir = 'Audio_Processing//midi_latents'
convert_images_to_latent_and_save(input_dir, output_dir)
