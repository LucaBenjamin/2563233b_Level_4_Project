import os
import torch
import numpy as np
from HFAutoencoder import ImageAutoencoder

# Used the autoencoder class to make all spectrogram images into latents
def convert_images_to_latent_and_save(input_directory, output_directory, image_size=128):
    autoencoder = ImageAutoencoder()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # process each file in dir
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_directory, filename)
            output_file = os.path.join(output_directory, f"latent_{filename.split('.')[0]}.npy")

            # get latent dist
            latent_dist = autoencoder.encode(file_path, image_size)
  
            # IMPORTAT: here is the magic number, idk if it's optimal for spectrograms but
            # EVERYTHING BREAKS WITHOUT IT
            latent_array = latent_dist.sample().numpy() * 0.18215 

            print(latent_array.shape)
            # save as .npy file
            np.save(output_file, latent_array)

input_dir = 'Evaluation//eval_spectrograms_128'
output_dir = 'Evaluation//eval_latents_128'
convert_images_to_latent_and_save(input_dir, output_dir)
