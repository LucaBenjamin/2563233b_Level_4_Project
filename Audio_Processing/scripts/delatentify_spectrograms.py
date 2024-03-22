import os
import torch
import numpy as np
from HFAutoencoder import ImageAutoencoder

def convert_images_to_latent_and_save(input_directory, output_directory):
    # initialize the ImageAutoencoder
    autoencoder = ImageAutoencoder()

    # output dir exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # proces input dir files
    for i, filename in enumerate(os.listdir(input_directory)):
        if filename.lower().endswith(('.npy')):
        
            file_path = os.path.join(input_directory, filename)
            output_file = os.path.join(output_directory, f"roundtripped_{i}.png")

            tensor = torch.tensor(np.load(file_path))  * (1 / 0.18215)
            # decode the to image tensor
            spect = autoencoder.decode(tensor)
            print(spect)
            # Save image :)
            autoencoder.save_image(spect, output_file)

input_dir = 'Evaluation//eval_latents_1024'
output_dir = 'Evaluation//eval_rt_1024'
convert_images_to_latent_and_save(input_dir, output_dir)
