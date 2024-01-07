from HugginFaceMelSpect import AudioSpectrogramConverter
from HFAutoencoder import ImageAutoencoder
import numpy as np
import torch

def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized = (data - min_val) / (max_val - min_val)
    return normalized, min_val, max_val

def denormalize(normalized_data, min_val, max_val):
    original_data = normalized_data * (max_val - min_val) + min_val
    return original_data

autoencoder = ImageAutoencoder()

spect_num = 10

loaded = np.load(f"Audio_Processing//latents//latent_spectogram_{spect_num}.npy")
loaded, min, max = normalize(loaded)
loaded = denormalize(loaded, min, max)
torchy = torch.from_numpy(loaded).float()

# print(torchy)
output_image = autoencoder.decode(torchy)

autoencoder.save_image(output_image, "random_test_spect.jpg")