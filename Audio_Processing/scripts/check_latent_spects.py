from HugginFaceMelSpect import AudioSpectrogramConverter
from HFAutoencoder import ImageAutoencoder
from normalise_latents import ArrayNormalizer
import numpy as np
import torch
autoencoder = ImageAutoencoder()
denormaliser = ArrayNormalizer("", -95.43647003173828, 89.94168853759766)
spect_num = 1276
np.set_printoptions(suppress=True,precision=3)
loaded = np.load(f"Audio_Processing//latents//latent_spectogram_{spect_num}.npy")
print(loaded)
loaded = denormaliser.denormalize_array(loaded)
print(loaded)
torchy = torch.from_numpy(loaded).float()

# print(torchy)
output_image = autoencoder.decode(torchy)

autoencoder.save_image(output_image, "random_test_spect.jpg")