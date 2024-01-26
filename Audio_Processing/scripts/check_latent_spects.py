from HFAutoencoder import ImageAutoencoder
import numpy as np
from normalise_latents import ArrayNormalizer
from HFAutoencoder import ImageAutoencoder
import torch

autoencoder = ImageAutoencoder()
spect_num = 0
np.set_printoptions(suppress=True,precision=3)
loaded = np.load(f"Audio_Processing//latents//latent_spectogram_{spect_num}.npy")
print(loaded)
print(loaded.min(), loaded.max())
# instantiate denormaliser
denormaliser = ArrayNormalizer("")
loaded = denormaliser.denormalize_array(loaded)
decoded = autoencoder.decode(torch.from_numpy(loaded))
autoencoder.save_image(decoded, "testing_decoded.jpg")
# autoencoder.save_image(output_image, "random_test_spect.jpg")