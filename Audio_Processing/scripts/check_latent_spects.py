from HFAutoencoder import ImageAutoencoder
import numpy as np
autoencoder = ImageAutoencoder()
spect_num = 1013
np.set_printoptions(suppress=True,precision=3)
loaded = np.load(f"Audio_Processing//latents//latent_spectogram_{spect_num}.npy")
print(loaded)
print(loaded.min(), loaded.max())


# autoencoder.save_image(output_image, "random_test_spect.jpg")