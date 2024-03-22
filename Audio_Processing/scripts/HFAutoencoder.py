import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from diffusers import AutoencoderKL
# Class to encode the spectrograms to latent space
class ImageAutoencoder:
    def __init__(self, model_url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"):
        # Initialize AutoencoderKL VAE
        self.model = AutoencoderKL.from_single_file(model_url)

    def load_image_as_tensor(self, file_path, image_size=512):
        # Load the image
        image = Image.open(file_path).convert('RGB')

        # Define the transformation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        # transform image if needed
        image_tensor = transform(image)

        # add necessary dimension
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor

    def encode(self, file_path, image_size = 256):
        # load image
        image_tensor = self.load_image_as_tensor(file_path, image_size=image_size)
        image_tensor = 2.0 * image_tensor - 1.0
        # encode to latent space
        with torch.no_grad():
            # not sure if encode() or tiled_encode() is better
            encoded = self.model.tiled_encode(image_tensor, return_dict=True)
        return encoded.latent_dist

    def decode(self, latent_dist):
        # decode from latent space to image
        with torch.no_grad():
            decoded = self.model.tiled_decode(latent_dist, return_dict=True)
        return (decoded.sample / 2.0 + 0.5).clamp(0, 1)  # Access the decoded images

    def save_image(self, tensor, file_path):
        # make sure values are between 0 and 1 fir PIL images
        tensor = tensor.clamp(0, 1)

        # make PIL image
        tensor = tensor.squeeze(0)  # Remove the batch dimension
        image = F.to_pil_image(tensor)

        # save it
        image.save(file_path)

# encoder = ImageAutoencoder()
# path = "VAE_Test_Before.png"
# out_path = "VAE_Test_After.png"
# encoded = encoder.encode(path)
# print(encoded.sample().numpy().shape)
# decoded = encoder.decode(encoded.sample())
# encoder.save_image(decoded, out_path)