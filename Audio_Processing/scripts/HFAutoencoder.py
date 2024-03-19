import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from diffusers import AutoencoderKL

class ImageAutoencoder:
    def __init__(self, model_url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"):
        # Initialize the AutoencoderKL model
        self.model = AutoencoderKL.from_single_file(model_url)

    def load_image_as_tensor(self, file_path, image_size=256):
        # Load the image
        image = Image.open(file_path).convert('RGB')

        # Define the transformation
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        # Apply the transformation
        image_tensor = transform(image)

        # Add a batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor

    def encode(self, file_path, image_size = 256):
        # Load and transform the image
        image_tensor = self.load_image_as_tensor(file_path, image_size=image_size)
        image_tensor = 2.0 * image_tensor - 1.0
        # Encode the image to latent space
        with torch.no_grad():
            encoded = self.model.tiled_encode(image_tensor, return_dict=True)
        return encoded.latent_dist

    def decode(self, latent_dist):
        # Decode from latent space to pixel space
        with torch.no_grad():
            decoded = self.model.tiled_decode(latent_dist, return_dict=True)
        return (decoded.sample / 2.0 + 0.5).clamp(0, 1)  # Access the decoded images

    def save_image(self, tensor, file_path):
        # Clip the tensor values to ensure they are in the [0, 1] range
        tensor = tensor.clamp(0, 1)

        # Convert the tensor to PIL Image
        tensor = tensor.squeeze(0)  # Remove the batch dimension
        image = F.to_pil_image(tensor)

        # Save the image
        image.save(file_path)

encoder = ImageAutoencoder()
path = "VAE_Test_Before.png"
out_path = "VAE_Test_After.png"
encoded = encoder.encode(path)
print(encoded.sample().numpy().shape)
decoded = encoder.decode(encoded.sample())
encoder.save_image(decoded, out_path)