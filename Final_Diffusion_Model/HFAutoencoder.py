import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from diffusers import AutoencoderKL
# COPY OF THE VERSION IN AUDIO_PROCESSING
# YES I KNOW IT'S HORRIBLE THAT I HAVE 2 OF THE SAME FILE
# IF I EVER GET AROUND TO MAKING THE PIPELINE A FULL PACKAGE 
# MY SINS CAN BE FORGIVEN (?)
class ImageAutoencoder:
    def __init__(self, model_url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"):

        self.model = AutoencoderKL.from_single_file(model_url)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device) 

    def load_image_as_tensor(self, file_path, image_size=256):

        image = Image.open(file_path).convert('RGB')


        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])


        image_tensor = transform(image)

        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor

    def encode(self, file_path, image_size = 512):

        image_tensor = self.load_image_as_tensor(file_path, image_size=image_size)
        image_tensor = 2.0 * image_tensor - 1.0

        with torch.no_grad():
            encoded = self.model.tiled_encode(image_tensor, return_dict=True)
        return encoded.latent_dist

    def decode(self, latent_dist):

        with torch.no_grad():
            decoded = self.model.tiled_decode(latent_dist, return_dict=True)
        return (decoded.sample / 2.0 + 0.5).clamp(0, 1) 

    def save_image(self, tensor, file_path):
        tensor = tensor.clamp(0, 1)

        tensor = tensor.squeeze(0)  
        image = F.to_pil_image(tensor)

        # Save the image
        image.save(file_path)
