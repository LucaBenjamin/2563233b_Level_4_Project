# load_model.py

import torch
from diffusers import UNet2DModel
from modified_pipeline import TestPipeline
from modified_scheduler import DDPMScheduler
import json
import safetensors.torch
from torchvision.transforms.functional import to_pil_image
from HFAutoencoder import ImageAutoencoder

class CustomModelLoader:
    def __init__(self, config_path, model_weights_path, scheduler_config_path):
        self.config_path = config_path
        self.model_weights_path = model_weights_path
        self.scheduler_config_path = scheduler_config_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scheduler = None
        self.load_config()
        self.load_model()
        self.load_scheduler()
        self.renormalisation_factor = (1.0 / 0.18215)

    def load_config(self):
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

    def load_model(self):
        # Assuming UNet2DModel architecture as per provided code
        self.model = UNet2DModel(
            sample_size=self.config['image_size'],
            in_channels=4,  # Adjust if necessary
            out_channels=4,
            layers_per_block=2,
            # block_out_channels=(128, 128, 256, 256, 512, 512),
            block_out_channels=(64, 64, 128, 128, 256, 256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        ).to(self.device)

        # Load the model weights from .safetensors file
        weights = safetensors.torch.load_file(self.model_weights_path)
        self.model.load_state_dict(weights)

    def load_scheduler(self):
        # Load scheduler configuration
        with open(self.scheduler_config_path, 'r') as f:
            scheduler_config = json.load(f)
        
        self.scheduler = DDPMScheduler(**scheduler_config)

    def generate_image(self, batch_size=1, seed=0):
        pipeline = TestPipeline(unet=self.model, scheduler=self.scheduler).to(self.device)
        
        # Generate images
        images = pipeline(batch_size=batch_size, generator=torch.manual_seed(seed)).images
        
        return images

if __name__ == "__main__":
    autoencoder = ImageAutoencoder()

    # Paths to model configuration, weights, and scheduler configuration
    config_path = 'Saved_Models///half_width_midi//config.json' 
    model_weights_path = 'Saved_Models//half_width_midi//diffusion_pytorch_model.safetensors'
    scheduler_config_path = 'Final_Diffusion_Model//test_out//scheduler//scheduler_config.json' 

    loader = CustomModelLoader(config_path, model_weights_path, scheduler_config_path)
    
    # Generate a new image
    images = loader.generate_image(batch_size=1, seed=42)
    
    # Save or display the generated images as needed
    for i, img_tensor in enumerate(images):
        img = img_tensor.unsqueeze(0)
        denormalised = img * loader.renormalisation_factor
        decoded_tensor = autoencoder.decode(denormalised)
        autoencoder.save_image(decoded_tensor, "generated_image_0.png")
