from dataclasses import dataclass
import torchvision.transforms as transforms
from dataset_class import NumpyDataset
import pylab as plt
import torchvision.transforms.functional as TF
import torch
from diffusers import UNet2DModel
from PIL import Image
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
import os
from accelerate import Accelerator
from accelerate import notebook_launcher
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path 
import glob

from HFAutoencoder import ImageAutoencoder
from modified_DDPM_pipeline import DDPMPipeline
from modified_DDPM_scheduler import DDPMScheduler

@dataclass
class TrainingConfig:
    image_size = 64  # the generated image resolution
    train_batch_size = 32
    eval_batch_size = 8  # how many images to sample during evaluation
    num_epochs = 100000
    gradient_accumulation_steps = 1
    learning_rate = 2e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 1
    seed = 0
    mixed_precision = "fp16"
    output_dir = "Final_Diffusion_Model//test_out"
    push_to_hub = False 

config = TrainingConfig()


# directory containing the latent spectrograms
image_dir = "Audio_Processing//midi_latents"

# Define transformations, if needed
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    # Add any other transforms you might need
])

# Create an instance of Spectrogram Dataset
dataset = NumpyDataset(npy_dir=image_dir)


train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

# instantiate autoencoder
autoencoder = ImageAutoencoder()

renormalisation_factor = (1.0 / 0.18215) # MAGIC NUMBER ALERT


# UNET MODEL
model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=4,  # the number of input channels, 4 for latent space
    out_channels=4,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    # block_out_channels=(64, 64, 128, 128, 256, 256),
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
model.to(device)

sample_image = dataset[0].unsqueeze(0).to(device)  # Move to the same device as the model

print("Input shape:", sample_image.shape)

print("Output shape:", model(sample_image, timestep=0).sample.shape)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule = 'scaled_linear', clip_sample = False,  thresholding = False)

noise = torch.randn(sample_image.shape).to(device)
timesteps = torch.LongTensor([50]).to(device)
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps).to(device)



noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise)


optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)



def evaluate(config, epoch, pipeline):
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images
 
    for i, img in enumerate(images):
        # Convert PIL image to tensor
        # print(img.max(), img.min())
        # print(img.shape)
        img_tensor = img.unsqueeze(0).to(device)
        # print("\n\n", img_tensor.shape)
        # print("\n\n", img_tensor)
        # print(img_tensor.max(), img_tensor.min())
        denormalised =img_tensor * renormalisation_factor

        decoded_tensor = autoencoder.decode(denormalised)
        
        
        autoencoder.save_image(decoded_tensor, f"Final_Diffusion_Model//test_out//samples//epoch_{epoch:04d}_decoded_{i}.png")


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            clean_images = batch
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            
            clean_images = clean_images.to(device)
            noise = noise.to(device)


            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps).to(device)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # CODE TO VIEW SOME DENOISING STEPS
            # test_out = noise_scheduler.step(noise_pred.cpu()[0], timesteps.cpu()[0], noisy_images.cpu()[0]).pred_original_sample
            # test_out = (test_out * renormalisation_factor).unsqueeze(0).to(device)
            # noised = (noisy_images[0] * renormalisation_factor).to(device)
            # noised = noised.unsqueeze(0)
            # if global_step % 500 == 0 or global_step % 501 == 0 or global_step % 502 == 0 or global_step % 503 == 0 or global_step % 504 == 0:
            #     noised = autoencoder.decode(noised)
            #     decoded_tensor = autoencoder.decode(test_out)
            #     autoencoder.save_image(decoded_tensor, f"Final_Diffusion_Model//test_out//samples//denoised//epoch_{epoch:04d}_denoised_{global_step}.png")
            #     autoencoder.save_image(noised, f"Final_Diffusion_Model//test_out//samples//denoised//epoch_{epoch:04d}_noisy_{global_step}.png")

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    print("Got here!")
                    pipeline.save_pretrained(config.output_dir)



train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
Image.open(sample_images[-1])