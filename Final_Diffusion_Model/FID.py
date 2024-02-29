import torch
from torchvision.models import inception_v3
import torchvision.transforms as transforms
from scipy.linalg import sqrtm
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def calculate_fid(model, real_images, fake_images):
    # Function to calculate activations
    def get_activations(images, model):
        model.eval()
        with torch.no_grad():
            activations = model(images)
        return activations.detach().cpu().numpy()

    # Load Inception model
    model = model.cuda()
    model.fc = torch.nn.Identity()  # Modify model to output activations before final layer
    
    # Calculate activations
    real_activations = get_activations(real_images, model)
    fake_activations = get_activations(fake_images, model)
    
    # Calculate mean and covariance
    mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = fake_activations.mean(axis=0), np.cov(fake_activations, rowvar=False)
    
    # Calculate Fréchet distance
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Example of loading datasets (adjust paths and transformations as needed)
real_dataset = ImageFolder(root='Audio_Processing//spectograms', transform=transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
]))
fake_dataset = ImageFolder(root='Sample_Outputs//Non_Midi//To_Audio', transform=transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
]))

# Create data loaders
real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)
fake_loader = DataLoader(fake_dataset, batch_size=32, shuffle=False)

# Initialize InceptionV3 model
inception_model = inception_v3(pretrained=True)

# Iterate over real and fake data loaders to calculate FID
# Note: This is a simplified example. In practice, you should accumulate all activations before calculating FID.
for real_images, _ in real_loader:
    for fake_images, _ in fake_loader:
        fid = calculate_fid(inception_model, real_images.cuda(), fake_images.cuda())
        print("FID score:", fid)
        break  # This break is just for demonstration; remove it in actual use
