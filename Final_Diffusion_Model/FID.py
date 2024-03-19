import torch
from torchvision.models import inception_v3
import torchvision.transforms as transforms
from scipy.linalg import sqrtm
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def calculate_fid(model, real_activations, fake_activations):
    mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = fake_activations.mean(axis=0), np.cov(fake_activations, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def get_activations(loader, model):
    model.eval()
    activations = []
    with torch.no_grad():
        l = len(loader)
        for images, _ in loader:
            if torch.cuda.is_available():
                images = images.cuda()
            act = model(images)
            activations.append(act.detach().cpu().numpy())
            print(f"{len(activations)} / {l}")
    activations = np.concatenate(activations, axis=0)
    return activations


# Transformations as required by Inception V3
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Setup for real images
real_dataset = ImageFolder(root='Audio_Processing//midi_spectograms', transform=transform)
real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)

# Setup for fake images
fake_dataset = ImageFolder(root='Sample_Outputs\Midi\Samples\To_Audio', transform=transform)
fake_loader = DataLoader(fake_dataset, batch_size=32, shuffle=False)

# Initialize and prepare model
inception_model = inception_v3(pretrained=True).cuda()
inception_model.fc = torch.nn.Identity()
inception_model.eval()

# Loaders remain the same as your setup
# Example of using the optimized get_activations function
real_activations = get_activations(real_loader, inception_model)
fake_activations = get_activations(fake_loader, inception_model)

# Now calculate FID once, using all accumulated activations
fid = calculate_fid(inception_model, real_activations, fake_activations)
print("FID score:", fid)
