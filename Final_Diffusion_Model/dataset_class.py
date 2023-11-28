from PIL import Image
from torch.utils.data import Dataset
import os

class SpectrogramDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        # Open the image in RGB mode
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image