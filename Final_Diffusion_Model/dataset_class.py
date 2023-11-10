import os
import torch
from torch.utils.data import Dataset
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, audio_dir, transform=None):
        self.audio_dir = audio_dir
        self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
        self.transform = transform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        waveform, sample_rate = torchaudio.load(audio_path)

        return waveform, sample_rate

# Test usage
audio_dir = "Audio_Processing\youtube_tunes\processed_clips"
audio_dataset = AudioDataset(audio_dir=audio_dir)


for i, (waveform, sample_rate) in enumerate(audio_dataset):
    print(waveform.shape, i)
    pass
