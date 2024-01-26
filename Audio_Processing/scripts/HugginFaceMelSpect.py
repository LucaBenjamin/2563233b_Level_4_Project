from diffusers import Mel
from PIL import Image
import soundfile as sf
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class AudioSpectrogramConverter:
    def __init__(self, x_res, y_res, sample_rate=16000, n_fft=2048, slice_length=5):
        # Initialize Mel with given settings
        self.mel = Mel(x_res=x_res, y_res=y_res, sample_rate=sample_rate, n_fft=n_fft)
        self.x_res = x_res
        self.y_res = y_res
        self.slice_length = slice_length
        self.sample_rate = sample_rate
        self.calculate_hop_length(x_res)

    def calculate_hop_length(self, x_res):
        # Calculate hop_length to fit the audio exactly into the given x_res
        total_samples = self.slice_length * self.sample_rate
        self.hop_length = int(np.ceil(total_samples / x_res))
        self.mel.hop_length = self.hop_length  # Update hop_length in Mel object

    def load_audio(self, audio_file):
        # Load the audio
        self.mel.load_audio(audio_file=audio_file)

    def audio_to_spectrogram(self, slice_number=0):
        # Convert audio to spectrogram
        spectrogram_image = self.mel.audio_slice_to_image(slice=slice_number)
        # Resize the spectrogram image to match the specified resolution
        spectrogram_image = spectrogram_image.resize((self.x_res, self.y_res))
        return spectrogram_image

    def save_spectrogram(self, spectrogram_image, file_name):
        # Save the spectrogram image
        spectrogram_image.save(file_name)

    def load_spectrogram_image(self, file_name):
        # Load the spectrogram image from a file
        image = Image.open(file_name)

        # Ensure the image is in the correct mode (e.g., 'L' for grayscale)
        if image.mode != 'L':
            image = image.convert('L')

        # Resize the image to the expected dimensions if necessary
        if image.size != (self.x_res, self.y_res):
            image = image.resize((self.x_res, self.y_res))

        return image

    def spectrogram_to_audio(self, spectrogram_image):
        # Convert the spectrogram back to audio
        return self.mel.image_to_audio(spectrogram_image)

    def save_audio(self, audio_data, file_name):
        # Save the reconstructed audio
        sf.write(file_name, audio_data, self.sample_rate)
