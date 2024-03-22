from diffusers import Mel
from PIL import Image
import soundfile as sf
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Class to make the mel-spectrogram images
class AudioSpectrogramConverter:
    def __init__(self, x_res, y_res, sample_rate=16000, n_fft=2048, slice_length=5):
        self.mel = Mel(x_res=x_res, y_res=y_res, sample_rate=sample_rate, n_fft=n_fft)
        self.x_res = x_res
        self.y_res = y_res
        self.slice_length = slice_length
        self.sample_rate = sample_rate
        self.calculate_hop_length(x_res)

    # Figures out hop length to get square dimensions
    def calculate_hop_length(self, x_res):
        total_samples = self.slice_length * self.sample_rate
        self.hop_length = int(np.ceil(total_samples / x_res))
        self.mel.hop_length = self.hop_length 

    def load_audio(self, audio_file):
        self.mel.load_audio(audio_file=audio_file)

    def audio_to_spectrogram(self, slice_number=0):
        # conversion done here
        spectrogram_image = self.mel.audio_slice_to_image(slice=slice_number)
        # resize incase exact dims are wrong
        spectrogram_image = spectrogram_image.resize((self.x_res, self.y_res))
        return spectrogram_image

    def save_spectrogram(self, spectrogram_image, file_name):
        spectrogram_image.save(file_name)

    def load_spectrogram_image(self, file_name):
        image = Image.open(file_name)

        # put image in greyscale
        if image.mode != 'L':
            image = image.convert('L')

        # resize if needed
        if image.size != (self.x_res, self.y_res):
            image = image.resize((self.x_res, self.sy_res))

        return image

    def spectrogram_to_audio(self, spectrogram_image):
        return self.mel.image_to_audio(spectrogram_image)

    def save_audio(self, audio_data, file_name):
        sf.write(file_name, audio_data, self.sample_rate)
