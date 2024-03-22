
import os
from HugginFaceMelSpect import AudioSpectrogramConverter

# Used to make test audio for evaluation
def convert_outs_to_audio(input_directory, output_directory, x_res = 256, y_res = 256):
    converter = AudioSpectrogramConverter(x_res, y_res)
  
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            converter.save_audio(converter.spectrogram_to_audio(converter.load_spectrogram_image(input_directory + "//" + filename)), output_directory + "//" + filename.split('.')[0] + "_audio.wav")

input_dir = 'Audio_Processing//midi_spectrograms//images'
output_dir = 'Audio_Processing//eval_audio_512'
convert_outs_to_audio(input_dir, output_dir, 512, 512)
