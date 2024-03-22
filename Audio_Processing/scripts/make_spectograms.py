import os
from HugginFaceMelSpect import AudioSpectrogramConverter
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Uses spect class to generate all the spectrograms from a dir of audio files
def convert_directory_to_spectrograms(input_directory, output_directory, size = 512):
    converter = AudioSpectrogramConverter(x_res=size, y_res=size)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, filename in enumerate(os.listdir(input_directory)):
        file_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, f"spectogram_{i}.png")

        converter.load_audio(file_path)
        spectrogram = converter.audio_to_spectrogram()
        converter.save_spectrogram(spectrogram, output_path)
# Example usage
input_dir = 'Audio_Processing//processed_midi_clips'
output_dir = 'Evaluation//eval_spectrograms_1024'
convert_directory_to_spectrograms(input_dir, output_dir)
