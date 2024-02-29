import os
from HugginFaceMelSpect import AudioSpectrogramConverter
# Example usage
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def convert_directory_to_spectrograms(input_directory, output_directory):
    converter = AudioSpectrogramConverter(x_res=512, y_res=512)
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, filename in enumerate(os.listdir(input_directory)):
        # Check if the file is a WAV file
      
        file_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, f"spectogram_{i}.png")

        converter.load_audio(file_path)
        spectrogram = converter.audio_to_spectrogram()
        converter.save_spectrogram(spectrogram, output_path)
# Example usage
input_dir = 'Audio_Processing//processed_midi_clips'
output_dir = 'Audio_Processing//midi_spectograms'
convert_directory_to_spectrograms(input_dir, output_dir)
