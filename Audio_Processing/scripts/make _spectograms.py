import os
from spectogram_processor import AudioProcessor



def convert_directory_to_spectrograms(input_directory, output_directory):
    processor = AudioProcessor()
    
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, filename in enumerate(os.listdir(input_directory)):
        # Check if the file is a WAV file
        file_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, f"spectogram_{i}.png")
        processor.spectrogram(file_path, output_path)
# Example usage
input_dir = 'Audio_Processing//youtube_tunes//processed_clips'
output_dir = 'Audio_Processing//spectograms'
convert_directory_to_spectrograms(input_dir, output_dir)
