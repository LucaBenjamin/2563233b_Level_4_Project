import subprocess
import os

def convert_directory_ieee_to_pcm(source_dir, target_dir):
    # Ensure target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.lower().endswith('.wav'):
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)

            # Construct and execute FFmpeg command
            command = [
                'ffmpeg',
                '-i', source_file,
                '-acodec', 'pcm_s16le',  # Convert to 16-bit PCM
                '-ar', '44100',          # Resample to 44100 Hz, if desired
                target_file
            ]
            subprocess.run(command, check=True)

            print(f"Converted '{source_file}' to '{target_file}'.")

# Example usage
source_directory = 'Audio_Processing//wav_tunes_from_midi'
target_directory = 'Audio_Processing//PCM_midi_clips'
convert_directory_ieee_to_pcm(source_directory, target_directory)
