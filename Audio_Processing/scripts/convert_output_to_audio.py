import wave
import numpy as np
import os
from HugginFaceMelSpect import AudioSpectrogramConverter


def increase_volume(input_filename, output_filename, factor):
    # Open the input file
    with wave.open(input_filename, 'r') as wav_file:
        # Extract audio parameters
        params = wav_file.getparams()
        # Read the waveform data
        audio_data = wav_file.readframes(params.nframes)

    # Convert audio data to numpy array
    if params.sampwidth == 1:
        data = np.frombuffer(audio_data, dtype=np.uint8) - 128
    elif params.sampwidth == 2:
        data = np.frombuffer(audio_data, dtype=np.int16)
    elif params.sampwidth == 4:
        data = np.frombuffer(audio_data, dtype=np.int32)
    else:
        raise ValueError("Unsupported sample width")

    # Increase volume
    new_data = data * factor
    # Clip values to prevent wrapping
    new_data = np.clip(new_data, -2**(8*params.sampwidth-1), 2**(8*params.sampwidth-1)-1)

    # Convert numpy array back to bytes
    new_audio_data = new_data.astype(data.dtype).tobytes()

    # Write the modified data to a new file
    with wave.open(output_filename, 'w') as new_wav_file:
        new_wav_file.setparams(params)
        new_wav_file.writeframes(new_audio_data)


def convert_outs_to_audio(input_directory, output_directory):
    converter = AudioSpectrogramConverter(x_res=512, y_res=512)
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process each file in the input directory
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            converter.save_audio(converter.spectrogram_to_audio(converter.load_spectrogram_image(input_directory + "//" + filename)), output_directory + "//" + filename.split('.')[0] + "_audio.wav")

# Example usage
input_dir = 'Sample_Outputs//Midi//Samples//To_Audio'
output_dir = 'Sample_Outputs//Midi//Samples//Audio'
convert_outs_to_audio(input_dir, output_dir)
converter = AudioSpectrogramConverter(x_res=512, y_res=512)
converter.save_audio(converter.spectrogram_to_audio(converter.load_spectrogram_image("epoch_4536_denoised_4536.png")), "second_out.wav")
