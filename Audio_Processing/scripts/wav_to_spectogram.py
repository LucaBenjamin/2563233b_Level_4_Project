import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.io import wavfile


def calculate_frame_step(waveform_length, desired_width=255, frame_length=508):
    return int(round((waveform_length - frame_length) / (desired_width - 1)))



def get_spectrogram(waveform, frame_step):
    # Convert the waveform to a spectrogram via a STFT.
    stft_result = tf.signal.stft(
        waveform, frame_length=508, frame_step=frame_step)
    
    magnitude = tf.abs(stft_result)
    phase = tf.math.angle(stft_result)
    
    return magnitude, phase, stft_result

def spectrogram_to_waveform(stft_result, frame_step):
    waveform = tf.signal.inverse_stft(
        stft_result, frame_length=512, frame_step=frame_step,
        window_fn=tf.signal.inverse_stft_window_fn(frame_step)
    )
    return waveform

def plot_spectrogram(spectrogram, ax):
    if spectrogram.shape[-1] == 1:
        spectrogram = np.squeeze(spectrogram, axis=-1)

    log_spec = np.log(spectrogram.T + np.finfo(float).eps)

    print("Spectrogram shape:", spectrogram.shape)
    print("Log_spec shape:", log_spec.shape)

    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


def save_waveform_to_wav(waveform, filename, sample_rate=16000):
    wavfile.write(filename, sample_rate, waveform)

def stereo_to_mono(waveform):
    return np.mean(waveform, axis=1)

def main():
    # Example of reading a WAV file, converting it to a spectrogram, 
    # and then converting it back to a waveform and saving it
    sample_rate, waveform = wavfile.read("Audio_Processing\processed_youtube_clips_sample\Piano3.wav_segment_5.wav")
    waveform = waveform.astype(np.float32)
    waveform /= 32768.0  # As int16 ranges from -32768 to 32767

    print("Waveform shape:", waveform.shape)  # Add this line

    frame_step = calculate_frame_step(len(waveform))

    magnitude, phase, stft_result = get_spectrogram(waveform, frame_step)

    print("STFT result shape:", stft_result.shape)  # Add this line
    
    # Plot the magnitude spectrogram for visualization
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_spectrogram(magnitude.numpy(), ax)
    ax.set_title("Magnitude Spectrogram")
    plt.show()

    # Convert the spectrogram back to waveform
    reconstructed_waveform = spectrogram_to_waveform(stft_result, frame_step)

    print(reconstructed_waveform.shape[0])
    
    # Save the reconstructed waveform as a WAV file
    save_waveform_to_wav(reconstructed_waveform.numpy(), "reconstructed_file.wav", sample_rate)

if __name__ == "__main__":
    main()
