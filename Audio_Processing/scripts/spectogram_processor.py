import os
import numpy as np
import librosa
import soundfile as sf

class AudioProcessor:

    @staticmethod
    def stereo_to_mono(waveform):
        return np.mean(waveform, axis=1)

    def spectrogram(self, audio_clip, output_path, n_fft=256, hop_length=156):
        waveform, sample_rate = librosa.load(audio_clip, sr=None)  # Load audio

        if waveform.ndim > 1:
            waveform = self.stereo_to_mono(waveform)

        # Compute the STFT
        stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)

        # Save only the magnitude of the STFT
        magnitude = np.abs(stft)
        np.save(output_path, magnitude)

        # Debug: Check the shape of the generated spectrogram
        print(f"Generated Spectrogram Shape: {magnitude.shape}")

        return magnitude

    def stft_to_wav(self, magnitude_npy_path, output_wav_path, sample_rate, n_fft=256, hop_length=156, n_iter=64):
        # Load the magnitude numpy file
        magnitude = np.load(magnitude_npy_path)

        # Use librosa's Griffin-Lim algorithm to estimate the phase and reconstruct the waveform
        waveform = librosa.griffinlim(magnitude, n_iter=n_iter, hop_length=hop_length, win_length=n_fft)

        # Save the reconstructed waveform as a wav file
        sf.write(output_wav_path, waveform, samplerate=sample_rate)

# Example usage
processor = AudioProcessor()
processor.spectrogram('sample.wav', 'test.npy')
processor.stft_to_wav('test.npy', 'reconstructed.wav', sample_rate=16000)  # Specify the sample rate here
