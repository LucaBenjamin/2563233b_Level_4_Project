import os
import numpy as np
import librosa
import soundfile as sf

class AudioProcessor:

    @staticmethod
    def stereo_to_mono(waveform):
        return np.mean(waveform, axis=1)

    def spectrogram(self, audio_clip, output_path, n_fft=512, hop_length=312):
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

    def griffin_lim(self, magnitude, n_fft=512, hop_length=312, n_iter=32):
        # Initialize the phase randomly
        phase = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
        stft_complex = magnitude * phase

        for _ in range(n_iter):
            waveform = librosa.istft(stft_complex, hop_length=hop_length)
            stft_reconstructed = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
            stft_complex = magnitude * np.exp(1j * np.angle(stft_reconstructed))

        return librosa.istft(stft_complex, hop_length=hop_length)

    def stft_to_wav(self, magnitude_npy_path, output_wav_path, sample_rate, n_fft=512, hop_length=312, n_iter=32):
        # Load the magnitude numpy file
        magnitude = np.load(magnitude_npy_path)

        # Perform Griffin-Lim to estimate the phase and reconstruct the waveform
        waveform = self.griffin_lim(magnitude, n_fft=n_fft, hop_length=hop_length, n_iter=n_iter)

        # Save the reconstructed waveform as a wav file
        sf.write(output_wav_path, waveform, samplerate=sample_rate)

# Example usage
processor = AudioProcessor()
processor.spectrogram('sample.wav', 'test.npy')
processor.stft_to_wav('test.npy', 'reconstructed.wav', sample_rate=16000)  # Specify the sample rate here