from HugginFaceMelSpect import AudioSpectrogramConverter
from HFAutoencoder import ImageAutoencoder

sample_audio_path = "Audio_Processing/processed_clips/classical_2.wav_segment_35.wav"
spect_output_path = "HFSpect.png"

converter = AudioSpectrogramConverter(x_res=512, y_res=512)
converter.load_audio(sample_audio_path)
spectrogram = converter.audio_to_spectrogram()
converter.save_spectrogram(spectrogram, spect_output_path)


autoencoder = ImageAutoencoder()
latent_dist = autoencoder.encode('HFSpect.png', image_size=512)  
print( latent_dist.sample().shape )
output_image = autoencoder.decode(latent_dist)
autoencoder.save_image(output_image, "rederederedouncoded.jpg")

converter.save_audio(converter.spectrogram_to_audio(converter.load_spectrogram_image("rederederedouncoded.jpg")), "final_step.wav")
converter.save_audio(converter.spectrogram_to_audio(converter.load_spectrogram_image("HFSpect.png")), "final_step_unencoded.wav")