from HugginFaceMelSpect import AudioSpectrogramConverter
from HFAutoencoder import ImageAutoencoder
import os

# This tests out the full audio processing pipeline to check for issues

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

sample_audio_path = "Audio_Processing\youtube_tunes\processed_clips\classical_1.wav_segment_1.wav"
spect_output_path = "Spect_Pipeline_Test.png"

converter = AudioSpectrogramConverter(x_res=512, y_res=512)
converter.load_audio(sample_audio_path)
spectrogram = converter.audio_to_spectrogram()
converter.save_spectrogram(spectrogram, spect_output_path)


autoencoder = ImageAutoencoder()
latent_dist = autoencoder.encode('HFSpect.png', image_size=512)

print( latent_dist.sample().shape )
sampled = (latent_dist.sample() * 0.18215)
sampled = sampled * (1.0 / 0.18215)
output_image = autoencoder.decode(sampled)
autoencoder.save_image(output_image, "Latent_Roundtrip.jpg")

converter.save_audio(converter.spectrogram_to_audio(converter.load_spectrogram_image("Latent_Roundtrip.jpg")), "final_step.wav")
converter.save_audio(converter.spectrogram_to_audio(converter.load_spectrogram_image(spect_output_path)), "final_step_unencoded.wav")