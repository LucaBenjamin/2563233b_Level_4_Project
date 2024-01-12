from HugginFaceMelSpect import AudioSpectrogramConverter
from HFAutoencoder import ImageAutoencoder
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

sample_audio_path = "Audio_Processing/youtube_tunes/processed_clips/classical_2.wav_segment_35.wav"
spect_output_path = "HFSpect.png"

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
autoencoder.save_image(output_image, "rederederedouncoded.jpg")

converter.save_audio(converter.spectrogram_to_audio(converter.load_spectrogram_image("rederederedouncoded.jpg")), "final_step.wav")
converter.save_audio(converter.spectrogram_to_audio(converter.load_spectrogram_image("HFSpect.png")), "final_step_unencoded.wav")