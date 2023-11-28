from spectogram_processor import AudioProcessor

audio_processor = AudioProcessor()
image_path = "test_spect.png"  
output_audio_path = "test_reconstructed.wav" 

audio_processor.inverse_spectrogram(image_path, output_audio_path)