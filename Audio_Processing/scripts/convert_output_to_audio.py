from HugginFaceMelSpect import AudioSpectrogramConverter
converter = AudioSpectrogramConverter(x_res=512, y_res=512)
converter.save_audio(converter.spectrogram_to_audio(converter.load_spectrogram_image("epoch_4536_denoised_4536.png")), "second_out.wav")
