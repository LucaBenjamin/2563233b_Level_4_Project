from HugginFaceMelSpect import AudioSpectrogramConverter
converter = AudioSpectrogramConverter(x_res=512, y_res=512)
converter.save_audio(converter.spectrogram_to_audio(converter.load_spectrogram_image("epoch_3999_decoded_7.png")), "first_out.wav")
