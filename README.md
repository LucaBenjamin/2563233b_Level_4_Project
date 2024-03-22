# Level 4 Project: Music Generation with Latent Audio Diffusion

This repository hosts the code for my Level 4 project, which aimed to produce music by using latent audio diffusion techniques.

## Directory Description

### `Audio_Processing`
Contains Python scripts for manipulating the audio data in various ways.

- `abc_to_midi.py`: Converts .abc files to .midi files.
- `audio_testing.py`: Tests audio shape and sample rate.
- `HuggingFaceMelSpect.py`: A class that implements Hugging Face's mel-spectrogram converter.
- `make_spectrograms.py`: Uses `HuggingFaceMelSpect.py` to convert a directory of audio clips to spectrograms.
- `convert_spectrograms_to_audio.py`: Uses `HuggingFaceMelSpect.py` to convert a directory of spectrograms back to audio clips.
- `convert_to_pcm.py`: Converts WAV audio to PCM format, which is needed later in the pipeline.
- `download_youtube_playlist.py`: Downloads all audio from videos in a YouTube playlist.
- `process_youtube_audio.py`: Processes audio downloaded from YouTube, splits into clips, resamples, etc.
- `HFAutoencoder.py`: A class that implements Stable Diffusion's VAE from Hugging Face, allows spectrograms to be converted to latent space.
- `latentify_spectrograms.py`: Uses `HFAutoencoder.py` to convert a directory of mel-spectrograms to latent space.
- `delatentify_spectrograms.py`: Converts a directory back from latent space to pixel space (used for testing).

- `spotify_scraper`: Contains a script to scrape Spotify audio. Not used in the final implementation.

### `Final_Diffusion_Model`
- `dataset_class.py`: A PyTorch dataset class for loading latents.
- `final_diffusion_model.py`: The code for the actual diffusion model including training and evaluation.
- `generate_from_trained.py`: A script to generate spectrogram images from a pre-trained model.
- `[modified pipelines]`: Edits to some existing pipelines to make them work with latents.

### `GAN_Testing` and `Initial_Diffusion_Testing`
Some initial naive testing with both types of model, not important for the final implementation.

### `Small_Sample_Outputs`
A small random selection of sample outputs from both final baseline models.


## Pipeline guide
This section will explain how to use the scripts as a pipeline to process the data and train the model.

### Before You Start
1. Download CUDA and make sure it works with PyTorch
2. (Not Essential) create a new python environment to work in.
3. `pip install` any packages needed.
4. Some directories, e.g. where the spectrogram images are stored, have hard coded locations within the scripts (relative to project root folder). Make sure you name your dirs the same or change the scripts.

### Audio Processing
1. Download audio with `download_youtube_playlist.py`, or download midi files and convert to WAVs manually.
2. Process the audio with `process_youtube_audio.py`.
3. Convert the clips into Mel-Spectrograms using `make_spectrograms.py`.
4. Convert spectrograms into latent space using `latentify_spectrograms.py`

### Model Training
1. Make sure `dataset_class.py` points to your latent .npy files.
2. Run `final_diffusion_model.py`! Samples will be generated in a directory named  `test_out`.