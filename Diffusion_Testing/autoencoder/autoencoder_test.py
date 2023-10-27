import torch
from diffusers import AutoencoderKL
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
import tensorflow as tf
import numpy as np
import pylab as plt
from scipy.io import wavfile

#CODE FROM wav_to_spectogram.py
def calculate_frame_step(waveform_length, desired_width=255, frame_length=508):
    return int(round((waveform_length - frame_length) / (desired_width - 1)))

def get_spectrogram(waveform, frame_step):
    # Convert the waveform to a spectrogram via a STFT.
    stft_result = tf.signal.stft(
        waveform, frame_length=508, frame_step=frame_step)
    
    magnitude = tf.abs(stft_result)
    phase = tf.math.angle(stft_result)
    
    return magnitude, phase, stft_result

def plot_spectrogram(spectrogram, filename="Diffusion_Testing//autoencoder//spectrogram.png"):
    if spectrogram.shape[-1] == 1:
        spectrogram = np.squeeze(spectrogram, axis=-1)

    log_spec = np.log(spectrogram.T + np.finfo(float).eps)

    print("Spectrogram shape:", spectrogram.shape)
    print("Log_spec shape:", log_spec.shape)

    # Display the log-spectrogram
    plt.imshow(log_spec, aspect='auto', cmap='inferno')
    
    # Remove the axes, labels, and ticks
    plt.axis('off')
    # Save the plot without borders
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()  # Close the plot to free up memory



# Load the model
url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
model = AutoencoderKL.from_single_file(url)
model.eval()  # Set model to evaluation mode

# Load and preprocess an image
def load_image(path):
    img = Image.open(path)
    transform = Compose([
        Resize((32, 32)),  # Resize image to model's expected input size
        ToTensor()
    ])
    return( transform(img).unsqueeze(0)) # Add batch dimension

# Encode and decode
def encode_decode_image(img_path):
    img_tensor = load_image(img_path)
    print(f"Image tensor shape: {img_tensor.shape}")
    img_tensor = img_tensor[:, :3, :, :]
    with torch.no_grad():
        encoded_output = model.tiled_encode(img_tensor)
        latent = encoded_output.latent_dist.mean  # Use mean as a representative encoding
        decoded_output = model.tiled_decode(latent)
    return decoded_output.sample[0]  # Returns the decoded image tensor

# Test with an generated spectogram
sample_rate, waveform = wavfile.read("Audio_Processing\processed_youtube_clips_sample\Piano3.wav_segment_5.wav")
waveform = waveform.astype(np.float32)
waveform /= 32768.0  # As int16 ranges from -32768 to 32767

print("Waveform shape:", waveform.shape)  # Add this line

frame_step = calculate_frame_step(len(waveform))

magnitude, phase, stft_result = get_spectrogram(waveform, frame_step)

plot_spectrogram(magnitude.numpy())

img_path = 'Diffusion_Testing//autoencoder//spectrogram.png'
decoded_img_tensor = encode_decode_image(img_path)


decoded_img = Image.fromarray((decoded_img_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8'))
decoded_img.show()
decoded_img.save("Diffusion_Testing//autoencoder//encoded.png")