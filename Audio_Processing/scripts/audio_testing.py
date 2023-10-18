import wave
import numpy as np

def load_wav(filename):
    with wave.open(filename, 'rb') as w:
        frames = w.readframes(w.getnframes())
        return np.frombuffer(frames, dtype=np.int16), w.getframerate()

# Load one of the split files into a numpy array
audio_data, sample_rate = load_wav("Audio_Processing\youtube_tunes\piano1_segment_0.wav")

audio_data = np.array(audio_data)

print(audio_data.shape)