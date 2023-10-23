import wave
import numpy as np
import os

def make_mono(filename):
    with wave.open(filename, 'rb') as source_wav:
        source_channels = source_wav.getnchannels()
        frames = np.frombuffer(source_wav.readframes(source_wav.getnframes()), dtype=np.int16)

        # Convert to mono if not already
        if source_channels > 1:
            frames = frames.reshape(-1, source_channels)
            frames = frames.mean(axis=1).astype(np.int16)

        return frames, source_wav.getframerate()


def resample(frames, original_rate, target_rate):
    if original_rate == target_rate:
        return frames

    # Calculate new length of data after resampling
    new_length = int(len(frames) * target_rate / original_rate)
    
    # Resample using linear interpolation
    resampled_frames = np.interp(
        np.linspace(0, len(frames), new_length, endpoint=False),
        np.arange(len(frames)),
        frames
    ).astype(np.int16)
    
    return resampled_frames


def resample(frames, original_rate, target_rate):
    if original_rate == target_rate:
        return frames

    # Calculate new length of data after resampling
    new_length = int(len(frames) * target_rate / original_rate)
    
    # Resample using linear interpolation
    resampled_frames = np.interp(
        np.linspace(0, len(frames), new_length, endpoint=False),
        np.arange(len(frames)),
        frames
    ).astype(np.int16)
    
    return resampled_frames


def split_audio(frames, framerate, segment_length=5):
    frames_per_segment = segment_length * framerate
    num_segments = int(len(frames) / frames_per_segment) + (1 if len(frames) % frames_per_segment else 0)
    segments = []

    for i in range(num_segments):
        start_frame = i * frames_per_segment
        end_frame = min((i+1) * frames_per_segment, len(frames))
        segments.append(frames[start_frame:end_frame])

    return segments

def save_segments(segments, framerate, prefix="segment"):
    for i, segment in enumerate(segments):
        with wave.open(f"{prefix}_{i}.wav", 'wb') as segment_wav:
            segment_wav.setnchannels(1)
            segment_wav.setsampwidth(2)  # Assuming 16-bit audio
            segment_wav.setframerate(framerate)
            segment_wav.writeframes(segment.tobytes())

def process_wav(filename, save_dir, target_rate = 16000):
    frames, original_rate = make_mono(filename)
    frames = resample(frames, original_rate, target_rate)
    segments = split_audio(frames, target_rate)
    save_segments(segments, target_rate, prefix = f"{save_dir}_segment")

youtube_files_dir = "Audio_Processing/youtube_tunes"

for filename in os.listdir(youtube_files_dir):
    file_path = os.path.join(youtube_files_dir, filename)
    if os.path.isfile(file_path):
        process_wav(file_path, "Audio_Processing/youtube_tunes/processed_clips/" + os.path.basename(file_path))

