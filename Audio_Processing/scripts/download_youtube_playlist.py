from pytube import Playlist
from moviepy.editor import *
import os

def download_videos_and_convert_to_wav(playlist_url, output_path):
    # create playlist object
    p = Playlist(playlist_url)
    # Initialize a counter for naming
    track_number = 1
    # Loop through all videos in the playlist
    for video in p.videos:
        try:
            # get the highest resolution stream available
            video_stream = video.streams.get_highest_resolution()
            # download the video stream
            video_file_path = video_stream.download(output_path=output_path)
            # load the file
            video_clip = VideoFileClip(video_file_path)
            # extract audio from file
            audio_clip = video_clip.audio
            # format filename
            audio_file_path = os.path.join(output_path, f"track_{track_number}.wav")
            #save the audio
            audio_clip.write_audiofile(audio_file_path)
            # increment counter
            track_number += 1
            
            print(f"Downloaded and converted video to WAV!: {audio_file_path}")
        except Exception as e:
            # error
            print(f"An error occurred with track {track_number}!!!: {e}")
        finally:
            # close the clips
            if 'video_clip' in locals():
                video_clip.close()
            if 'audio_clip' in locals():
                audio_clip.close()
            # delete video
            if os.path.exists(video_file_path):
                os.remove(video_file_path)

playlist_url = 'https://www.youtube.com/playlist?list=PLW9Nl3YHfwUpgh1GCoh2qPrzwzGDLQ6UF'
download_videos_and_convert_to_wav(playlist_url, output_path='Audio_Processing/yt_playlist_downloaded/')
