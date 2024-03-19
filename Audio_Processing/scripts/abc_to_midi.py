import os
from music21 import converter

path = "C:/Users/Luca/Desktop/Dissertation/FolkGAN/Initial_GAN_Testing/Basic_Audio_Processing"
midi_path = f"{path}/tunes"

with open(f'{path}/musicABC.txt') as abc_music:
    for i, song in enumerate(abc_music.read().split('\n\n')):
        try:
            s = converter.parseData(song.strip(), format='abc')
            
            # create midi with music21
            midi_file = f'{midi_path}/tune_{i}.mid'
            s.write('midi', fp=midi_file)
            
            print(f"Saved {midi_file}")

        except Exception as e:
            print(f"Error processing tune {i}: {str(e)}")
            continue

