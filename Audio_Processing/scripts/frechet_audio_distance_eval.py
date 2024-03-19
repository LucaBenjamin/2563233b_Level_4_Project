from frechet_audio_distance import FrechetAudioDistance

# Initialize the Frechet Audio Distance calculator with your chosen model
# Example with vggish
frechet = FrechetAudioDistance(
    model_name="vggish",
    sample_rate=16000,
    use_pca=False, 
    use_activation=False,
    verbose=True
)

# Paths to the directories containing the audio files for comparison
background_set_path = "Audio_Processing//processed_midi_clips"
eval_set_path = "Audio_Processing//eval_audio_512"

# Compute the Frechet Audio Distance
fad_score = frechet.score(background_set_path, eval_set_path, dtype="float32")

print(f"FAD Score: {fad_score}")
