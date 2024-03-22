from frechet_audio_distance import FrechetAudioDistance

# using google VGGish (standard)
frechet = FrechetAudioDistance(
    model_name="vggish",
    sample_rate=16000,
    use_pca=False, 
    use_activation=False,
    verbose=True
)

# paths to the dirs containing the audio files for comparison, backround is the clean validation set
# eval_set is what you want to comapre it to
background_set_path = "Evaluation//background_audio"
eval_set_path = "Evaluation//eval_100_ts_audio"

# compute the FAD across the two dists
fad_score = frechet.score(background_set_path, eval_set_path, dtype="float32")

print(f"FAD Score: {fad_score}")
