from jiwer import wer, Compose, RemovePunctuation, ToLowerCase, RemoveMultipleSpaces, Strip

# Create a preprocessing pipeline
transform = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip()
])

with open("video_9_groundtruth.txt") as f:
    reference = f.read()

with open("rev_video9.txt") as f:
    hypothesis = f.read()

# Apply transform manually
reference_processed = transform(reference)
hypothesis_processed = transform(hypothesis)

# Now compute WER
error = wer(reference_processed, hypothesis_processed)
print(f"Word Error Rate (WER): {error:.3f}")
