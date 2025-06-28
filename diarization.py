import whisper
import subprocess
import os
from pyannote.audio import Pipeline
from utils import words_per_segment

# ==== CONFIGURATION ====
YOUTUBE_URL = "https://www.youtube.com/watch?v=KJUUwBXhfXA"  # Replace with actual URL
HUGGINGFACE_TOKEN = # insert token
AUDIO_FILE = "downloaded_audio.wav"
OUTPUT_FILE = "video1_just_transcript.txt"
MODEL_SIZE = "turbo"  # Options: tiny, base, small, medium, large
# =======================

# Step 1: Download and convert YouTube video to WAV using yt-dlp
print("[INFO] Downloading audio...")
subprocess.run([
    "yt-dlp",
    "-x", "--audio-format", "wav",
    "-o", AUDIO_FILE,
    YOUTUBE_URL
], check=True)

# Step 2: Load Whisper and pyannote models
print("[INFO] Loading models...")
model = whisper.load_model(MODEL_SIZE)
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HUGGINGFACE_TOKEN
)

# Step 3: Run diarization
print("[INFO] Running speaker diarization...")
diarization_result = pipeline(AUDIO_FILE)

# Step 4: Run transcription
print("[INFO] Running Whisper transcription...")
transcription_result = model.transcribe(AUDIO_FILE, word_timestamps=True)

# Step 5: Match words to speaker segments
print("[INFO] Aligning words to speakers...")
final_result = words_per_segment(transcription_result, diarization_result)

# Step 6: Save results to text file
print(f"[INFO] Saving transcript to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w") as f:
    for _, segment in final_result.items():
        line = f'{segment["start"]:.3f}\t{segment["end"]:.3f}\t{segment["speaker"]}\t{segment["text"]}\n'
        f.write(line)

print("[DONE] Transcription and diarization complete.")