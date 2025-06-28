import subprocess
import os
import torch
import torchaudio
from pyannote.audio import Pipeline
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq

# ==== CONFIGURATION ====
YOUTUBE_URL = "https://www.youtube.com/watch?v=ZYsX7_0xigo" \
""
HUGGINGFACE_TOKEN = # insert your token here
AUDIO_FILE = "downloaded_audio.wav"
OUTPUT_FILE = "transcript.txt"
HF_MODEL_NAME = "openai/whisper-large-v3"
# =======================

# Step 1: Download audio from YouTube
print("[INFO] Downloading audio...")
subprocess.run([
    "yt-dlp",
    "-x", "--audio-format", "wav",
    "-o", AUDIO_FILE,
    YOUTUBE_URL
], check=True)

# Step 2: Load PyAnnote diarization pipeline
print("[INFO] Loading speaker diarization model...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HUGGINGFACE_TOKEN
)

# Step 3: Run diarization
print("[INFO] Running speaker diarization...")
diarization_result = diarization_pipeline(AUDIO_FILE)

# Step 4: Load Hugging Face Whisper model
print("[INFO] Loading Whisper model...")
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=HF_MODEL_NAME,
    return_timestamps="word",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device=0 if torch.cuda.is_available() else -1
)

# Step 5: Transcribe
print("[INFO] Running transcription...")
asr_output = asr_pipe(AUDIO_FILE)

# Step 6: Align transcription with diarization
print("[INFO] Aligning words to speakers...")


def align_words_with_speakers(asr_output, diarization_result, margin=0.2):
    segments = []
    words = asr_output["chunks"]
    speaker_segments = list(diarization_result.itertracks(yield_label=True))

    for word in words:
        word_start, word_end = word["timestamp"]
        assigned_speaker = "unknown"

        for (segment, _, speaker) in speaker_segments:
            if segment.start - margin <= word_start <= segment.end + margin:
                assigned_speaker = speaker
                break

        segments.append({
            "start": word_start,
            "end": word_end,
            "speaker": assigned_speaker,
            "text": word["text"]
        })

    return segments


final_segments = align_words_with_speakers(asr_output, diarization_result)

# Optional: Merge by speaker and create blocks of speech
from itertools import groupby


def merge_segments(segments):
    merged = []
    for speaker, group in groupby(segments, key=lambda x: x["speaker"]):
        group = list(group)
        merged.append({
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "speaker": speaker,
            "text": " ".join([w["text"] for w in group])
        })
    return merged


merged_segments = merge_segments(final_segments)

# Step 7: Save to file
print(f"[INFO] Saving transcript to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w") as f:
    for seg in merged_segments:
        line = f'{seg["start"]:.2f}\t{seg["end"]:.2f}\t{seg["speaker"]}\t{seg["text"]}\n'
        f.write(line)

print("[INFO] Done!")
