#!/usr/bin/env python3

#-- this programms splits a .wav audio into segments, stocks timestamps in a file, then speaker are identified and corresponding speach is transcripted
import torch
import whisper
import soundfile as sf
from pyannote.audio import Pipeline, Model, Inference
import tempfile
#from pathlib import Path

output_file = "diarization.txt"
input_file="Recordings/cedric.wav"

# --- Load pretrained diarization pipeline ---
print("\nLoading diarization pipeline\n")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token="put_your_token_here"
)

# --- Set device ---#avoids using heavy Gpu oriented dependency (CUDA/Nvidia)
device = torch.device("cpu")
pipeline.to(device)

# --- load audio function (NO torchcodec) ---
def load_audio(path):
    waveform, sample_rate = sf.read(path)

    waveform = torch.tensor(waveform.T, dtype=torch.float32)

    # Ensure shape = (channels, time)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    return {"waveform": waveform, "sample_rate": sample_rate}

# --- split audio from timestamps ---
def split(start, end):        
    start_sample = int(start * audio["sample_rate"])
    end_sample = int(end * audio["sample_rate"])

    if end_sample <= start_sample:
        return None

    chunk = audio["waveform"][:, start_sample:end_sample]

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=True)
    sf.write(tmp.name, chunk.T, audio["sample_rate"])

    return tmp
# --- embed all samples function ---


# --- embed (from splited) and compare with samples ---


# --- transcript (from splited) ---
def transcript(diarization, output_file):


    with open(output_file, "a", encoding="utf-8") as f:

        for i, (segment, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):

            start = segment.start
            end = segment.end

            print(f"{speaker}: {start:.2f}s -> {end:.2f}s")

            # Split audio
            tmp_file = split(start, end)

            if tmp_file is None:
                continue

            # Transcribe
            result = model.transcribe(
                tmp_file.name,
                language="fr",
                task="transcribe"
                )
            text = result["text"].strip()

            # Write result
            line = f"{speaker} - {text}\n"
            f.write(line)

            print(line)


# --- Load audio file ---
audio = load_audio(input_file)

# --- Run diarization ---
print("\ndiarizing\n")
result = pipeline(audio)
print("\ndiarization finished\n")

print("\nLoading Whisper model\n")
model = whisper.load_model("small")
print("\nBegin Transcription\n")

# --- Extract diarization safely (version-compatible) --- #really needed?
if hasattr(result, "speaker_diarization"):
    diarization = result.speaker_diarization
"""
elif hasattr(result, "diarization"):
    diarization = result.diarization
else:
    diarization = result  # fallback for older versions"""

# ---  Process Results ---

print("\nProcessing segments:\n")

# ---  Process Results ---

print("\nProcessing segments:\n")

# Initialize file
with open(output_file, "w", encoding="utf-8") as f:
    f.write("Speaker diarization result:\n\n")

# Run transcription ONCE
if hasattr(diarization, "itertracks"):
    transcript(diarization, output_file)

"""
    # Fallback if itertracks does not exist #only timestamps, no speaker (could implement later identification if needed)
    elif hasattr(diarization, "segments"):
        for segment in diarization.segments:
            f.write("UNKNOWN":)"""
        
