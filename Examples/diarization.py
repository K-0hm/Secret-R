
import torch
import soundfile as sf
from pyannote.audio import Pipeline

input_file="../Recordings/cedric.wav"

# --- 1) Load pretrained diarization pipeline ---
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token="put_your_token_here"
)

# --- 2) Set device ---
device = torch.device("cpu")
pipeline.to(device)

# --- 3) Function to load audio (NO torchcodec) ---
def load_audio(path):
    waveform, sample_rate = sf.read(path)

    waveform = torch.tensor(waveform.T, dtype=torch.float32)

    # Ensure shape = (channels, time)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    return {"waveform": waveform, "sample_rate": sample_rate}

# --- 4) Load audio file ---
audio = load_audio(input_file)

# --- 5) Run diarization ---
result = pipeline(audio)

# --- 6) Extract diarization safely (version-compatible) ---
if hasattr(result, "speaker_diarization"):
    diarization = result.speaker_diarization
"""
elif hasattr(result, "diarization"):
    diarization = result.diarization
else:
    diarization = result  # fallback for older versions
"""
# --- 7) Print results (robust iteration) ---
print("\nSpeaker diarization result:\n")

# Try standard method first
if hasattr(diarization, "itertracks"):
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        print(f"{speaker}: {segment.start:.2f}s → {segment.end:.2f}s")
"""
# Fallback if itertracks does not exist
elif hasattr(diarization, "segments"):
    for segment in diarization.segments:
        print(segment)

# Final fallback (generic iteration)
else:
    try:
        for item in diarization:
            print(item)
    except Exception as e:
        print("Could not iterate diarization output:", e)
        print("Type:", type(diarization))
        print("Attributes:", dir(diarization))
"""