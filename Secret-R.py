
#-- this programms splits a .wav audio into segments, stocks timestamps in a file, then speaker are identified and corresponding speach is transcripted
import torch
import soundfile as sf
from pyannote.audio import Pipeline
import tempfile

output_file = "diarization.txt"
input_file="cedric.wav"

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
        start_sample = int(start[i] * audio["sample_rate"])
        end_sample = int(end * audio["sample_rate"])
        sf.write(f"audio{i}.wav", chunk, audio["sample_rate"])

# --- embed samples function ---


# --- embed (from splited) and compare with samples ---


# --- transcript (from splited) ---



# --- Load audio file ---
audio = load_audio(input_file)

# --- Run diarization ---
print("\ndiarizing\n")
result = pipeline(audio)

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

with open(output_file, "w", encoding="utf-8") as f: #write in result file
    f.write("Speaker diarization result:\n\n")


    # Try standard method first
    if hasattr(diarization, "itertracks"):#standard expected flow (itertracks has both timestamps and speaker)
        for segment, _, speaker in diarization.itertracks(yield_label=True):

            print(f"{speaker}:")

"""
    # Fallback if itertracks does not exist #only timestamps, no speaker (could implement later identification if needed)
    elif hasattr(diarization, "segments"):
        for segment in diarization.segments:
            f.write("UNKNOWN":)"""
        
