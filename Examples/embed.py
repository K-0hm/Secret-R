#!/usr/bin/env python3
#"/home/Robotech/Réunion/Réunion_1/jerem.wav"

import torch
import soundfile as sf
from pyannote.audio import Model, Inference
from scipy.spatial.distance import cdist


input_file1="../Voice_Sample/noe.wav"
input_file2="../Voice_Sample/mathurin.wav"

# --- 1) Load pretrained embedding model ---
model = Model.from_pretrained(
    "pyannote/embedding",
    use_auth_token="put_your_token_here"
)

# --- 2) Create Inference wrapper ---
device = torch.device("cpu")
inference = Inference(model, window="whole", device=device)

# --- 3) Function to load audio into waveform dict ---
def load_audio(path):
    waveform, sample_rate = sf.read(path)
    waveform = torch.tensor(waveform.T, dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}

# --- 4) Load audio files ---
audio1 = load_audio(input_file1)
audio2 = load_audio(input_file2)

# --- 5) Run inference (returns NumPy array directly) ---
emb1_np = inference(audio1)
emb2_np = inference(audio2)

# --- 6) Ensure 2D for cdist ---
if emb1_np.ndim == 1:
    emb1_np = emb1_np.reshape(1, -1)
if emb2_np.ndim == 1:
    emb2_np = emb2_np.reshape(1, -1)

# --- 7) Compare embeddings ---
distance = cdist(emb1_np, emb2_np, metric="cosine")[0, 0]
print("Cosine distance between embeddings:", distance)
print("Embedding shape:", emb1_np.shape)