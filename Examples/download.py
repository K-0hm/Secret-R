# download_models.py
from huggingface_hub import snapshot_download
import getpass

# Ask for Hugging Face token
hf_token = getpass.getpass("Enter your Hugging Face token (input hidden): ")

# Download pyannote/embedding
print("Downloading pyannote/embedding...")
embedding_path = snapshot_download("pyannote/embedding", token=hf_token)
print(f"pyannote/embedding downloaded to: {embedding_path}\n")

# Download pyannote/speaker-diarization-3.1
print("Downloading pyannote/speaker-diarization-3.1...")
diarization_path = snapshot_download("pyannote/speaker-diarization-3.1", token=hf_token)
print(f"pyannote/speaker-diarization-3.1 downloaded to: {diarization_path}\n")

print("All models downloaded and cached locally.")