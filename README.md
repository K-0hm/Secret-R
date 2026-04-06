# Secret-R
Pipeline that aims to take a meeting as an input and output a report writen in lateX.

## Audio Processing 
This project will first focus in making a transcription with identified speakers from a recorded conversation compared with sample voices.

The neural networks used for this transcription are:

- Open AI's [whisper transcription](https://github.com/openai/whisper?tab=MIT-1-ov-file) from 76MB to 1.6GB

- Pyannote's [speaker diarization pipeline](https://huggingface.co/pyannote/speaker-diarization-3.1) ~500Mo

- Pyannote's [voice embedding](https://huggingface.co/pyannote/embedding) ~100Mo
The main dependency of this part is the [pyannote's python toolbox library](https://github.com/pyannote/pyannote-audio?tab=MIT-1-ov-file).
