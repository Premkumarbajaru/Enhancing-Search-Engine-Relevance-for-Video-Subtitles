import os
import torch
import soundfile as sf
from datetime import datetime
from transformers import pipeline

class AudioProcessor:
    """Handles audio saving and transcription using Whisper ASR."""

    def __init__(self, save_dir="recordings"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=0 if torch.cuda.is_available() else -1,
            generate_kwargs={"language": "en"}
        )

    def save_audio(self, audio_file):
        """Saves the recorded or uploaded audio file and returns the file path."""

        if hasattr(audio_file, "read"):
            audio_bytes = audio_file.read()
        else:
            raise ValueError("Invalid audio input. Expected a file-like object.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        file_path = os.path.join(self.save_dir, filename)

        with open(file_path, "wb") as f:
            f.write(audio_bytes)

        return file_path

    def transcribe_audio(self, wav_path):
        """Converts audio file to text using Whisper ASR."""
        waveform, sample_rate = sf.read(wav_path)

        waveform = torch.tensor(waveform, dtype=torch.float32).to(self.device)

        result = self.asr_pipeline({
            "array": waveform.squeeze().cpu().numpy(),
            "sampling_rate": sample_rate
        })

        return result['text']
