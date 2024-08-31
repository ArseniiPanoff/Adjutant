import os

import pandas as pd
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, delimiter='|')
        if 'audio_path' not in self.df.columns or 'transcription' not in self.df.columns:
            raise ValueError("CSV file must contain 'audio_path' and 'transcription' columns")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.iloc[idx]['audio_path']
        transcription = self.df.iloc[idx]['transcription']

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sample_rate = torchaudio.load(audio_path)
        mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=80,  # Should match your model’s expected mel bins
            n_fft=2048  # Example FFT size; adjust if necessary
        )
        mel_spectrogram = mel_transform(waveform)
        mel_spectrogram_db = T.AmplitudeToDB()(mel_spectrogram)

        mel_spectrogram_db = mel_spectrogram_db[:, :416]  # Ensure length matches the target length

        return mel_spectrogram_db.squeeze().numpy(), transcription
