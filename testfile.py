import re

from mel.dataset import AudioDataset


def get_unique_tokens_normalized(file_path):
    with open(file_path, 'r') as f:
        transcriptions = [line.strip().split('|')[1].lower() for line in f]

    # Normalize by removing punctuation and numbers
    transcriptions = [re.sub(r'[^\w\s]', '', transcription) for transcription in transcriptions]

    tokens = [word for transcription in transcriptions for word in transcription.split()]
    unique_tokens = set(tokens)
    return unique_tokens

# Usage
unique_tokens = get_unique_tokens_normalized('filelists/combined_filelist.txt')
vocab_size = len(unique_tokens)
print(f"Number of unique tokens after normalization: {vocab_size}")

def get_max_mel_length(dataset):
    max_length = 0

    for mel_spectrogram, _ in dataset:
        if mel_spectrogram.shape[1] > max_length:
            max_length = mel_spectrogram.shape[1]

    return max_length

# Usage
dataset1 = AudioDataset('filelists/combined_filelist.txt')
max_mel_length = get_max_mel_length(dataset1)
print(f"Maximum mel spectrogram length: {max_mel_length}")