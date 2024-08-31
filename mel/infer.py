import librosa
import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from mel.attention import Attention
from model import Encoder, Decoder, TTSModel
import numpy as np
import soundfile as sf  # Use soundfile for saving audio

# Configurations
CSV_FILE = '/home/lwolf/PycharmProjects/Adjutant/filelists/combined_filelist.txt'
MODEL_PATH = '/home/lwolf/PycharmProjects/Adjutant/mel/models/tts_model.pth'
VOCAB_SIZE = 16681  # Adjust as needed
EMB_DIM = 256
HIDDEN_DIM = 256
OUTPUT_DIM = 80  # Number of mel bins (you used 80 in the training)
MAX_LEN = 416  # The length of the mel spectrograms

# Dummy vocabulary for illustration; replace with your actual vocabulary
vocab = {word: idx for idx, word in enumerate("your_vocab_list".split())}

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def text_to_tensor(transcriptions, vocab, max_length=416):
    sequences = [[vocab.get(word, 0) for word in transcription.split()] for transcription in transcriptions]
    sequences_padded = [seq + [0] * (max_length - len(seq)) if len(seq) < max_length else seq[:max_length] for seq in sequences]
    text_tensor = torch.tensor(sequences_padded, dtype=torch.long)
    return text_tensor

def mel_to_waveform(mel_spectrogram, sr=22050, n_fft=2048, hop_length=512, n_mels=80):
    """
    Convert a mel spectrogram back to a waveform using the Griffin-Lim algorithm.
    """
    # Ensure mel_spectrogram is non-negative
    mel_spectrogram = np.clip(mel_spectrogram, 0, None)

    # Create mel basis matrix
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

    # Compute power spectrogram
    power_spectrogram = np.dot(mel_basis.T, mel_spectrogram)

    # Recover waveform using Griffin-Lim algorithm
    waveform = librosa.griffinlim(power_spectrogram, n_fft=n_fft, hop_length=hop_length)

    return waveform

def save_waveform(waveform, filename, sr=22050):
    """
    Save the waveform to a file using soundfile.
    """
    # Normalize waveform to the range [-1, 1]
    waveform = waveform / np.max(np.abs(waveform))
    sf.write(filename, waveform, sr)

def infer():
    # Load model
    encoder = Encoder(vocab_size=VOCAB_SIZE, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM).to(device)
    attention = Attention(hidden_dim=HIDDEN_DIM).to(device)
    decoder = Decoder(hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, attention=attention, max_len=MAX_LEN).to(device)
    model = TTSModel(encoder, decoder).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Create dataset and dataloader
    dataset = AudioDataset(CSV_FILE)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for mel_spectrograms, transcriptions in dataloader:
            mel_spectrograms = mel_spectrograms.to(torch.float32).to(device)

            # Convert transcriptions to tensors
            text_inputs = text_to_tensor(transcriptions, vocab).to(device)

            # Generate outputs
            outputs = model(text_inputs, mel_spectrograms)
            mel_spectrogram_output = outputs[0].squeeze(0).cpu().numpy()  # Shape: [80, length]

            # Handle mel_spectrogram length
            if mel_spectrogram_output.shape[1] != MAX_LEN:
                if mel_spectrogram_output.shape[1] < MAX_LEN:
                    mel_spectrogram_output = np.pad(mel_spectrogram_output, ((0, 0), (0, MAX_LEN - mel_spectrogram_output.shape[1])), mode='constant')
                else:
                    mel_spectrogram_output = mel_spectrogram_output[:, :MAX_LEN]

            # Convert mel spectrogram to waveform
            waveform = mel_to_waveform(mel_spectrogram_output)

            # Save the waveform to a file
            output_filename = 'output.wav'
            save_waveform(waveform, output_filename)

            # Optionally print the output shape and transcription
            print(f'Output mel spectrogram shape: {mel_spectrogram_output.shape}')
            print(f'Transcription: {transcriptions}')
            print(f'Waveform saved to {output_filename}')
            break

if __name__ == '__main__':
    infer()