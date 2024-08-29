import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from model import Encoder, Decoder, TTSModel
import torch.optim as optim

# Configurations
CSV_FILE = '../filelists/combined_filelist.txt'
BATCH_SIZE = 16  # Increased batch size for more stable training
NUM_EPOCHS = 30  # More epochs to give the model time to learn
LEARNING_RATE = 0.00001  # Adjusted learning rate
VOCAB_SIZE = 16681  # Set to the number of unique tokens
EMB_DIM = 512  # Embedding dimension
HIDDEN_DIM = 512  # Hidden dimension
OUTPUT_DIM = 80  # Number of mel frequency bins
MAX_SPECTROGRAM_LENGTH = 416  # Set to the maximum length of mel spectrograms
MODEL_SAVE_PATH = 'models/tts_model.pth'  # Path to save the model

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def collate_fn(batch):
    mel_spectrograms, transcriptions = zip(*batch)

    mel_spectrograms_padded = [
        np.pad(mel, ((0, 0), (0, MAX_SPECTROGRAM_LENGTH - mel.shape[1])), mode='constant') if mel.shape[1] < MAX_SPECTROGRAM_LENGTH else mel[:, :MAX_SPECTROGRAM_LENGTH]
        for mel in mel_spectrograms]

    mel_spectrograms_array = np.array(mel_spectrograms_padded, dtype=np.float32)
    mel_spectrograms_tensor = torch.from_numpy(mel_spectrograms_array)

    return mel_spectrograms_tensor, transcriptions

def text_to_tensor(transcriptions, vocab, max_length=416):
    sequences = [[vocab.get(word, 0) for word in transcription.split()] for transcription in transcriptions]
    sequences_padded = [seq + [0] * (max_length - len(seq)) if len(seq) < max_length else seq[:max_length] for seq in sequences]

    text_tensor = torch.tensor(sequences_padded, dtype=torch.long)
    return text_tensor

def train():
    dataset = AudioDataset(CSV_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    encoder = Encoder(vocab_size=VOCAB_SIZE, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM).to(device)
    decoder = Decoder(hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, max_len=MAX_SPECTROGRAM_LENGTH).to(device)
    model = TTSModel(encoder, decoder).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    # Generate vocab from your dataset
    vocab = {word: idx for idx, word in enumerate("your_vocab_list".split())}

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        total_samples = 0

        for mel_spectrograms, transcriptions in dataloader:
            mel_spectrograms = mel_spectrograms.to(device)
            text_inputs = text_to_tensor(transcriptions, vocab, max_length=416).to(device)

            optimizer.zero_grad()
            outputs = model(text_inputs, mel_spectrograms)

            # Check if shapes match
            if outputs.shape != mel_spectrograms.shape:
                print(f'Error: Outputs shape {outputs.shape} does not match target shape {mel_spectrograms.shape}')
                continue

            loss = criterion(outputs, mel_spectrograms)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item() * mel_spectrograms.size(0)
            total_samples += mel_spectrograms.size(0)

        avg_loss = epoch_loss / total_samples
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {avg_loss}')

        # Step the scheduler
        scheduler.step(avg_loss)

    # Save the model after training
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'Model saved to {MODEL_SAVE_PATH}')

if __name__ == '__main__':
    train()