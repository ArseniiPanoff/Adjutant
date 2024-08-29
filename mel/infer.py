import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from model import Encoder, Decoder, TTSModel

# Configurations
CSV_FILE = '/home/lwolf/PycharmProjects/Adjutant/filelists/combined_filelist.txt'
MODEL_PATH = '/mel/models/tts_model.pth'
VOCAB_SIZE = 50  # Adjust as needed
EMB_DIM = 256
HIDDEN_DIM = 512
OUTPUT_DIM = 80  # Number of mel bins (you used 80 in the training)
MAX_LEN = 995  # The length of the mel spectrograms

# Dummy vocabulary for illustration; replace with your actual vocabulary
vocab = {word: idx for idx, word in enumerate("your_vocab_list".split())}


def text_to_tensor(transcriptions, vocab, max_length=50):
    sequences = [[vocab.get(word, 0) for word in transcription.split()] for transcription in transcriptions]
    sequences_padded = [seq + [0] * (max_length - len(seq)) if len(seq) < max_length else seq[:max_length] for seq in
                        sequences]

    text_tensor = torch.tensor(sequences_padded, dtype=torch.long)
    return text_tensor


def infer():
    # Load model
    encoder = Encoder(vocab_size=VOCAB_SIZE, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM)
    decoder = Decoder(hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, max_len=MAX_LEN)
    model = TTSModel(encoder, decoder)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Create dataset and dataloader
    dataset = AudioDataset(CSV_FILE)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for mel_spectrograms, transcriptions in dataloader:
            mel_spectrograms = torch.tensor(mel_spectrograms, dtype=torch.float32)

            # Convert transcriptions to tensors
            text_inputs = text_to_tensor(transcriptions, vocab)

            # Generate outputs
            outputs = model(text_inputs, mel_spectrograms)

            # Process outputs: e.g., save or display mel spectrograms
            print(f'Output: {outputs.shape}')
            print(f'Transcription: {transcriptions}')

            # Optionally: Convert mel spectrogram back to waveform (if desired)
            # waveform = some_function_to_convert_mel_to_waveform(outputs)
            # Save or play the waveform


if __name__ == '__main__':
    infer()


