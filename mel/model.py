import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_len=995):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.max_len = max_len

    def forward(self, x, hidden):
        outputs, (hidden, cell) = self.rnn(x, hidden)
        predictions = self.fc(outputs)

        # Ensure predictions match max_len
        if predictions.size(1) < self.max_len:
            # Create padding tensor on the same device as predictions
            padding = torch.zeros((predictions.size(0), self.max_len - predictions.size(1), predictions.size(2)),
                                  device=predictions.device)
            predictions = torch.cat((predictions, padding), dim=1)
        elif predictions.size(1) > self.max_len:
            predictions = predictions[:, :self.max_len, :]

        # Transpose predictions to match target shape [batch_size, output_dim, max_len]
        predictions = predictions.transpose(1, 2)

        return predictions, (hidden, cell)

class TTSModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(TTSModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, text_input, mel_target):
        encoder_output, hidden = self.encoder(text_input)
        mel_outputs, _ = self.decoder(encoder_output, hidden)
        return mel_outputs