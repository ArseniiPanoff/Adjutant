import torch
import torch.nn as nn

from mel.cbhg import CBHG

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout_rate=0.3):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.cbhg = CBHG(emb_dim, hidden_dim)  # CBHG output should match hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)  # hidden_dim should match

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        cbhg_output = self.cbhg(embedded.permute(0, 2, 1))  # Permute to (batch_size, seq_len, hidden_dim)

        # Print dimensions for debugging
        #print(f'Encoder embedded shape: {embedded.shape}')
        #print(f'Encoder CBHG output shape: {cbhg_output.shape}')

        outputs, (hidden, cell) = self.rnn(cbhg_output)  # No need to permute back

        # Print dimensions for debugging
        #print(f'Encoder RNN output shape: {outputs.shape}')
        #print(f'Encoder hidden state shape: {hidden.shape}')
        #print(f'Encoder cell state shape: {cell.shape}')

        return outputs, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, attention, max_len=416, dropout_rate=0.3):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.attention = attention
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.max_len = max_len

        # Linear layer to match the mel bins (80) to hidden_dim (256)
        self.linear = nn.Linear(80, hidden_dim)

    def forward(self, x, hidden, encoder_outputs):
        # Apply attention
        context, attn_weights = self.attention(hidden[0], encoder_outputs)

        # Print original shape of x
        #print(f'Original x shape: {x.shape}')

        # Transpose x to have shape (batch_size, seq_len, mel_bins)
        x = x.transpose(1, 2)  # Change from (batch_size, mel_bins, seq_len) to (batch_size, seq_len, mel_bins)

        # Apply linear transformation to the mel_bins dimension (last dimension)
        x = self.linear(x)

        # Print shape after linear transformation
        #print(f'x shape after linear and transpose: {x.shape}')

        # Repeat context to match x's sequence length dimension
        context_repeated = context.unsqueeze(1).repeat(1, x.size(1), 1)

        # Concatenate context with x
        rnn_input = torch.cat((x, context_repeated), dim=2)

        # Print dimensions for debugging
        #print(f'Decoder rnn_input shape: {rnn_input.shape}')

        # Check rnn_input shape
        assert rnn_input.size(
            -1) == self.hidden_dim * 2, f"Expected rnn_input size {self.hidden_dim * 2}, but got {rnn_input.size(-1)}"

        # Ensure hidden and cell state are tensors (tuple of tensors)
        if isinstance(hidden, tuple):
            hidden_state, cell_state = hidden
        else:
            hidden_state, cell_state = hidden, torch.zeros_like(hidden)

        # Pass hidden and cell states to LSTM
        outputs, (hidden_state, cell_state) = self.rnn(rnn_input, (hidden_state, cell_state))
        outputs = self.dropout(outputs)
        predictions = self.fc(outputs)
        # Transpose predictions to match target shape (if necessary)
        predictions = predictions.transpose(1, 2)
        return predictions, (hidden_state, cell_state), attn_weights


class TTSModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(TTSModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, text_input, mel_target):
        encoder_output, hidden = self.encoder(text_input)

        # Print dimensions for debugging
        #print(f'TTSModel encoder_output shape: {encoder_output.shape}')
        #print(f'TTSModel hidden state shape: {hidden[0].shape}')

        mel_outputs, _, attn_weights = self.decoder(mel_target, hidden, encoder_output)

        # Print dimensions for debugging
        #print(f'TTSModel mel_outputs shape: {mel_outputs.shape}')
        #print(f'TTSModel attention weights shape: {attn_weights.shape}')

        return mel_outputs, attn_weights

