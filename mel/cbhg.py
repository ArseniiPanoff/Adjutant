import torch
import torch.nn as nn
import torch.nn.functional as F

class CBHG(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CBHG, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
        ])
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        #print(f'Input shape: {x.shape}')  # Should be [batch_size, input_dim, seq_len]
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = x.permute(0, 2, 1)  # Change shape to [batch_size, seq_len, hidden_dim]
        #print(f'After conv and permute shape: {x.shape}')
        x, _ = self.rnn(x)
        #print(f'Output of RNN shape: {x.shape}')  # Should be [batch_size, seq_len, hidden_dim]
        return x



