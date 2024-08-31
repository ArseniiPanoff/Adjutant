import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        hidden = hidden[-1]
        batch_size = hidden.size(0)

        hidden_expanded = hidden.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
        #print(f'Hidden expanded shape: {hidden_expanded.shape}')  # Should be [batch_size, seq_len, hidden_dim]
        #print(f'Encoder outputs shape: {encoder_outputs.shape}')  # Should be [batch_size, seq_len, hidden_dim]

        combined = torch.cat((hidden_expanded, encoder_outputs), dim=2)
        #print(f'Combined shape: {combined.shape}')  # Should be [batch_size, seq_len, hidden_dim * 2]

        attn_weights = torch.sum(self.v * torch.tanh(self.attn(combined)), dim=2)
        attn_weights = F.softmax(attn_weights, dim=1)
        #print(f'Attention weights shape: {attn_weights.shape}')  # Should be [batch_size, seq_len]

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        #print(f'Context shape: {context.shape}')  # Should be [batch_size, 1, hidden_dim]

        return context.squeeze(1), attn_weights



