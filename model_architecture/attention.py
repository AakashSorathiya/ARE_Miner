import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, input):
        attention_weights = torch.tanh(self.attention(input))
        attention_weights = torch.softmax(attention_weights, dim=1)
        out = input * attention_weights
        return out