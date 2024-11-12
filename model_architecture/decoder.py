import torch.nn as nn

class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim, tagset_size) -> None:
        super().__init__()
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)
    
    def forward(self, decoder_inputs):
        out, hidden = self.decoder(decoder_inputs)
        emissions = self.fc(out)
        return emissions