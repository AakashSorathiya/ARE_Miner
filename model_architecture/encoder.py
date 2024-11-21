import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, glove_embeddings):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(glove_embeddings, freeze=False)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
    
    def forward(self, inputs):
        embeds = self.embedding(inputs)
        out, hidden = self.encoder(embeds)
        return out, hidden