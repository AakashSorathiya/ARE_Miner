import torch.nn as nn
import torchcrf
  
class Seq2SeqModel(nn.Module):
    def __init__(self, tagset_size, encoder, decoder, attention):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.crf = torchcrf.CRF(tagset_size, batch_first=True)

    def forward(self, sentence):
        out, _ = self.encoder(sentence)
        out = self.attention(out)
        emissions = self.decoder(out)
        
        return emissions
    
    def loss(self, emissions, tags, mask=None):
        return -self.crf(emissions, tags, mask=mask, reduction='mean')
    
    def decode(self, emissions, mask=None):
        return self.crf.decode(emissions, mask=mask)