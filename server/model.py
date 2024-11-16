import torch
import json
import os
from model_architecture import Seq2SeqModel, LSTMDecoder, BiLSTMEncoder, SelfAttention

def get_model():
    HIDDEN_DIM = 512
    TAGSET_SIZE = 4
    glove_embeddings = torch.load('model_codes/model/embeddings.pth')
    
    encoder = BiLSTMEncoder(glove_embeddings.shape[1], HIDDEN_DIM, glove_embeddings)
    # encoder.load_state_dict(torch.load('model_codes/model/encoder_dict.pth'))
    decoder = LSTMDecoder(HIDDEN_DIM, TAGSET_SIZE)
    # decoder.load_state_dict(torch.load('model_codes/model/decoder_dict.pth'))
    attention = SelfAttention(HIDDEN_DIM)
    # attention.load_state_dict(torch.load('model_codes/model/attention_dict.pth'))
    model = Seq2SeqModel(TAGSET_SIZE, encoder, decoder, attention)
    model.load_state_dict(torch.load('model_codes/model/model_dict.pth'))
    # model = torch.load('model_codes/model/model.pth')
    model.eval()
    return model

def get_vocabs():
    with open(os.path.join('model_codes/model/', 'vocabs.json'), 'r') as f:
        vocabs = json.load(f)
    return vocabs['input_vocab'], vocabs['target_vocab']