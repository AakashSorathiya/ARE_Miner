import torch
import json
import os
from model_architecture import Seq2SeqModel, LSTMDecoder, BiLSTMEncoder, SelfAttention
from transformers import pipeline

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

model_type = 'new_model'

def get_model():
    if model_type=='baseline':
        tfrex_model = 'quim-motger/t-frex-xlnet-base-cased'
        model = pipeline("ner", model=tfrex_model, device=device)
    else:
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
        model.load_state_dict(torch.load('model_codes/model/model_dict.pth',  map_location=torch.device(device)))
        # model = torch.load('model_codes/model/model.pth')
        # model.to(device)
        model.eval()
    return model

def get_vocabs():
    with open(os.path.join('model_codes/model/', 'vocabs.json'), 'r') as f:
        vocabs = json.load(f)
    return vocabs['input_vocab'], vocabs['target_vocab']