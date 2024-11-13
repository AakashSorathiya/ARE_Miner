import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchtext.vocab import GloVe
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import ast
from model_architecture import Seq2SeqModel, BiLSTMEncoder, LSTMDecoder, SelfAttention

truth_dataset = pd.read_csv('datafiles/true_tags.csv')
truth_dataset.info()

truth_dataset['clean_content'] = truth_dataset['clean_content'].apply(ast.literal_eval)
truth_dataset['tags'] = truth_dataset['tags'].apply(ast.literal_eval)
truth_dataset = truth_dataset[truth_dataset['clean_content'].apply(len) > 0]

all_sentences = truth_dataset['clean_content'].to_list()
all_tags = truth_dataset['tags'].to_list()

word_to_ix = {word: i+1 for i, word in enumerate(set([w for s in all_sentences for w in s]))}
word_to_ix['<PAD>']=0
word_to_ix['<UNK>']=len(word_to_ix)
tag_to_ix = {'<PAD>': 0, 'B': 1, 'I': 2, 'O': 3}
ix_to_tag = {ix: tag for tag, ix in tag_to_ix.items()}

BATCH_SIZE = 32

def prepare_data(sentences, tags, word_to_ix, tag_to_ix, pad_idx=0):
    max_len = max(len(s) for s in sentences)
    
    sentences_idx = [[word_to_ix[word] for word in sent] + [pad_idx] * (max_len - len(sent)) for sent in sentences]
    tags_idx = [[tag_to_ix[tag] for tag in tag_seq] + [pad_idx] * (max_len - len(tag_seq)) for tag_seq in tags]
    
    sentences_tensor = torch.tensor(sentences_idx, dtype=torch.long)
    tags_tensor = torch.tensor(tags_idx, dtype=torch.long)
    
    return TensorDataset(sentences_tensor, tags_tensor)

train_data = prepare_data(all_sentences, all_tags, word_to_ix, tag_to_ix)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

HIDDEN_DIM = 512
VOCAB_SIZE = len(word_to_ix)
TAGSET_SIZE = len(tag_to_ix)
EMBEDDING_DIM = 300

glove = GloVe(name='6B', dim=EMBEDDING_DIM, cache='model_codes/.vector_cache')

glove_embeddings = torch.zeros(VOCAB_SIZE, EMBEDDING_DIM)

for word, idx in word_to_ix.items():
    if word in glove.stoi:
        glove_embeddings[idx] = glove[word]
    else:
        glove_embeddings[idx] = torch.randn(EMBEDDING_DIM)

encoder = BiLSTMEncoder(EMBEDDING_DIM, HIDDEN_DIM, glove_embeddings)
decoder = LSTMDecoder(HIDDEN_DIM, TAGSET_SIZE)
attention = SelfAttention(HIDDEN_DIM)
model = Seq2SeqModel(TAGSET_SIZE, encoder, decoder, attention)

optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        total_loss = 0
        for sentences_batch, tags_batch in train_loader:
            mask = (sentences_batch != 0)
            optimizer.zero_grad()
            
            emissions = model(sentences_batch)
            # print(emissions.shape, tags_batch.shape)
            loss = model.loss(emissions, tags_batch, mask)
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()
        print(f"Train Loss: {total_loss / len(train_loader)}")
    
train_model(model, train_loader, optimizer, 15)

torch.save(model, 'model_codes/model/model.pth')
torch.save(glove_embeddings, 'model_codes/model/embeddings.pth')
torch.save(model.state_dict(), 'model_codes/model/model_dict.pth')
torch.save(encoder.state_dict(), 'model_codes/model/encoder_dict.pth')
torch.save(decoder.state_dict(), 'model_codes/model/decoder_dict.pth')
torch.save(attention.state_dict(), 'model_codes/model/attention_dict.pth')