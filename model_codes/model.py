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

cross_domain_data = []
for app in truth_dataset['App id'].value_counts().keys():
    # print(app)
    dev_sentences = truth_dataset[truth_dataset['App id']!=app]['clean_content'].to_list()
    dev_tags = truth_dataset[truth_dataset['App id']!=app]['tags'].to_list()
    test_sentences = truth_dataset[truth_dataset['App id']==app]['clean_content'].to_list()
    test_tags = truth_dataset[truth_dataset['App id']==app]['tags'].to_list()

    # print(len(dev_sentences), len(test_sentences))

    train_sentences, val_sentences, train_tags, val_tags = train_test_split(dev_sentences, dev_tags, test_size=0.2, random_state=42)

    train_data = prepare_data(train_sentences, train_tags, word_to_ix, tag_to_ix)
    val_data = prepare_data(val_sentences, val_tags, word_to_ix, tag_to_ix)
    test_data = prepare_data(test_sentences, test_tags, word_to_ix, tag_to_ix)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    cross_domain_data.append({'app': app, 'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader, 'test_sentences': test_sentences})

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

def train_model(model, train_loader, optimizer):
    model.train()
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
    
def evaluate_model(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for sentences_batch, tags_batch in val_loader:
            mask = (sentences_batch != 0)
            emissions = model(sentences_batch)
            loss = model.loss(emissions, tags_batch, mask)
            total_loss += loss.item()
    print(f"Validation Loss: {total_loss / len(val_loader)}")

def evaluate_model(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for sentences_batch, tags_batch in val_loader:
            mask = (sentences_batch != 0)
            emissions = model(sentences_batch)
            loss = model.loss(emissions, tags_batch, mask)
            total_loss += loss.item()
    print(f"Validation Loss: {total_loss / len(val_loader)}")

def train_eval_loop(model, train_loader, val_loader, optimizer, epochs=10):
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        for phase in ['train', 'val']:
            if phase=='train':
                train_model(model, train_loader, optimizer)
            else:
                evaluate_model(model, val_loader)

def test_model(model, test_loader):
    model.eval()
    all_predictions = []
    masks = []
    with torch.no_grad():
        for sentences_batch, _ in test_loader:
            mask = (sentences_batch != 0)
            emissions = model(sentences_batch)
            predictions = model.decode(emissions, mask=mask)
            pred_tags = [[ix_to_tag[t] for t in seq] for seq in predictions]
            
            all_predictions.extend(pred_tags)
            masks.extend(mask)
    
    return all_predictions, masks

def calculate_metrics_1(predictions, true_tags, true_tokens):
    
    # print(len(predictions), len(true_tags), len(test_sentences))
    levels = 3
    
    def extract_entities(seq, sentence):

        entities = []
        current_entity = None
        
        for i, tag in enumerate(seq):
            if tag == 'B':
                if current_entity:
                    entities.append(current_entity)
                current_entity = [sentence[i]]
            elif tag == 'I':
                if current_entity is None:
                    current_entity = [sentence[i]]
                else:
                    current_entity.append(sentence[i])
            elif tag == 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def is_match(f1, f2, n):
        """
        Check if two features match at level n.
        Conditions:
        1. One feature is equal to or is a subset of the other
        2. Absolute length difference is at most n
        """
        f1=set(f1)
        f2=set(f2)
        
        is_subset = f1.issubset(f2) or f2.issubset(f1)
        length_diff = abs(len(f1) - len(f2))
        
        return is_subset and length_diff <= n


    all_true_entites = []
    all_pred_entites = []
    
    for pred_seq, true_seq, token_seq in zip(predictions, true_tags, true_tokens):
        true_entities = extract_entities(true_seq, token_seq)
        pred_entities = extract_entities(pred_seq, token_seq)
        # print(pred_entities)
        # print(true_entities)

        all_true_entites.append(true_entities)
        all_pred_entites.append(pred_entities)

    total_true = len(all_true_entites)
    total_pred = len(all_pred_entites)
    metrics = {}
    # print(total_pred, total_true, all_levels_TPs)

    for level in range(levels):
        tp = 0
        fp = 0
        fn = 0

        for true_entities, pred_entities in zip(all_true_entites, all_pred_entites):   
            matched_true = set()
            for pred_entity in pred_entities:
                found_match = False
                
                for i, true_entity in enumerate(true_entities):
                    if i not in matched_true and is_match(pred_entity, true_entity, level):
                        tp += 1
                        matched_true.add(i)
                        found_match = True
                        break
                
                if not found_match:
                    fp += 1

            fn += len(true_entities) - len(matched_true)
            # print(fn)
    
        print(level, tp, fp, fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        level_name = ['exact', 'n-1', 'n-2'][level]
        metrics.update({
            f'{level_name}_precision': precision,
            f'{level_name}_recall': recall,
            f'{level_name}_f1': f1
        })
    
    return metrics

def train_test_model_across_domains(epochs):
    results=[]
    
    for data in cross_domain_data:
        # print(f'For app: {data['app']}')
        # model = Seq2SeqModel(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, TAGSET_SIZE, glove_embeddings)
        # optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_eval_loop(model, data['train_loader'], data['val_loader'], optimizer, epochs)
        test_predictions, masks = test_model(model, test_loader)
        f1_scores = calculate_metrics_1(test_predictions, test_tags, test_sentences)
        results.append({data['app']: f1_scores})
    
    return results

results = train_test_model_across_domains(10)

torch.save(model, 'model_codes/model/model.pth')
torch.save(glove_embeddings, 'model_codes/model/embeddings.pth')
torch.save(model.state_dict(), 'model_codes/model/model_dict.pth')
torch.save(encoder.state_dict(), 'model_codes/model/encoder_dict.pth')
torch.save(decoder.state_dict(), 'model_codes/model/decoder_dict.pth')
torch.save(attention.state_dict(), 'model_codes/model/attention_dict.pth')