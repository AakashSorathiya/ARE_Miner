import pandas as pd
from fastapi import HTTPException
from textblob import TextBlob
from server.model import get_model, get_vocabs
import torch
from torch.utils.data import DataLoader, TensorDataset
# from nltk.tokenize import sent_tokenize
import re
from bs4 import BeautifulSoup
from collections import defaultdict
import json
# import stanza
# from stanza.models.common.doc import Document
import time
# from stanza_batch import batch
# from typing import List
import spacy
# import collections

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = get_model()
model.to(device)
word_to_ix, tag_to_ix = get_vocabs()
ix_to_tag = {ix: tag for tag, ix in tag_to_ix.items()}
# nlp_pipeline = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,sentiment')
nlp_pipeline = spacy.load('en_core_web_sm')
BATCH_SIZE = 32
pad_idx=0


def processEdaRequest(dataframe: pd.DataFrame):
    # data validation
    print(dataframe.columns.values)
    if not set(['App', 'Review', 'Date']).issubset(dataframe.columns.values):
        raise HTTPException(status_code=400, detail='Invalid file format. Required columns (App, Review, and Date) are missing.')

    # reviews per app
    reviews_per_app = dataframe.groupby('App')['Review'].count()
    app_distribution = {}
    for app in reviews_per_app.keys():
        app_distribution.update({app: int(reviews_per_app.get(app))})

    # sentiment distribution
    dataframe['sentiment'] = dataframe['Review'].apply(_getSentiment)
    sentiments_count = dataframe.groupby('sentiment')['Review'].count()
    sentiment_distribution = {}
    for sentiment in sentiments_count.keys():
        sentiment_distribution.update({sentiment: int(sentiments_count.get(sentiment))})

    # avg word count
    dataframe['word_count'] = dataframe['Review'].apply(_getWordCount)
    avg_word_count = dataframe['word_count'].mean()

    # time distribution
    pd_date = pd.to_datetime(dataframe['Date'])
    reviews_over_time = pd_date.groupby(pd_date.dt.to_period('M')).size()
    time_distribution = {}
    for date in reviews_over_time.keys():
        time_distribution.update({date.start_time.strftime('%Y-%m'): int(reviews_over_time.get(date))})

    # convert to json
    json_records = json.loads(dataframe.to_json(orient='index'))
    
    # create response
    response = {'avg_word_count': avg_word_count, 'sentiment_distribution': sentiment_distribution, 'app_distribution': app_distribution, 
                'time_distribution': time_distribution, 'records': json_records}

    return response

def processExtractRequirementRequest(dataframe):
    # data validation
    print(dataframe.columns.values)
    if not set(['App', 'Review', 'Date']).issubset(dataframe.columns.values):
        raise HTTPException(status_code=400, detail='Invalid file format. Required columns (App, Review, and Date) are missing.')

    # start = time.process_time()
    # preprocess data for training
    all_clean_reviews, reviews_tensors, all_tokens = _preprocessData(dataframe)
    data_loader = DataLoader(reviews_tensors, batch_size=BATCH_SIZE, shuffle=False)
    # print(f'Time for preprocess: {time.process_time()-start}')
    # get all predictions
    # start = time.process_time()
    all_predictions = _predictRequirements(data_loader)
    # print(f'Time for prediction: {time.process_time()-start}')
    # convert tags to requirements and corresponsing sentiments
    # start = time.process_time()
    all_requirements_sentiments = _extractRequirementsAndSentiments(all_clean_reviews, all_tokens, all_predictions)
    # print(f'Time for _extractRequirementsAndSentiments: {time.process_time()-start}')

    # get statistics
    # start = time.process_time()
    dataframe = pd.concat([dataframe, pd.DataFrame({'requirements': all_requirements_sentiments})], axis=1)
    dataframe['pd_date'] = pd.to_datetime(dataframe['Date'])
    statistics = _getRequirementsStatistics(dataframe)
    dataframe = dataframe.drop('pd_date', axis=1)
    # print(f'Time for statistics: {time.process_time()-start}')

    response = statistics
    response.update({'records': json.loads(dataframe.to_json(orient='index'))})
    return response

def _getWordCount(review):
    return len(str(review).split())

def _getSentiment(review):
    # doc = nlp_pipeline(review)
    score = TextBlob(review).sentiment.polarity
    # score = doc.sentences[0].sentiment
    if score < 0:
        return 'negative'
    elif score == 0:
        return 'neutral'
    else:
        return 'positive'
    
# def _runStanzaPipeline(reviews):
#     docs: List[Document] = []
#     for doc in batch(reviews, nlp_pipeline, batch_size=32):
#         docs.append(doc)
#     return docs

def _runSpacyPipeline(reviews):
    return list(nlp_pipeline.pipe(reviews, disable=["tok2vec", "parser", "ner"]))

def _preprocessData(dataframe: pd.DataFrame):
    # return sentences (for sentiments) and sentences tensors (for model)
    # all_clean_reviews = []
    # all_review_tensors = []
    # all_tokens = []
    reviews = dataframe['Review'].to_list()
    all_clean_reviews = _cleanContent(reviews)
    spacy_docs = _runSpacyPipeline(all_clean_reviews)
    all_tokens = [[token.lemma_ for token in doc] for doc in spacy_docs]
    max_len = max([len(tokens) for tokens in all_tokens])
    reviews_idx = [[word_to_ix.get(word, word_to_ix['<UNK>']) for word in tokens] + [pad_idx]*(max_len-len(tokens)) for tokens in all_tokens]
    reviews_tensors = torch.tensor(reviews_idx, dtype=torch.long)
    # tensor_dataset = TensorDataset(reviews_tensors)
    # for i in range(len(reviews)):
    #     # sentences_for_review = sent_tokenize(review)
    #     # sentence_tensors_for_review = []
    #     # review_tokens = []
    #     # for sentence in sentences_for_review:
    #     #     tokens = _cleanContent(sentence)
    #     #     sentence_idx = [word_to_ix.get(word, word_to_ix['<UNK>']) for word in tokens]
    #     #     sentence_tensor = torch.tensor([sentence_idx], dtype=torch.long)
    #     #     sentence_tensors_for_review.append(sentence_tensor)
    #     #     review_tokens.append(tokens)
    #     # all_sentences.append(sentences_for_review)
    #     # all_sentence_tensors.append(sentence_tensors_for_review)
    #     # all_tokens.append(review_tokens)
    #     # clean_review, tokens = _cleanContent(review)
    #     # tokens = [token.lemma for token in stanza_docs[i].sentences[0].words]
    #     tokens = [token.lemma_ for token in spacy_docs[i]]
    #     review_idx = [word_to_ix.get(word, word_to_ix['<UNK>']) for word in tokens]
    #     review_tensor = torch.tensor([review_idx], dtype=torch.long)
    #     all_tokens.append(tokens)
    #     # all_clean_reviews.append(clean_review)
    #     all_review_tensors.append(review_tensor)
    
    return all_clean_reviews, reviews_tensors, all_tokens

def _cleanContent(reviews):
    # print(sentence)
    clean_reviews=[]
    for review in reviews:
        review = BeautifulSoup(str(review), features='lxml').get_text()
        review = re.sub("[^a-zA-Z]", " ", review)
        review = review.lower()
        clean_review = ' '.join(review.split())
        clean_reviews.append(clean_review)
    # tokens = [token.lemma for token in nlp_pipeline(clean_review).sentences[0].words]
    # return clean_review, tokens
    return clean_reviews

def _predictRequirements(data_loader: DataLoader):
    # return predicted BIO tags
    model.eval()
    all_predictions = []
    for reviews_batch in data_loader:
        reviews_batch = reviews_batch.to(device)
        mask = (reviews_batch != 0)
        emissions = model(reviews_batch)
        batch_pred_idx = model.decode(emissions, mask)
        batch_pred_tags = [[ix_to_tag[t] for t in pred_idx] for pred_idx in batch_pred_idx]
        all_predictions.extend(batch_pred_tags)

    # for review_tensor in all_review_tensors:
        # review_predictions = []
        # for sentence_tensor in sentence_tensors_for_review:
        #     sentence_tensor = sentence_tensor.to(device)
        #     emissions = model(sentence_tensor)
        #     pred_tags_ix = model.decode(emissions)
        #     pred_tags = [ix_to_tag[t] for t in pred_tags_ix[0]]
        #     review_predictions.append(pred_tags)
        # review_tensor = review_tensor.to(device)
        # emissions = model(review_tensor)
        # pred_tags_ix = model.decode(emissions)
        # pred_tags = [ix_to_tag[t] for t in pred_tags_ix[0]]
        # all_predictions.append(pred_tags)

    return all_predictions

def _extractRequirementsAndSentiments(all_clean_reviews, all_tokens, all_predictions):
    all_requirements_sentiments = []
    for i, clean_review in enumerate(all_clean_reviews):
        review_requirements_sentiments = []
        # for j, sentence in enumerate(review_sentences):
        #     tokens, tags = all_tokens[i][j], all_predictions[i][j]
        #     requirements = _tagToRequirements(tokens, tags)
        #     sentiment = _getSentiment(sentence)
        tokens, tags = all_tokens[i], all_predictions[i]
        requirements = _tagToRequirements(tokens, tags)
        sentiment = _getSentiment(clean_review)
        for req in requirements:
            review_requirements_sentiments.append({'requirement': req, 'sentiment': sentiment})
        all_requirements_sentiments.append(review_requirements_sentiments)
    return all_requirements_sentiments

def _tagToRequirements(tokens, tags):
    entities = []
    current_entity = None
    
    for i, tag in enumerate(tags):
        if tag == 'B':
            if current_entity:
                entities.append(' '.join(current_entity))
            current_entity = [tokens[i]]
        elif tag == 'I':
            if current_entity is None:
                current_entity = [tokens[i]]
            else:
                current_entity.append(tokens[i])
        elif tag == 'O':
            if current_entity:
                entities.append(' '.join(current_entity))
                current_entity = None
    
    if current_entity:
        entities.append(' '.join(current_entity))
    
    return entities

def _getRequirementsStatistics(dataframe: pd.DataFrame):
    requirements_for_app = defaultdict(int)
    word_distribution = defaultdict(int)
    sentiment_distribution = defaultdict(int)
    requirements_for_reviews = defaultdict(int)
    time_distribution = defaultdict(int)
    requirements_frequency = defaultdict(int)

    for idx, row in dataframe.iterrows():
        # print(row)
        # if not isinstance(row['requirements'], float):
        num_requirements = len(row['requirements'])
        requirements_for_app[row['App']]+=num_requirements
        requirements_for_reviews[num_requirements]+=1
        time_distribution[row['pd_date'].strftime('%Y-%m')]+=num_requirements
        for req in row['requirements']:
            word_len = _getWordCount(req['requirement'])
            word_distribution[word_len]+=1
            sentiment_distribution[req['sentiment']]+=1
            requirements_frequency[req['requirement']]+=1
    
    # freq_requirements = collections.Counter(requirements_frequency)
    freq_requirements = dict(sorted(requirements_frequency.items(), key=lambda x: -x[1])[:5])
    return {'distribution_over_apps': requirements_for_app, 'word_count_distribution': word_distribution, 'sentiment_distribution': sentiment_distribution, 
            'distribution_over_reviews': requirements_for_reviews, 'distribution_over_time': time_distribution, 'top_five_req': freq_requirements}