import pandas as pd
from fastapi import HTTPException
from textblob import TextBlob
from server.model import get_model, get_vocabs
import torch

model = get_model()
word_to_ix, tag_to_ix = get_vocabs()
ix_to_tag = {ix: tag for tag, ix in tag_to_ix.items()}


def processEdaRequest(dataframe: pd.DataFrame):
    # convert to tokens and calculate avg word length, calculate reviews per app, and sentiments
    print(dataframe.columns.values)
    if not set(['App', 'Review', 'Date']).issubset(dataframe.columns.values):
        raise HTTPException(status_code=400, detail='Invalid file format. Required columns (App, Review, and Date) are missing.')

    reviews_per_app = dataframe.groupby('App')['Review'].count()
    app_distribution = {}
    for app in reviews_per_app.keys():
        # print(type(reviews_per_app.get(app)))
        app_distribution.update({app: int(reviews_per_app.get(app))})
    # print(app_distribution)

    # dataframe['polarity'] = dataframe['Review'].apply(_getPolarity)
    dataframe['sentiment'] = dataframe['Review'].apply(_getSentiment)
    sentiments_count = dataframe.groupby('sentiment')['Review'].count()
    sentiment_distribution = {}
    for sentiment in sentiments_count.keys():
        sentiment_distribution.update({sentiment: int(sentiments_count.get(sentiment))})
    # print(sentiment_distribution)

    dataframe['word_count'] = dataframe['Review'].apply(_getWordCount)
    avg_word_count = dataframe['word_count'].mean()
    # print(type(avg_word_count))

    pd_date = pd.to_datetime(dataframe['Date'])
    reviews_over_time = pd_date.groupby(pd_date.dt.to_period('M')).size()
    time_distribution = {}
    for date in reviews_over_time.keys():
        time_distribution.update({date.start_time.strftime('%Y-%m'): int(reviews_over_time.get(date))})

    json_records = dataframe.to_json(orient='records')
    
    response = {'avg_word_count': avg_word_count, 'sentiment_distribution': sentiment_distribution, 'app_distribution': app_distribution, 'time_distribution': time_distribution, 'records': json_records}
    print(response)
    return response

def processExtractRequirementRequest(dataframe):
    # preprocess data, calculate sentiments, get the predicted tags, extract requirements
    sentences, sentences_tensors = _preprocessData(dataframe)
    predictions = _predictRequirements(sentences_tensors)
    requirements = _tagToRequirements(sentences.split(), predictions)
    print(predictions, requirements)
    return

def _getWordCount(review):
    return len(str(review).split())

def _getSentiment(review):
    score = TextBlob(review).sentiment.polarity
    if score == 0:
        return 'neutral'
    elif score>0:
        return 'positive'
    else:
        return 'negative'

def _preprocessData(dataframe):
    # return sentences (for sentiments) and sentences tensors (for model)
    
    inference_sentence = 'The app crashes when I try to share photos with my contacts from another social network'
    tokens = inference_sentence.split()
    sentence_idx = [word_to_ix.get(word, word_to_ix['<UNK>']) for word in tokens]
    print(sentence_idx)
    sentence_tensor = torch.tensor([sentence_idx], dtype=torch.long)
    return inference_sentence, sentence_tensor

def _predictRequirements(data):
    # return predicted BIO tags
    model.eval()
    emissions = model(data)
    pred_tags_ix = model.decode(emissions)
    pred_tags = [ix_to_tag[t] for t in pred_tags_ix[0]]
    return pred_tags

def _tagToRequirements(sentences, tags):
    entities = []
    current_entity = None
    
    for i, tag in enumerate(tags):
        if tag == 'B':
            if current_entity:
                entities.append(current_entity)
            current_entity = [sentences[i]]
        elif tag == 'I':
            if current_entity is None:
                current_entity = [sentences[i]]
            else:
                current_entity.append(sentences[i])
        elif tag == 'O':
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities