import pandas as pd
from fastapi import HTTPException
from textblob import TextBlob

def processEdaRequest(dataframe: pd.DataFrame):
    # convert to tokens and calculate avg word length, calculate reviews per app, and sentiments
    print(dataframe.columns.values)
    if not set(['App', 'Review']).issubset(dataframe.columns.values):
        raise HTTPException(status_code=400, detail='Invalid file format. Required columns (App and Review) are missing.')

    reviews_per_app = dataframe.groupby('App')['Review'].count()
    print(reviews_per_app)

    dataframe['polarity'] = dataframe['Review'].apply(_getPolarity)
    dataframe['word_count'] = dataframe['Review'].apply(_getWordCount)

    return dataframe

def processExtractRequirementRequest(dataframe):
    # preprocess data, calculate sentiments, get the predicted tags, extract requirements
    sentences, sentences_tensors = _preprocessData(dataframe)
    predictions = _predictRequirements(sentences_tensors)
    return

def _getWordCount(review):
    return len(str(review).split())

def _getPolarity(review):
    return TextBlob(review).sentiment.polarity

def _preprocessData(dataframe):
    # return sentences (for sentiments) and sentences tensors (for model)
    return

def _predictRequirements(data):
    # return predicted BIO tags
    return