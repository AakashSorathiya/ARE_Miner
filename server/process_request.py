import pandas as pd

def process_eda_request(dataframe):
    # convert to tokens and calculate avg word length, calculate reviews per app, and sentiments
    return

def process_extract_requirement_request(dataframe):
    # preprocess data, calculate sentiments, get the predicted tags, extract requirements
    sentences, sentences_tensors = _preprocess_data(dataframe)
    predictions = _predict_requirements(sentences_tensors)
    return

def _preprocess_data(dataframe):
    # return sentences (for sentiments) and sentences tensors (for model)
    return

def _predict_requirements(data):
    # return predicted BIO tags
    return