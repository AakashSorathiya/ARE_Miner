from fastapi import FastAPI, UploadFile
from process_request import process_eda_request, process_extract_requirement_request

API = FastAPI()

# /get_eda - input: csv, output: {avg_word_length: x, sentiments: {positive: a1, neutral: a2, negative: a3}, app_distribution: {app1: n1, app2: n2}}
@API.post('/get_eda')
def get_eda(csv_file: UploadFile):
    process_eda_request()
    return

# /extract_requirements - input:csv, output: {requirements: ['r1', 'r2'], sentiments: ['negative' 'positive']}
@API.post('extract_requirements')
def extract_requirements(csv_file: UploadFile):
    process_extract_requirement_request()
    return