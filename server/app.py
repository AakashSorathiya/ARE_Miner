from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from server.process_request import processEdaRequest, processExtractRequirementRequest
import uvicorn
import pandas as pd
import time

# sorting based on number of words
# filter based on sentiments and range of number of words
# checkbox (app, sentiments) to filter requirements
# distribution of requirements over number of reviews, number of requirements for each app, sentiment distribution for requirements, distribution over time, number of words for requirements

API = FastAPI()
API.add_middleware(HTTPSRedirectMiddleware)

# /eda - input: csv, output: csv
# format of input csv file: App,Review,Date
# format of output json: {avg_word_count: 12, sentiment_distribution: {negative: 12, positive: 12, neutral: 12}, app_distribution: {app1: 12, app2: 12}, time_distribution: {date1: numberOfRecords}, records: [{App,ReviewId,Review,Date,sentiment,word_count}]}
@API.post('/eda')
def perform_eda(csv_file: UploadFile):
    print(f'request start, reading file, {csv_file.filename}')
    
    filename = csv_file.filename
    file_format = filename.split('.')[-1]
    if not file_format=='csv':
        raise HTTPException(status_code=400, detail='CSV file format is required.')
    
    df = pd.read_csv(csv_file.file)
    print(f'df created, size - {len(df)}')
    
    data = processEdaRequest(df)
    
    return data

# /extract_requirements - input:csv, output: csv
# format of input csv file: App,Review
# format of output csv file: App,Review,Requirements,Sentiments
@API.post('/extract_requirements')
def extract_requirements(csv_file: UploadFile):
    # start = time.process_time()
    filename = csv_file.filename
    file_format = filename.split('.')[-1]
    if not file_format=='csv':
        raise HTTPException(status_code=400, detail='CSV file format is required.')
    
    df = pd.read_csv(csv_file.file)
    print(f'df created, size - {len(df)}')
    
    response = processExtractRequirementRequest(df)
    # print(f'Time to complete the request: {time.process_time()-start}')
    return response

if __name__ == "__main__":
    uvicorn.run(API, host="0.0.0.0", port=80, reload=True, ssl_keyfile='../key.pem', ssl_certfile='../cert.pem')