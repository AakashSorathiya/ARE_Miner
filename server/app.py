from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from process_request import processEdaRequest, processExtractRequirementRequest
import uvicorn
import pandas as pd

API = FastAPI()

# /eda - input: csv, output: csv
# format of input csv file: App,Review
# format of output csv file: App,Review,polarity,word_count
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
    print(data.info())
    
    output_file = data.to_csv(index=False)

    return StreamingResponse(
        iter([output_file]),
        media_type='text/csv',
        headers={'Content-Disposition': 'attachment;file_name=eda.csv'}
    )

# /extract_requirements - input:csv, output: csv
# format of input csv file: App,Review
# format of output csv file: App,Review,Requirements,Sentiments
@API.post('/extract_requirements')
def extract_requirements(csv_file: UploadFile):
    processExtractRequirementRequest()
    return

if __name__ == "__main__":
    uvicorn.run(API, host="0.0.0.0", port=8000, reload=True)