import pandas as pd
import uvicorn
from fastapi import FastAPI
from schema import Credit_Score_In
import pickle
from typing import List

app = FastAPI()

model = pickle.load(open('lgbmodel.pkl', 'rb'))

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post("/predict")
def predict_credit_score(features: List[Credit_Score_In]):
    df = pd.DataFrame([i.dict() for i in features])
    # pred = model.predict(df)
    pred_prob = model.predict_proba(df)
    results = pd.DataFrame()

    for i in range(len(pred_prob)):
        # dct = {'prediction' : pred[i],'pred_proba' : pred_prob[i].max()}
        dct = {'pred_proba' : pred_prob[i].max()}
        df_dct = pd.DataFrame([dct])
        results = pd.concat([results, df_dct], ignore_index=True)
        #print(results.head())
      
    return results.to_dict(orient="records")

if __name__ == '__main__':
   uvicorn.run(app, host='127.0.0.1', port=8000)
