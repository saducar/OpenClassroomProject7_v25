from parser import suite
from typing import Union
import pandas as pd
import uvicorn
from fastapi import FastAPI
from schema import Credit_Score_In
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open('lgbmodel.pkl', 'rb'))

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post("/predict")
def predict_credit_score(data: Credit_Score_In):
    data = data.dict()
    # print(data)
    df = pd.DataFrame([data])
    # pred = model.predict(df)
    pred_prob = model.predict_proba(df)
    #{'prediction' : pred[0].tolist(),'prediction_probabilities' : pred_prob[0].max().tolist()}
    return {'prediction_probabilities' : pred_prob[0].max().tolist()}

if __name__ == '__main__':
   uvicorn.run(app,host='127.0.0.1', port=8000)


