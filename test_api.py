import pandas as pd
import requests
import json

df = pd.read_csv('test_clean_data.csv')
df = df.drop('Unnamed: 0',axis=1)
df_json = df[:1].to_json(orient = "records")
# header = {"Content-Type": "application/json"}
  
url = 'https://apipredictions.herokuapp.com/predict'

# payload={"data": df_json}

r = requests.post(url, json=json.loads(df_json))

print(r.json())


