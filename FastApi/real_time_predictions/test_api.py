import pandas as pd
import requests
import json

df = pd.read_csv('test_clean_data.csv')
df = df.drop('Unnamed: 0',axis=1)
header = {"Accept": "application/json","Content-Type": "application/json"}
url = 'http://127.0.0.1:8000/predict'

# payload={"data": df[:1].to_dict(orient="records")}
# r = requests.post(url, data=payload)
# print(r.json())
# print(df[:1])
df_json = df[:1].to_json(orient = "records")
# r = requests.post(url, json=json.loads(df_json), headers=header)

# print(r.json())

print(df_json)