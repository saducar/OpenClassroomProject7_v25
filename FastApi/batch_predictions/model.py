from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import lightgbm as lgb
import re

train_data = pd.read_csv('new_train_clean_data.csv')

train_data = train_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
# print(train_data.columns)
x = train_data.drop(['Unnamed0', 'TARGET'],axis=1).copy()
y = train_data['TARGET'].copy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=42)

lgb = lgb.LGBMClassifier(class_weight='balanced',n_estimators=300,learning_rate=0.05)
lgb.fit(x_train, y_train)

pickle.dump(lgb, open('lgbmodel.pkl', 'wb'))

model = pickle.load(open('lgbmodel.pkl', 'rb'))
predict = model.predict(x_test)
print(predict)
