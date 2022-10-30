import pickle
import pandas as pd
import lightgbm as lgb
import re
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('./data/train_data.csv')

train_data = train_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
# print(train_data.columns)
x = train_data.drop(['Unnamed: 0', 'Unnamed: 0.1','TARGET'],axis=1).copy()
y = train_data['TARGET'].copy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=42)

cat_cols = ['NAME_CONTRACT_TYPE_x','CODE_GENDER','FLAG_OWN_CAR', 'FLAG_OWN_REALTY','CNT_CHILDREN','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','FLAG_MOBIL','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL',
            'FLAG_EMAIL','REGION_RATING_CLIENT','WEEKDAY_APPR_PROCESS_START_x','HOUR_APPR_PROCESS_START_x','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY',
            'LIVE_CITY_NOT_WORK_CITY','FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9',
             'FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15',
            'FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21','FLAG_LAST_APPL_PER_CONTRACT','NAME_CONTRACT_TYPE_y',
            'NFLAG_LAST_APPL_IN_DAY']

num_cols = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY_x','REGION_POPULATION_RELATIVE','DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION',
            'DAYS_ID_PUBLISH','CNT_FAM_MEMBERS','EXT_SOURCE_2','EXT_SOURCE_3','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','DAYS_LAST_PHONE_CHANGE','AMT_REQ_CREDIT_BUREAU_HOUR',
            'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR','AMT_ANNUITY_y','AMT_APPLICATION','AMT_GOODS_PRICE',
            'NAME_CASH_LOAN_PURPOSE','NAME_CONTRACT_STATUS','DAYS_DECISION','NAME_PAYMENT_TYPE','CODE_REJECT_REASON','NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY','NAME_PORTFOLIO','NAME_PRODUCT_TYPE','SELLERPLACE_AREA',
            'CNT_PAYMENT','NAME_YIELD_GROUP','AMT_CREDIT_PERCENT','AMT_APPLICATION_PERCENT','AMT_GOODS_PRICE_PERCENT']

# Creating column transformer to select all column names that has Binary & a list for non Binary.

# Define categorical pipeline for imputing the missing values
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Define numerical pipeline for imputing the missing values
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

# Combine categorical and numerical pipelines for column transformer
preprocessor = ColumnTransformer([
    ('cat', cat_pipe, cat_cols),
    ('num', num_pipe, num_cols)
])

import lightgbm as lgb

lgb = lgb.LGBMClassifier(class_weight='balanced',n_estimators=300,learning_rate=0.05)

pipe_lb = Pipeline([
    ('preprocess', preprocessor),('model',lgb)
])

pipe_lb.fit(x_train,y_train)

pickle.dump(lgb, open('./data/lgbmodel.pkl', 'wb'))

# model = pickle.load(open('./data/lgbmodel.pkl', 'rb'))
# predict = model.predict(x_test)
# print(predict)
