import streamlit as st 
import pandas as pd
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle 
import requests
import json
import time
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from lime.lime_tabular import LimeTabularExplainer
import shap

st.set_option('deprecation.showPyplotGlobalUse', False)

# Webapp title
st.title('Home Credit Default Risk')

# loading the data

@st.cache(allow_output_mutation=True)
def load_data(file):
    df = pd.read_csv(f'./data/{file}.csv')
    return df

train_df = load_data('train_data')
result = load_data('results')

# Pie Chart

st.write('Pie Chart')
fig4 = plt.figure(figsize=(8, 8))

labels=['Default', 'Non-Default']
sizes=[result[result['Class']==1]['Class'].count(), result[result['Class']==0]['Class'].count()]

plt.pie(sizes, labels=labels, 
        autopct='%1.1f%%', 
        shadow=True,
        startangle=200,
        explode = [0, 0.1])

plt.axis('equal')
#plt.savefig("./images/PieChart.png")
st.pyplot(fig4)

# Credit Amount 

st.write('Credit Amount')
amtCredit=result.sort_values(by='AMT_CREDIT', ascending=False)[['SK_ID_CURR', 'AMT_CREDIT']]
amtCredit.set_index('SK_ID_CURR')[:20].plot.barh(figsize=(10, 10))
#plt.savefig("./images/CreditAmount.png")
plt.show()
st.pyplot()

result["Age(years)"] = abs(result['DAYS_BIRTH']) // 365
train_df["Age(years)"] = abs(train_df['DAYS_BIRTH']) // 365

def get_cat(x):
    return str(x//10*10) + '-' + str((x//10*10)+10)

result['Age_cat']=list(map(get_cat, result["Age(years)"]))
train_df["Age_cat"]=list(map(get_cat, train_df["Age(years)"]))

data = result.groupby(by="Age_cat", as_index=False).mean()
data2 = train_df.groupby(by="Age_cat", as_index=False).mean()

# print(data.head())
st.write("Age Vs Amount Income")
fig1 = plt.figure(figsize = (10, 5))
plt.plot(data["Age_cat"], data["AMT_INCOME_TOTAL"], color="blue")
# result.groupby(['Age(years)','AMT_INCOME_TOTAL']).sum().unstack().plot()
plt.title("Age(years) vs Amount of Income")
#plt.savefig("./images/" + "AGE_AMT_OF_INCOME" + ".png")
# plt.show()
st.pyplot(fig1)
#

st.write("Age vs Total Amount Credit")
fig2 = plt.figure(figsize=(10, 5))
plt.plot(data["Age_cat"], data["AMT_CREDIT"], color="red")
plt.title("Age(years) vs Amount of Credit")
#plt.savefig("./images/" + "AGE_AMT_OF_CREDIT" + ".png")
# plt.show()
st.pyplot(fig2)
#

st.write("Age vs Flag Own Realty")
fig3 = plt.figure(figsize=(10, 5))
plt.plot(data["Age_cat"], data["FLAG_OWN_REALTY"], color="green")
plt.title("Age(years) vs Flag Own Realty")
#plt.savefig("./images/" + "AGE_FLAG_OWN_REALITY" + ".png")
# plt.show()
st.pyplot(fig3)
#
st.write("Age vs Days Employed")
fig4 = plt.figure(figsize=(10, 5))
plt.plot(data2["Age_cat"], data2["DAYS_EMPLOYED"], color="violet")
plt.title("Age(years) vs Days Employed")
#plt.savefig("./images/" + "AGE_DAYS_EMPLOYED" + ".png")
plt.show()
st.pyplot(fig4)

focus={'AMT_CREDIT_PERCENT': "the average between the loan and the income",
       'AMT_APPLICATION':'For how much credit did client ask on the previous application',
       'DAYS_EMPLOYED':'How many days before the application the person started current employment',
       'DAYS_BIRTH':"Client's age in days at the time of application",
       'AMT_GOODS_PRICE':'Goods price of good that client asked for (if applicable) on the previous application',
       'AMT_ANNUITY_x':'Annuity of previous application',
       'AMT_INCOME_TOTAL':'Income of the client',
       'AMT_CREDIT':' Credit amount of the loan'}

st.write(focus)

st.subheader('Result Plots')
is_default = False

try:
    id = st.text_input('Enter Client ID:')
    #id=394688
    prob = result.loc[result['SK_ID_CURR']==id]['TARGET'].values[0]*100
    is_default = prob >= 50
    st.write(f'The client {id} has a {str(round(prob, 1))}% risk of defaulting on their loan.')

    if prob < 80:
      st.write('The client will not get credit')
    else:
        st.write('The client will get credit')
except:
    pass

try:
    result['SK_ID_CURR'] = result['SK_ID_CURR'].astype('str')
    result['DAYS_BIRTH'] = abs(result['DAYS_BIRTH'])
    client = result[result['SK_ID_CURR']==id]
    sameClass = result[result['Class']==int(client['Class'].values[0])]
    if int(client['Class'])==1:
        oppClass=result[result['Class']==0]
    else:
        oppClass=result[result['Class']==1]

    for key, val in focus.items():

        temp = pd.DataFrame(columns=['Target','Average','SameGroup','OppGroup'])
        temp['Target']=client[key]
        temp['Average']=np.average(result[key].values)
        temp['SameGroup']=np.average(sameClass[key].values)
        temp['OppGroup']=np.average(oppClass[key].values)
        temp = temp.T
        fig9 = plt.figure(figsize=(10, 5))
        plt.barh(temp.index, temp[temp.columns[0]], color=plt.cm.Accent_r(np.arange(len(temp))))
        plt.title(key)
        #plt.savefig("./images/"+key+".png")
        plt.show()
        st.pyplot(fig9)

    if is_default:

        st.write("Age Vs Amount Income")
        fig10 = plt.figure(figsize=(10, 5))
        # fig_1, ax_1 = plt.subplots()

        plt.bar(data["Age_cat"], data["AMT_INCOME_TOTAL"], color="blue")
        plt.hlines(y=client["AMT_INCOME_TOTAL"], xmin=0, xmax="60-70")
        plt.vlines(x=client["Age_cat"], ymin=0, ymax=client["AMT_INCOME_TOTAL"]+10000)
        # result.groupby(['Age(years)','AMT_INCOME_TOTAL']).sum().unstack().plot()
        plt.title("Age Groups vs Average Amount of Income")
        #plt.savefig("./images/" + "AVG_AGE_AMT_OF_INCOME_BAR" + ".png")
        plt.show()
        st.pyplot(fig10)

        st.write("Age vs Total Amount Credit")
        fig11 = plt.figure(figsize=(10, 5))
        plt.bar(data["Age_cat"], data["AMT_CREDIT"], color="red")
        plt.hlines(y=client["AMT_CREDIT"], xmin=0, xmax="60-70")
        plt.vlines(x=client["Age_cat"], ymin=0, ymax=client["AMT_CREDIT"]+10000)
        plt.title("Age Groups vs Average Amount of Credit")
        #plt.savefig("./images/" + "AVG_AGE_AMT_OF_CREDIT" + ".png")
        plt.show()
        st.pyplot(fig11)
        
    else:
        pass
        
except:
  print('Please enter client ID again')


## Lime and Shap plots

st.subheader('Lime and Shap plots')

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

X = train_df.drop(['Unnamed: 0', 'Unnamed: 0.1','TARGET'],axis=1)
y = train_df['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

#import lightgbm as lgb

#lgb = lgb.LGBMClassifier(class_weight='balanced',n_estimators=300,learning_rate=0.05)

# pipe_lb = Pipeline([
#     ('preprocess', preprocessor),('model',lgb)
# ])

# pipe_lb.fit(X_train,y_train)

model = model = pickle.load(open('./data/lgbmodel.pkl', 'rb'))

## Applying the LIME for LightGBM
pipe = Pipeline([('preprocessor', preprocessor)]) 
train_data = pipe.fit_transform(X_train)
test_data = pipe.fit_transform(X_test)

result_df = result[result['SK_ID_CURR']==id]
idx=result.index[0] # the rows of the dataset
    
lime_plt = st.button('LIME Plot')

if lime_plt:
    
    st.subheader('Lime Explanation Plot')

    class_names = [0, 1]
    #instantiate the explanations for the data set
    limeexplainer = LimeTabularExplainer(train_data, class_names=class_names, feature_names = X_train.columns, discretize_continuous = False)
    exp = limeexplainer.explain_instance(test_data[idx], model.predict_proba, num_features=10, labels=class_names)
    components.html(exp.as_html(), height=800)

shap_plt = st.button('SHAP Plot')

if shap_plt:
    
    st.subheader("Shap Explanation Plot") 

    sub_sampled_train_data = shap.sample(train_data, 1000, random_state=0) # use 1000 samples of train data as background data

    subsampled_test_data = test_data[idx].reshape(1,-1)

    # explain first sample from test data
    start_time = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(subsampled_test_data)
    elapsed_time = time.time() - start_time

    st.write(f"Tree Explainer SHAP run time for lightgbm is {round(elapsed_time,3)} in seconds")
    st.write(f"SHAP expected value: {[explainer.expected_value]}")
    st.write(f"Model mean value : {[model.predict_proba(train_data).mean(axis=0)]}")
    st.write(f"Model prediction for test data : {[model.predict_proba(subsampled_test_data)]}")
    
    st.write('Shap Summary Plot')
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values, subsampled_test_data, feature_names=X_train.columns, max_display=10)
    st.pyplot(fig)
