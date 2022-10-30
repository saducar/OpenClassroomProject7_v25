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

result = load_data('results')
train_df = load_data('train_data')

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

# Creating column transformer to select all column names that has Binary & a list for non Binary.
data = load_data('results')
X = data.drop(['Unnamed: 0','Class', 'TARGET', 'Age(years)', 'Age_cat'],axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.array(X_train)
X_test = np.array(X_test)

model = pickle.load(open('./data/lgbmodel.pkl', 'rb'))

## Applying the LIME for LightGBM

lime_plt = st.button('LIME Plot')

if lime_plt:
    
    st.subheader('Lime Explanation Plot')

    class_names = [0, 1]
    #instantiate the explanations for the data set
    limeexplainer = LimeTabularExplainer(X_train, class_names=class_names, feature_names = X_train.columns, discretize_continuous = False)
    X_df = result[result['SK_ID_CURR']==id]
    idx = X_df.index[0] # the rows of the dataset
    exp = limeexplainer.explain_instance(X_test[idx], model.predict_proba, num_features=10, labels=class_names)
    components.html(exp.as_html(), height=800)

shap_plt = st.button('SHAP Plot')

if shap_plt:
    
    st.subheader("Shap Explanation Plot") 

    sub_sampled_train_data = shap.sample(X_train, 1000, random_state=42) # use 1000 samples of train data as background data
    X_df = result[result['SK_ID_CURR']==id]
    idx = X_df.index[0] # the rows of the dataset
    subsampled_test_data = X_test[idx].reshape(1,-1)

    # explain first sample from test data
    start_time = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(subsampled_test_data)
    elapsed_time = time.time() - start_time

    st.write(f"Tree Explainer SHAP run time for lightgbm is {round(elapsed_time,3)} in seconds")
    st.write(f"SHAP expected value: {[explainer.expected_value]}")
    st.write(f"Model mean value : {[model.predict_proba(X_train).mean(axis=0)]}")
    st.write(f"Model prediction for test data : {[model.predict_proba(subsampled_test_data)]}")
    
    st.write('Shap Summary Plot')
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap_values, subsampled_test_data, feature_names=X_train.columns, max_display=10)
    st.pyplot(fig)
