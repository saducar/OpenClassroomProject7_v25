import streamlit as st 
import pandas as pd
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)

# Webapp title
st.title('Home Credit Default Risk')

# loading the data

@st.cache(allow_output_mutation=True)
def load_data(file):
    df = pd.read_csv(f'./data/{file}.csv')
    return df

train_df = load_data('train_data')

st.subheader('Distribution of Variables')
variables = st.selectbox(
  'Choose Variable for Density Plot',
  ('DAYS_BIRTH', 'AMT_ANNUITY_x','EXT_SOURCE_2', 'EXT_SOURCE_3', 'CREDIT_INCOME_PERCENT', 
  'ANNUITY_INCOME_PERCENT','AMT_APPLICATION_PERCENT','FLAG_OWN_REALTY','AMT_INCOME_TOTAL','NAME_EDUCATION_TYPE',
  'DAYS_EMPLOYED','AMT_GOODS_PRICE_PERCENT'))

fig1 = plt.figure(figsize=(10,8))
train_df[variables]=train_df[variables].abs()
sns.kdeplot(train_df.loc[train_df['TARGET']==0,variables]/365,label='target==0')
sns.kdeplot(train_df.loc[train_df['TARGET']==1,variables]/365,label='target==1')
plt.legend()
plt.xlabel(variables)
plt.ylabel('Density')
plt.title(f'Distribution of {variables} by target value')
st.pyplot(fig1)

st.subheader('Scatter plots between Variables')
variable1 = st.selectbox(
  'Choose Variable for X-axis',
  ('DAYS_BIRTH', 'AMT_ANNUITY_x','EXT_SOURCE_2', 'EXT_SOURCE_3', 'CREDIT_INCOME_PERCENT', 
  'ANNUITY_INCOME_PERCENT','AMT_APPLICATION_PERCENT','FLAG_OWN_REALTY','AMT_INCOME_TOTAL','NAME_EDUCATION_TYPE',
  'DAYS_EMPLOYED','AMT_GOODS_PRICE_PERCENT'))
variable2 = st.selectbox(
  'Choose Variable for Y-axis',
  ('DAYS_BIRTH', 'AMT_ANNUITY_x','EXT_SOURCE_2', 'EXT_SOURCE_3', 'CREDIT_INCOME_PERCENT', 
  'ANNUITY_INCOME_PERCENT','AMT_APPLICATION_PERCENT','FLAG_OWN_REALTY','AMT_INCOME_TOTAL','NAME_EDUCATION_TYPE',
  'DAYS_EMPLOYED','AMT_GOODS_PRICE_PERCENT'))

fig2 = plt.figure(figsize=(10,8))
sns.scatterplot(data=train_df, x=variable1, y=variable2, hue="TARGET")
plt.legend()
plt.xlabel(variable1)
plt.ylabel(variable2)
plt.title(f'Scatter plot between the {variable1} and {variable2} by target value')
st.pyplot(fig2)

st.subheader('Correlation Heatmaps')
dataset = st.selectbox(
  'Choose data for Correlation Heatmap',
  ('Credit_Card_Balance', 'Installments_Payments', 'Previous_Application'))

fig3 = plt.figure(figsize=(20,15))
df = load_data(dataset)
corr = df.corr().abs()
sns.heatmap(corr, cmap=plt.cm.RdYlBu_r, vmin=-0.25, annot=True, vmax=0.6)
plt.title('Correlation Heatmap')
#plt.savefig("./images/"+dataset+".png")
st.pyplot(fig3)

st.subheader('Predicting the credit is default or not')

# Utility function
def yes_no(value):
    if value=="Yes":
       return 1
    else:
       return 0

st.write('Choose the feature values')

own_car = st.selectbox(
  'Choose whether client owned Car or not',
  ("Yes", "No"))
own_car = yes_no(own_car)

own_realty = st.selectbox(
  'Choose whether client owned Realty or not',
  ("Yes", "No"))
own_realty = yes_no(own_realty)

days_birth = st.number_input('Enter the Days of Birth:')

days_employed = st.number_input('Enter the Days of Employed:')

income_amt = st.number_input('Enter the Income Amount:')

credit_amt = st.number_input('Enter the Credit Amount:')

# Create a empty dataframe
df = pd.DataFrame()
# These values comes from web app
df['FLAG_OWN_CAR'] = [own_car]
df['FLAG_OWN_REALTY'] = [own_realty]
df['DAYS_BIRTH'] = [days_birth]
df['DAYS_EMPLOYED'] = [days_employed]
df['AMT_INCOME_TOTAL'] = [income_amt]
df['AMT_CREDIT'] = [credit_amt]

# These values are calculated by taking majority votes
# data['feature'].value_counts().argmax()

df['NAME_CONTRACT_TYPE_x'] = 0
df['CODE_GENDER'] = 0
df['CNT_CHILDREN'] = 0
df['NAME_TYPE_SUITE'] = 0
df['NAME_INCOME_TYPE'] = 0
df['NAME_FAMILY_STATUS'] = 0
df['NAME_HOUSING_TYPE'] = 0
df['FLAG_MOBIL'] = 0
df['FLAG_WORK_PHONE'] = 0
df['FLAG_CONT_MOBILE'] = 0
df['FLAG_PHONE'] = 0
df['FLAG_EMAIL'] = 0
df['REGION_RATING_CLIENT'] = 0
df['HOUR_APPR_PROCESS_START_x'] = 0
df['REG_REGION_NOT_WORK_REGION'] = 0
df['LIVE_REGION_NOT_WORK_REGION'] = 0
df['REG_CITY_NOT_WORK_CITY'] = 0
df['FLAG_DOCUMENT_2'] = 0
df['FLAG_DOCUMENT_4'] = 0
df['FLAG_DOCUMENT_8'] = 0
df['FLAG_DOCUMENT_9'] = 0
df['FLAG_DOCUMENT_10'] = 0
df['FLAG_DOCUMENT_11'] = 0
df['FLAG_DOCUMENT_12'] = 0
df['FLAG_DOCUMENT_13'] = 0
df['FLAG_DOCUMENT_14'] = 0
df['FLAG_DOCUMENT_15'] = 0
df['FLAG_DOCUMENT_16'] = 0
df['FLAG_DOCUMENT_17'] = 0
df['FLAG_DOCUMENT_18'] = 0
df['FLAG_DOCUMENT_19'] = 0
df['FLAG_DOCUMENT_20'] = 0
df['FLAG_DOCUMENT_21'] = 0

# These values are calculated by taking average
# data['feature'].mean()
df['AMT_ANNUITY_x'] = 29411.8
df['REGION_POPULATION_RELATIVE'] = 0.021226
df['DAYS_REGISTRATION'] = -4967.7
df['DAYS_ID_PUBLISH'] = 3051
df['CNT_FAM_MEMBERS'] = 2.1
df['EXT_SOURCE_3'] = 0.411173
df['DEF_30_CNT_SOCIAL_CIRCLE'] = 0.1
df['DEF_60_CNT_SOCIAL_CIRCLE'] = 0.1
df['DAYS_LAST_PHONE_CHANGE'] = -1077.8
df['AMT_REQ_CREDIT_BUREAU_HOUR'] = 0.0
df['AMT_REQ_CREDIT_BUREAU_DAY'] = 0.0
df['AMT_REQ_CREDIT_BUREAU_WEEK'] = 0.0
df['AMT_REQ_CREDIT_BUREAU_MON'] = 0.0
df['AMT_REQ_CREDIT_BUREAU_QRT'] = 0.5
df['AMT_REQ_CREDIT_BUREAU_YEAR'] = 1.7
df['NAME_CONTRACT_TYPE_y'] = 0.76
df['AMT_ANNUITY_y'] = 15111.41113
df['AMT_APPLICATION'] = 154240.1683
df['AMT_GOODS_PRICE'] = 200458.1548
df['FLAG_LAST_APPL_PER_CONTRACT'] = 1.0
df['NFLAG_LAST_APPL_IN_DAY'] = 1.0
df['NAME_CASH_LOAN_PURPOSE'] = 22.64
df['NAME_CONTRACT_STATUS'] =0.416
df['DAYS_DECISION'] = -899.8
df['NAME_PAYMENT_TYPE'] = 1.02
df['CODE_REJECT_REASON'] = 6.19
df['NAME_CLIENT_TYPE'] = 1.22
df['NAME_GOODS_CATEGORY'] = 17.66
df['NAME_PORTFOLIO'] = 2.67
df['NAME_PRODUCT_TYPE'] = 0.48
df['SELLERPLACE_AREA'] = 404.48
df['CNT_PAYMENT'] = 14.267221
df['NAME_YIELD_GROUP'] = 1.97
df['AMT_CREDIT_PERCENT'] = 3.167544
df['AMT_APPLICATION_PERCENT'] = 5.477633
df['AMT_GOODS_PRICE_PERCENT'] = 4.367816

st.write(df)

df_json = df.to_json(orient = "records")
url = 'https://apipredictions.herokuapp.com/predict'
r = requests.post(url, json=json.loads(df_json))
r = r.json()[0]
pred = r['pred_proba']
pred = np.round(pred,3)

submit = st.button('Predict')

if submit:

  if pred>=0.5:
    st.write(f'Prediction: {pred}')
    st.write('Home Credit is Default')
  else:
    st.write(f'Prediction: {pred}')
    st.write('Home Credit is not Default')
