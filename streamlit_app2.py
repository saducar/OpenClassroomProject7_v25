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
  'ANNUITY_INCOME_PERCENT','CREDIT_TERM','AMT_APPLICATION_PERCENT','FLAG_OWN_REALTY','AMT_INCOME_TOTAL','NAME_EDUCATION_TYPE',
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
  'ANNUITY_INCOME_PERCENT','CREDIT_TERM','AMT_APPLICATION_PERCENT','FLAG_OWN_REALTY','AMT_INCOME_TOTAL','NAME_EDUCATION_TYPE',
  'DAYS_EMPLOYED','AMT_GOODS_PRICE_PERCENT'))
variable2 = st.selectbox(
  'Choose Variable for Y-axis',
  ('DAYS_BIRTH', 'AMT_ANNUITY_x','EXT_SOURCE_2', 'EXT_SOURCE_3', 'CREDIT_INCOME_PERCENT', 
  'ANNUITY_INCOME_PERCENT','CREDIT_TERM','AMT_APPLICATION_PERCENT','FLAG_OWN_REALTY','AMT_INCOME_TOTAL','NAME_EDUCATION_TYPE',
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
