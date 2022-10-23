import pandas as pd
data = pd.read_csv("train_data_domain.csv")

columns = ['DAYS_BIRTH', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_EMPLOYED', 'CREDIT_INCOME_PERCENT',
           'ANNUITY_INCOME_PERCENT','CREDIT_TERM','DAYS_EMPLOYED_PERCENT', "TARGET"]
data_filtered = data[columns]
data_filtered.to_csv("train_data_domain_filtered.csv")


columns = ['Class', 'FLAG_OWN_REALTY', 'SK_ID_CURR', 'DAYS_BIRTH', 'AMT_CREDIT_PERCENT', 'AMT_APPLICATION', 'DAYS_BIRTH', 'AMT_GOODS_PRICE', 'AMT_ANNUITY_x', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', "TARGET"]
data = pd.read_csv("result.csv")
data_filtered = data[columns]
data_filtered.to_csv("results_filtered.csv")
