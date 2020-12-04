import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# Loads in dataset and applies header names, extracted from adult.names file
headers = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'martial_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_over_50']
income_data = pd.read_csv('adult.csv')
income_data.columns = headers

# Replaces question mark entry with null values to allow removal of missing values
for col in headers: 
	income_data.loc[income_data[col] == '?', col] = None

# Cleans data set by setting binary variables for outcome var and sex var, also introduces a binary var for US native country
income_data['US_Native'] = 0
income_data['Male'] = 0
income_data['native_status_graph'] = 0

income_data.loc[income_data.income_over_50 == '<=50K', 'income_over_50'] = 'No'
income_data.loc[income_data.income_over_50 == '>50K', 'income_over_50'] = 'Yes'
income_data.loc[income_data.sex == 'Male', 'Male'] = 1
income_data.loc[income_data.sex == 'Female', 'Male'] = 0
income_data.loc[income_data.native_country == 'United-States', 'US_Native'] = 1
income_data.loc[income_data.US_Native == 1, 'native_status_graph'] = 'Yes'
income_data.loc[income_data.US_Native == 0, 'native_status_graph'] = 'No'

# Reports missing values as proportion of total data set

percent_missing = income_data.isnull().sum() * 100 / len(income_data)
missing_value_income_data = pd.DataFrame({'column_name': income_data.columns,'percent_missing': percent_missing})
print(missing_value_income_data)
cleaned_data = income_data.dropna()











