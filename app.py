import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Opens data set documentation
# p = open('adult.names', 'r')
# print(p.read())

# Loads in dataset and applies header names, extracted from adult.names file
headers = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'martial_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'inc_over_50']
income_data = pd.read_csv('adult.csv')
income_data.columns = headers

# Replaces question mark entry with null values to allow removal of missing values
for col in headers: 
	income_data.loc[income_data[col] == '?', col] = None

# Cleans data set by setting binary variables for outcome var and sex var, also introduces a binary var for US native country
income_data['US_Native'] = 0
income_data.loc[income_data.inc_over_50 == '<=50K', 'inc_over_50'] = 0
income_data.loc[income_data.inc_over_50 == '>50K', 'inc_over_50'] = 1
income_data.loc[income_data.sex == 'Male', 'sex'] = 1
income_data.loc[income_data.sex == 'Female', 'sex'] = 0
income_data.loc[income_data.native_country == 'United-States', 'US_Native'] = 1

# Reports missing values as proportion of total data set
print(income_data.isnull().sum()/len(income_data))
cleaned_data = income_data.dropna()

print(cleaned_data.occupation.value_counts())

# for col in headers: 
# 	print(income_data[col].describe())
# pd.plotting.scatter_matrix(income_data)

# print(income_data)
# for line in income_data: 
# 	print(line)
# if income_data['class'] == '>50K': 
# 	income_data['class'] == 1
# else: 
# 	income_data['class'] == 0






