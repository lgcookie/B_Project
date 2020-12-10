import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# Loads in dataset and applies header names, extracted from adult.names file
headers = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'martial_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_over_50']
income_data = pd.read_csv('adult.csv')
income_data.columns = headers
# print(income_data.income_over_50.describe())
# Reads info
# with open('adult.names', 'r') as p:
# 	print(p.read())
# Replaces question mark entry with null values to allow removal of missing values
for col in headers: 
	income_data.loc[income_data[col] == '?', col] = None
# print(income_data.education_num.unique())
# print(income_data.education.unique())
# Cleans data set by setting binary variables for outcome var and sex var, also introduces a binary var for US native country
income_data['US_Native'] = 0
income_data['Male'] = 0
income_data['income_over_50'] = income_data['income_over_50']
income_data['native_status_graph'] = 0
income_data['income_over_50_graph'] = income_data['income_over_50']
income_data['White'] = 0
income_data['Married'] = 0


income_data.loc[income_data.sex == 'Male', 'Male'] = 1
income_data.loc[income_data.sex == 'Female', 'Male'] = 0
income_data.loc[income_data.native_country == 'United-States', 'US_Native'] = 1
income_data.loc[income_data.US_Native == 1, 'native_status_graph'] = 'Yes'
income_data.loc[income_data.US_Native == 0, 'native_status_graph'] = 'No'
income_data.loc[income_data.income_over_50 == '<=50K', 'income_over_50'] = 0
income_data.loc[income_data.income_over_50 == '>50K', 'income_over_50'] = 1
income_data.loc[income_data.race == 'White', 'White'] = 1
income_data.loc[income_data.martial_status == 'Married-civ-spouse', 'Married'] = 1

# Reports missing values as proportion of total data set

percent_missing = income_data.isnull().sum() * 100 / len(income_data)
missing_value_income_data = pd.DataFrame({'column_name': income_data.columns,'percent_missing': percent_missing})
# print(missing_value_income_data)
cleaned_data = income_data.dropna()
cleaned_data = cleaned_data.drop('fnlwgt', axis =1)
# print(cleaned_data.income_over_50.describe())

# # Produces histograms for capital gain and capital loss
# fig, axes = plt.subplots(1,2)

# axes[0] = sns.histplot(data = cleaned_data, x = 'capital_loss', bins='auto', ax=axes[0], color='#0504aa', hue = 'income_over_50_graph')
# axes[1] = sns.histplot(data = cleaned_data, x = 'capital_gain', bins='auto', ax=axes[1], color='#0504aa', hue = 'income_over_50_graph')
# # cleaned_data.hist('capital_gain', bins='auto', ax=axes[1], color='#0504aa', grid = True)
# axes[0].set_title('Capital Loss Distribution')
# axes[1].set_title('Capital Gain Distribution')
# axes[0].set_ylabel('Count')
# axes[1].set_ylabel('Count')
# axes[0].set_xlabel('Value ($)')
# axes[1].set_xlabel('Value ($)')
# plt.tight_layout()
# plt.show()








