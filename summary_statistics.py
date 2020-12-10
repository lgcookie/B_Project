import seaborn as sns
import pandas as pd
from data_clean import cleaned_data
from scipy import stats

cat_vars = ['education', 'martial_status', 'occupation', 'race', 'sex', 'native_country', 'workclass']
num_vars = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week', 'income_over_50']

cleaned_data.describe().to_csv("filename.csv")

for cat in cat_vars: 
	print(cleaned_data[cat].value_counts())


corr_matrix = cleaned_data[['capital_loss', 'capital_gain', 'fnlwgt', 'income_over_50']]

for cat in corr_matrix.columns[:-1]:
	print("----------------")
	print(f"\nVariable: {cat}")
	print(stats.pointbiserialr(corr_matrix[cat], corr_matrix.income_over_50))
	
