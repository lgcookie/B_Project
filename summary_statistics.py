import seaborn as sns

from data_clean import cleaned_data

cat_vars = ['education', 'martial_status', 'occupation', 'race', 'sex', 'native_country']
num_vars = ['age', 'workclass', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

for var in num_vars: 
	print(cleaned_data[var].describe())
	
for cat in cat_vars: 
	print(cleaned_data[cat].value_counts())