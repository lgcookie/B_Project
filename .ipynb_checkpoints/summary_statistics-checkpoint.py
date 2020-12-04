from data_clean import cleaned_data


headers = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'martial_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'inc_over_50']
cat_vars = ['education', 'martial_status', 'occupation', 'race', 'sex', 'native_country']
num_vars = ['age', 'workclass', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

print(cleaned_data.describe(include = 'all'))

for var in num_vars: 
	print(cleaned_data[var].describe())

for cat in cat_vars: 
	print(cleaned_data[cat].value_counts())