import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from data_clean import cleaned_data

# Demonstrates that martial status should be segregated to married vs not married
table=pd.crosstab(cleaned_data.martial_status, cleaned_data.income_over_50)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Income over 50K')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Individuals')
plt.savefig('mariral_vs_inc_stack')
plt.tight_layout()
plt.show()

# Checks for correlation between continuous variables
correlation_matrix = cleaned_data.drop(['income_over_50', 'Married', 'US_Native', 'White', 'Male'], axis=1).corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots()

# Generates a diverging colour map
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draws the heatmap and displays the output
sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Heat Map Showing Correlation Between\n Continuous Variables')
plt.tight_layout()
plt.show()
