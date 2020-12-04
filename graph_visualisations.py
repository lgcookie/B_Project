import matplotlib.pyplot as plt
import seaborn as sns

from data_clean import cleaned_data


headers = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'martial_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'inc_over_50']
cat_vars = ['education', 'martial_status', 'occupation', 'race', 'sex', 'native_country']
num_vars = ['age', 'workclass', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

def label_function(val):
    return f'{val / 100 * len(cleaned_data):.0f}\n{val:.0f}%'

# Grouped boxplot
fig, axes = plt.subplots(2, 2, figsize = (8,8))
ax1 = sns.boxplot(ax=axes[0,0], x = 'income_over_50', y="education_num", hue="income_over_50", notch = True, data=cleaned_data, palette=["m", "g"]).set(
    xlabel='Income Over 50k', 
    ylabel='Education Number')
ax2 = sns.boxplot(ax=axes[1,0], x = 'income_over_50', y="hours_per_week", hue="income_over_50", data=cleaned_data, palette=["m", "g"]).set(
    xlabel='Income Over 50k', 
    ylabel='Hours per Week')

cleaned_data.groupby('race').size().plot(kind='pie', autopct='%1.0f%%', textprops={'fontsize': 10}, ax=axes[0,1])
cleaned_data.groupby('native_status_graph').size().plot(kind='pie', autopct=label_function, textprops={'fontsize': 10},
                                 colors=['violet', 'lime'], ax=axes[1,1])
axes[0,1].set_ylabel('Race', size=10)
axes[1,1].set_ylabel('US Native Country', size=10)

fig1, axes1 = plt.subplots(2, 3,figsize=(10, 8))
ax3 = sns.countplot(ax = axes1[0,0], x = 'martial_status', hue = 'income_over_50', data = cleaned_data).set(
    xlabel='Martial Status')

axes1[0,0].set_xticklabels(axes1[0,0].get_xticklabels(), rotation=40, ha="right")
ax4 = sns.countplot(ax = axes1[0,1], x = 'native_status_graph', hue = 'income_over_50', data = cleaned_data).set(
    xlabel='US Native')
ax5 = sns.countplot(ax = axes1[1,0], y = 'education', hue = 'income_over_50', data = cleaned_data).set(
    ylabel='Education Level')
axes1[1,0].set_xticklabels(axes1[1,0].get_xticklabels(), rotation=90, ha="right")
ax6 = sns.countplot(ax = axes1[1,1], x = 'race', hue = 'income_over_50', data = cleaned_data).set(
    xlabel='Race')
axes1[1,1].set_xticklabels(axes1[1,1].get_xticklabels(), rotation=90, ha="right")
ax7 = sns.countplot(ax = axes1[0,2], x = 'occupation', hue = 'income_over_50', data = cleaned_data).set(
    xlabel='Occupation')
axes1[0,2].set_xticklabels(axes1[0,2].get_xticklabels(), rotation=90, ha="right")

ax8 = sns.violinplot(ax = axes1[1,2], x='income_over_50', y = 'age', hue="income_over_50", data = cleaned_data)


plt.tight_layout()
plt.show()