import matplotlib.pyplot as plt
import seaborn as sns

from data_clean import cleaned_data


headers = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'martial_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'inc_over_50']
cat_vars = ['education', 'martial_status', 'occupation', 'race', 'sex', 'native_country']
num_vars = ['age', 'workclass', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

def label_function(val):
    return f'{val / 100 * len(cleaned_data):.0f}\n{val:.0f}%'


fig, axes = plt.subplots(2, 2, figsize = (8,8))
ax1 = sns.boxplot(ax=axes[0,0], x = 'income_over_50_graph', y="education_num", hue="income_over_50_graph", notch = True, data=cleaned_data, palette=["m", "g"]).set(
    xlabel='Income Over 50k', 
    ylabel='Education Number')
ax2 = sns.boxplot(ax=axes[1,0], x = 'income_over_50_graph', y="hours_per_week", hue="income_over_50_graph", data=cleaned_data, palette=["m", "g"]).set(
    xlabel='Income Over 50k', 
    ylabel='Hours per Week')

cleaned_data.groupby('race').size().plot(kind='pie', autopct='%1.0f%%', textprops={'fontsize': 10}, ax=axes[0,1])
cleaned_data.groupby('native_status_graph').size().plot(kind='pie', autopct=label_function, textprops={'fontsize': 10},
                                 colors=['violet', 'green'], ax=axes[1,1])
axes[0,1].set_ylabel('Race', size=18)
axes[1,1].set_ylabel('US Native Country', size=18)

fig1, axes1 = plt.subplots(2, 3,figsize=(10, 8))
ax3 = sns.countplot(ax = axes1[0,0], x = 'martial_status', hue = 'income_over_50_graph', data = cleaned_data).set(
    xlabel='Martial Status')
axes1[0,0].set_xticklabels(axes1[0,0].get_xticklabels(), rotation=40, ha="right")
axes1[0,0].legend(loc='best')

ax4 = sns.countplot(ax = axes1[0,1], x = 'native_status_graph', hue = 'income_over_50_graph', data = cleaned_data).set(
    xlabel='US Native')
axes1[0,1].legend(loc='best')

ax5 = sns.countplot(ax = axes1[1,0], y = 'education', hue = 'income_over_50_graph', data = cleaned_data).set(
    ylabel='Education Level')
axes1[1,0].legend(loc='best')
axes1[1,0].set_xticklabels(axes1[1,0].get_xticklabels(), rotation=90, ha="right")

ax6 = sns.countplot(ax = axes1[1,1], x = 'race', hue = 'income_over_50_graph', data = cleaned_data).set(
    xlabel='Race')
axes1[1,1].legend(loc='best')
axes1[1,1].set_xticklabels(axes1[1,1].get_xticklabels(), rotation=90, ha="right")

ax7 = sns.countplot(ax = axes1[0,2], x = 'occupation', hue = 'income_over_50_graph', data = cleaned_data).set(
    xlabel='Occupation')
axes1[0,2].set_xticklabels(axes1[0,2].get_xticklabels(), rotation=90, ha="right")
axes1[0,2].legend(loc='best')

ax8 = sns.violinplot(ax = axes1[1,2], y='income_over_50_graph', x = 'age', hue="income_over_50_graph", data = cleaned_data, legend_out = False).set(ylabel='Income over 50')
axes1[1,2].legend(loc='best')
fig2, axes2 = plt.subplots(2,2, figsize = (14,6))
ax9 = sns.scatterplot(y = 'workclass' , x= 'race', data = cleaned_data, hue = 'income_over_50', palette = 'flare', alpha = 0.5, ax = axes2[0,0], legend = False, s = 200).set(xlabel='Race', ylabel= 'Work Classification')
ax10 = sns.scatterplot(y = 'education_num' , x= 'race', data = cleaned_data, hue = 'income_over_50', palette = 'flare', alpha = 0.5, ax = axes2[0,1], legend = False, s = 200).set(xlabel='Race', ylabel= 'Education Classification')
ax11 = sns.scatterplot(y = 'occupation' , x= 'race', data = cleaned_data, hue = 'income_over_50', palette = 'flare', alpha = 0.5, ax = axes2[1,0], legend = False, s = 200).set(xlabel='Race', ylabel = 'Occupation Classification')
ax12 = sns.scatterplot(y = 'hours_per_week' , x= 'race', data = cleaned_data, hue = 'income_over_50', palette = 'flare', alpha = 0.5, ax = axes2[1,1], legend = True, s = 200).set(xlabel='Race', ylabel = 'Hours per Week')

fig3, axes3 = plt.subplots(2)

plt.tight_layout()
plt.show()

# legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)