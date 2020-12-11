import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.cluster.hierarchy as shc
from sklearn import preprocessing, metrics, neighbors
from sklearn.cluster import AgglomerativeClustering
import gower

from data_clean import cleaned_data


# First specification with only continuous variables
unsup_df = cleaned_data[['age', 'hours_per_week', 'capital_loss', 'capital_gain']]

# Normalise the continuous variables
to_be_scaled = ['age', 'hours_per_week', 'capital_loss', 'capital_gain']
for x in to_be_scaled: 
	to_be_scaled = unsup_df.pop(x) 
	scaled = preprocessing.scale(to_be_scaled)
	unsup_df.insert(1,x,scaled)


# First specification clusters are specified varying the linkage method

model_ave = AgglomerativeClustering(n_clusters= 2, affinity = 'euclidean', linkage = 'average', compute_full_tree = False)
model_comp = AgglomerativeClustering(n_clusters= 2, affinity = 'euclidean', linkage = 'complete', compute_full_tree = False)
model_ward = AgglomerativeClustering(n_clusters= 2, affinity = 'euclidean', linkage = 'ward', compute_full_tree = False)

# Give each cluster specification the dataset
model_ave.fit(unsup_df)
model_comp.fit(unsup_df)
model_ward.fit(unsup_df)

# # Calculate and print performance measures
print("V Measure Average", metrics.homogeneity_completeness_v_measure(cleaned_data.income_over_50, model_ave.labels_))
print("V Measure Complete", metrics.homogeneity_completeness_v_measure(cleaned_data.income_over_50, model_comp.labels_))
print("V Measure Ward", metrics.homogeneity_completeness_v_measure(cleaned_data.income_over_50, model_ward.labels_))

# Produce Dendrograms
# Cuts dataset to speed up computational times
unsupd_df = unsup_df.head(100)
model_ave_den = shc.linkage(unsupd_df, 'average')
model_comp_den = shc.linkage(unsupd_df, 'complete')
model_ward_den = shc.linkage(unsupd_df, 'ward')
figure, axes = plt.subplots(3, figsize=(10, 7), sharex=True, sharey=False) 
figure.suptitle('Dendrograms')
axes0 = shc.dendrogram(model_ave_den, leaf_rotation = 90., leaf_font_size = 8.,ax= axes[0])
axes1 = shc.dendrogram(model_comp_den, leaf_rotation = 90., leaf_font_size = 8., ax= axes[1])
axes2 = shc.dendrogram(model_ward_den, leaf_rotation = 90., leaf_font_size = 8., ax= axes[2])
axes[0].set_title('Average')
axes[1].set_title('Complete')
axes[2].set_title('Ward')
axes[0].set_xlabel('Sample index')
axes[1].set_xlabel('Sample index')
axes[2].set_xlabel('Sample index')
axes[0].set_ylabel('Distance')
axes[1].set_ylabel('Distance')
axes[2].set_ylabel('Distance')
plt.tight_layout()
plt.show()



# Second specification with continuous variables and binary indicators
unsup1_df = cleaned_data[['age', 'education_num', 'White', 'Male', 'US_Native', 'hours_per_week', 'Married', 'occupation', 'capital_gain', 'capital_loss']]
to_be_scaled = ['age', 'hours_per_week', 'capital_loss', 'capital_gain']
for x in to_be_scaled: 
	to_be_scaled = unsup1_df.pop(x) 
	scaled = preprocessing.scale(to_be_scaled)
	unsup1_df.insert(1,x,scaled)

# Computes clusters in average linkage method
model1_ave = AgglomerativeClustering(n_clusters= 2, affinity = 'precomputed', linkage = 'average', compute_full_tree = False)
model1_comp = AgglomerativeClustering(n_clusters= 2, affinity = 'precomputed', linkage = 'complete', compute_full_tree = False)
X = gower.gower_matrix(unsup1_df)
model1_ave.fit_predict(X)
model1_comp.fit_predict(X)
print("V Measure average", metrics.homogeneity_completeness_v_measure(cleaned_data.income_over_50, model1_ave.labels_))
print("V Measure complete", metrics.homogeneity_completeness_v_measure(cleaned_data.income_over_50, model1_comp.labels_))

Creates Dendograms for smaller samples
unsup1d_df = unsup1_df.head(100)
X = gower.gower_matrix(unsup1d_df)
model1_ave_den = shc.linkage(X, 'average')
model1_comp_den = shc.linkage(X, 'complete')

figure, axes = plt.subplots(2, figsize=(10, 7), sharex=False, sharey=True) 
figure.suptitle('Dendrograms')
axes0 = shc.dendrogram(model1_ave_den, leaf_rotation = 90., leaf_font_size = 8.,ax= axes[0])
axes1 = shc.dendrogram(model1_comp_den, leaf_rotation = 90., leaf_font_size = 8., ax= axes[1])
axes[0].set_title('Average')
axes[1].set_title('Complete')
axes[0].set_xlabel('Sample index')
axes[1].set_xlabel('Sample index')
axes[0].set_ylabel('Distance')
axes[1].set_ylabel('Distance')
plt.tight_layout()
plt.show()
