import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import seaborn as sns
from data_clean import cleaned_data
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# One-hot encodes workclass and occupation
workclass_dummies = pd.get_dummies(cleaned_data['workclass'])
occupation_dummies = pd.get_dummies(cleaned_data['occupation'])

# Merges the resulting dataframes
cleaned_vars = cleaned_data[['age','education_num', 'Married', 'White', 'Male', 'hours_per_week', 'US_Native', 'capital_gain', 'capital_loss', 'income_over_50']]
frames = [workclass_dummies, occupation_dummies, cleaned_vars]
reg_data = pd.concat(frames, axis =1)

# Scales continuous variables
to_be_scaled = ['age', 'hours_per_week', 'capital_gain', 'capital_loss']
for x in to_be_scaled: 
	to_be_scaled = reg_data.pop(x) 
	scaled = preprocessing.scale(to_be_scaled)
	reg_data.insert(1,x,scaled)

# # Seperates the dependent and independent variables
X = reg_data.loc[:, reg_data.columns != 'income_over_50']
y = reg_data.loc[:, reg_data.columns == 'income_over_50']
y=y.astype('int')

# Splits into training and test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify = y)

# Runs logistical model and presents results
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary2())

# Preforms specification with all variables
logreg = LogisticRegression(max_iter = 4000)
logreg.fit(X_train, y_train.values.ravel())

# Uses test sample to predict
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# Produces confusion matrix and classification report
confusion_matrix1 = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(confusion_matrix1)


########### RFE Specification ######################


logreg = LogisticRegression(max_iter = 4000)
rfe = RFE(logreg, n_features_to_select=7)
rfe = rfe.fit(X_train, y_train.values.ravel())

# Identifies variables that have highest predictive power
vars_status = rfe.support_

# Creates list containing the most relevant features
vars_to_inc = list()
reg_final_vars=X.columns.values.tolist()
for i in range(len(reg_final_vars)): 
	if vars_status[i] == True: 
		vars_to_inc.append(reg_final_vars[i])

# Extracts the relevant variables from dataset
X=X_train[vars_to_inc]
X_test = X_test[vars_to_inc]


# Preforms specification with most relevant features
logit_model=sm.Logit(y_train,X)
result=logit_model.fit()
print(result.summary2())


### Remove private service as insignificant, reruns the regression
X_final_train = X.drop('Priv-house-serv', axis =1)
X_final_test = X_test.drop('Priv-house-serv', axis =1)
logit_model=sm.Logit(y_train,X_final_train)
result=logit_model.fit()
print(result.summary2())

# Preforms accuracy tests on final specification and prints confusion matrix
logreg = LogisticRegression(max_iter = 4000)
logreg.fit(X_final_train, y_train.values.ravel())
y_pred = logreg.predict(X_final_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_final_test, y_test)))
confusion_matrix2 = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(confusion_matrix2)

### Constructs ROC curve
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = logreg.predict_proba(X_final_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Model')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Model')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
# show the legend
plt.legend()
# show the plot
plt.show()

results_text = result.summary2().as_text()


