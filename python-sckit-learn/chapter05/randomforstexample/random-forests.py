#Import Libraries and Data
#Import Libraries and Data
#Import Libraries and Data
#Import Libraries and Data

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
import pandas as pd
from ggplot import *

number = preprocessing.LabelEncoder()

train=pd.read_csv('data/Train.csv')
test=pd.read_csv('data/Test.csv')
train.head()

#Convert Categorical data to number
#Convert Categorical data to number
#Convert Categorical data to number
#Convert Categorical data to number

def convert(data):
    number = preprocessing.LabelEncoder()
    data['Gender'] = number.fit_transform(data.Gender)
    data['City'] = number.fit_transform(data.City)
    data['Salary_Account'] = number.fit_transform(data.Salary_Account)
    data['Employer_Name'] = number.fit_transform(data.Employer_Name)
    data['Mobile_Verified'] = number.fit_transform(data.Mobile_Verified)
    data['Var1'] = number.fit_transform(data.Var1)
    data['Filled_Form'] = number.fit_transform(data.Filled_Form)
    data['Device_Type'] = number.fit_transform(data.Device_Type)
    data['Var2'] = number.fit_transform(data.Var2)
    data['Source'] = number.fit_transform(data.Source)
    data=data.fillna(0)
    return data

train=convert(train)
test=convert(test)

#Divide Train into train and Validate
#Divide Train into train and Validate
#Divide Train into train and Validate
#Divide Train into train and Validate

train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75
train, validate = train[train['is_train']==True], train[train['is_train']==False]


#select Inuput & Target Feature
#select Inuput & Target Feature
#select Inuput & Target Feature
#select Inuput & Target Feature

features=['Gender',
'City',
'Monthly_Income',
'Loan_Amount_Applied',
'Loan_Tenure_Applied',
'Existing_EMI',
'Employer_Name',
'Salary_Account',
'Mobile_Verified',
'Var5',
'Var1',
'Loan_Amount_Submitted',
'Loan_Tenure_Submitted',
'Interest_Rate',
'Processing_Fee',
'EMI_Loan_Submitted',
'Filled_Form',
'Device_Type',
'Var2',
'Source',
'Var4']
x_train = train[list(features)].values
x_validate=validate[list(features)].values
y_train = train['Disbursed'].values
y_validate = validate['Disbursed'].values
x_test=test[list(features)].values


#Build Random Forest
#Build Random Forest
#Build Random Forest
#Build Random Forest

rf = RandomForestClassifier(n_estimators=1000)
rf.fit(x_train, y_train)


#Look at Important Feature
#Look at Important Feature
#Look at Important Feature
#Look at Important Feature

importances = rf.feature_importances_
indices = np.argsort(importances)

ind=[]
for i in indices:
    ind.append(features[i])

import matplotlib.pyplot as plt

plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)),ind)
plt.xlabel('Relative Importance')
plt.show()

#Plot ROC_AUC curve and cross validate
#Plot ROC_AUC curve and cross validate
#Plot ROC_AUC curve and cross validate
#Plot ROC_AUC curve and cross validate

disbursed = rf.predict_proba(x_validate)
fpr, tpr, _ = roc_curve(y_validate, disbursed[:,1])
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
ggplot(df, aes(x='fpr', y='tpr')) + geom_line() + geom_abline(linetype='dashed')

roc_auc = auc(fpr, tpr)
print roc_auc

#Predict for test data set and export test data set 
#Predict for test data set and export test data set
#Predict for test data set and export test data set
#Predict for test data set and export test data set

disbursed = rf.predict_proba(x_test)
test['Disbursed']=disbursed[:,1]
test.to_csv('c:/sunila/practical-machine-learning/ch-05/random-forests/Solution.csv', columns=['ID','Disbursed'],index=False)