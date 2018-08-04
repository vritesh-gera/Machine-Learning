import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Importing dataset
dataset_normal=pd.read_csv('Risk stratification - Normal Category.csv')
dataset_prehypertension=pd.read_csv('Risk stratification - Pre-Hypertension Category.csv')
dataset_stage1=pd.read_csv('Risk stratification - Stage 1 Category.csv')
dataset_stage2=pd.read_csv('Risk stratification - Stage 2 Category.csv')
dataset_stage3=pd.read_csv('Risk stratification - Stage 3 Category.csv')
#Creating the master data set
master_dataset=pd.concat([dataset_normal,dataset_prehypertension,dataset_stage1,dataset_stage2,dataset_stage3],axis=0,sort=False)
#Adjusting the indexes
master_dataset.reset_index(drop=True,inplace=True)

#Dropping the extra columns
master_dataset.drop(master_dataset.columns[15],axis=1,inplace=True)
master_dataset.drop(master_dataset.columns[0],axis=1,inplace=True)
master_dataset.drop(master_dataset.columns[2],axis=1,inplace=True)
#Creating matrix of features and dependant variable vector
X=master_dataset.iloc[:,:-1].values
y=master_dataset.iloc[:,12].values
#As you cannot view X in the current version as it is an object of mixed types
X_viewable=pd.DataFrame(X)
#Data Analysis
Paras=master_dataset.describe()
master_dataset['Systolic(mmHg)'].hist(bins=200)
master_dataset['Gender'].hist(bins=200)
master_dataset['Age'].hist(bins=200)
master_dataset['Diastolic(mmHg)'].hist(bins=200)
master_dataset['BMI(lbs/in2)'].hist(bins=200)
master_dataset['High Density Lipo(mg/dL)'].hist(bins=200)
master_dataset['Low Density Lipo(mg/dL)'].hist(bins=200)
master_dataset['Triglycerides(mmol/L)'].hist(bins=200)
master_dataset['Cholestrol(mg/dL)'].hist(bins=200)
master_dataset['Micro Albumin(mg)'].hist(bins=200)
master_dataset['Urin Albumin(mg/g)'].hist(bins=200)
master_dataset['smoking Status'].hist(bins=200)
dataset_normal['smoking Status'].value_counts().plot(kind='bar')
dataset_prehypertension['smoking Status'].value_counts().plot(kind='bar')
dataset_stage1['smoking Status'].value_counts().plot(kind='bar')
dataset_stage2['smoking Status'].value_counts().plot(kind='bar')
dataset_stage3['smoking Status'].value_counts().plot(kind='bar')

#Encoding Categorical data
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencoder_X=LabelEncoder()
X[:, 0]= labelencoder_X.fit_transform(X[:,0])

labelencoder_smoke=LabelEncoder()
X[:, 11]= labelencoder_smoke.fit_transform(X[:,11])


