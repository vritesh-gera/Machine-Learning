import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Importing dataset
dataset_normal=pd.read_csv('Risk stratification - Normal Category.csv')
dataset_prehypertension=pd.read_csv('Risk stratification - Pre-Hypertension Category.csv')
dataset_stage1=pd.read_csv('Risk stratification - Stage 1 Category.csv')
dataset_stage2=pd.read_csv('Risk stratification - Stage 2 Category.csv')
dataset_stage3=pd.read_csv('Risk stratification - Stage 3 Category.csv')
mydata=pd.read_csv('mydata.csv')
#Rearranging the dataset
from sklearn.cross_validation import train_test_split
normal_train, normal_test, prehypertension_train, prehypertension_test,stage1_train, stage1_test,stage2_train, stage2_test,stage3_train, stage3_test = train_test_split(dataset_normal, dataset_prehypertension,dataset_stage1,dataset_stage2,dataset_stage3, test_size = 0.25, random_state = 0)

#Creating the training data set
master_train_dataset=pd.concat([normal_train,prehypertension_train,stage1_train,stage2_train,stage3_train],axis=0,sort=False)
#Creating test set
master_test_dataset=pd.concat([normal_test,prehypertension_test,stage1_test,stage2_test,stage3_test],axis=0,sort=False)
#Adjusting the indexes
master_train_dataset.reset_index(drop=True,inplace=True)
master_test_dataset.reset_index(drop=True,inplace=True)
#Dropping the extra columns
master_train_dataset.drop(master_train_dataset.columns[15],axis=1,inplace=True)
master_train_dataset.drop(master_train_dataset.columns[0],axis=1,inplace=True)
master_train_dataset.drop(master_train_dataset.columns[2],axis=1,inplace=True)
master_test_dataset.drop(master_test_dataset.columns[15],axis=1,inplace=True)
master_test_dataset.drop(master_test_dataset.columns[0],axis=1,inplace=True)
master_test_dataset.drop(master_test_dataset.columns[2],axis=1,inplace=True)
mydata.drop(mydata.columns[15],axis=1,inplace=True)
mydata.drop(mydata.columns[0],axis=1,inplace=True)
mydata.drop(mydata.columns[2],axis=1,inplace=True)


#Creating matrix of features and dependant variable vector
X_train=master_train_dataset.iloc[:,:-1].values
y_train=master_train_dataset.iloc[:,12].values
X_test=master_test_dataset.iloc[:,:-1].values
y_test=master_test_dataset.iloc[:,12].values
mydata_train=mydata.iloc[:,:-1].values
mydata_test=mydata.iloc[:,12].values

#Encoding Categorical data
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencoder_X=LabelEncoder()
X_train[:, 0]= labelencoder_X.fit_transform(X_train[:,0])
X_test[:, 0]= labelencoder_X.fit_transform(X_test[:,0])
mydata_train[:, 0]= labelencoder_X.fit_transform(mydata_train[:,0])
labelencoder_smoke=LabelEncoder()
X_train[:, 11]= labelencoder_smoke.fit_transform(X_train[:,11])
X_test[:, 11]= labelencoder_smoke.fit_transform(X_test[:,11])
mydata_train[:, 11]= labelencoder_smoke.fit_transform(mydata_train[:,11])
onehotencoder = OneHotEncoder(categorical_features = [11])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.fit_transform(X_test).toarray()
mydata_train = onehotencoder.fit_transform(mydata_train).toarray()
print(onehotencoder.feature_indices_)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
mydata_train = sc.fit_transform(mydata_train)
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred1 = classifier.predict(mydata_train)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm1 = confusion_matrix(mydata_test, y_pred1)