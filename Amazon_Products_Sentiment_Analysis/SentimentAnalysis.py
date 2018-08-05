import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#Importing the dataset
dataset=pd.read_csv('amz.txt', delimiter = '\t', quoting = 3)
#Data Analysis
description=dataset.describe()
dataset['reviews.doRecommend'].value_counts()
dataset['brand'].value_counts()
dataset['categories'].value_counts()
dataset['manufacturer'].value_counts()
dataset['reviews.numHelpful'].value_counts()
# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
dataset['reviews.text'].fillna('@',inplace=True)
for i in range(0, 41421):
    review = re.sub('[^a-zA-Z]', ' ', dataset['reviews.text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 40000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 14].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

