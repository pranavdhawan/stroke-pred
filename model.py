# Simple Linear Regression

'''
This model predicts the salary of the employ based on experience using simple linear regression model.
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import requests
import json

# Importing the dataset
df = pd.read_csv('dataset.csv')
df.drop(['id'],axis=1,inplace=True)

# X = df.iloc[:, :-1].values
# y = df.iloc[:, 1].values

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

lb = LabelBinarizer()
label_encoder = LabelEncoder()
df['gender'] = lb.fit_transform(df['gender']) # male = 0 female = 1
df['ever_married'] = lb.fit_transform(df['ever_married']) # no = 0 yes = 1
df['work_type'] = label_encoder.fit_transform(df['work_type']) # private = 2 self = 3 gov = 0 child = 1
df['Residence_type'] = label_encoder.fit_transform(df['Residence_type']) # urban = 1 rural = 0 
df['smoking_status'] = label_encoder.fit_transform(df['smoking_status']) # formerly = 1 never = 2 regular = 3 unknown = 0
df.dropna(inplace=True)



from sklearn.model_selection import train_test_split


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X=df.drop(['stroke'],axis=1)
y=df['stroke']

from sklearn.linear_model import LogisticRegression

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

reg = LogisticRegression()
reg.fit(x_train, y_train)


# Predicting the Test set results
pre = reg.predict(x_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pre, y_test)

# Saving model using pickle
pickle.dump(reg, open('model.pkl','wb'))

# Loading model to compare the results
# model = pickle.load( open('model.pkl','rb'))
# print(model.predict([[1.8]]))
