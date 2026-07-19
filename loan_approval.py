import joblib
import pandas as pd
import numpy as np
import matplotlib .pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load dataset
data = pd.read_csv("dataset/loan_approval_dataset.csv")

# display first 5 rows
print(data.head())

# check dataset shape
print(data.shape)

# check information about columns
print(data.info())

# check missing values
print(data.isnull().sum())

# convert text data into numbers
data[' education'] = data[' education'].map({' Graduate':1, ' Not Graduate':0})
data[' self_employed'] = data[' self_employed'].map({' Yes':1, ' No':0})
data[' loan_status'] = data[' loan_status'].map({' Approved':1, ' Rejected':0})

# check converted data
print(data.head())

# seperate features and target
x = data.drop(' loan_status', axis=1)
y = data[' loan_status']
print(x.columns)
print(len(x.columns))

print(x.head())
print(y.head())

# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
 x, y, test_size=0.2, random_state=42
 )
 
print("training data shape:", x_train.shape)
print("testing data shape:", x_test.shape)

# create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# train the model
model.fit(x_train, y_train)

print("model trained successfully!")

# make predictions
y_pred = model.predict(x_test)

# calculate accuracy 
accuracy = accuracy_score(y_test, y_pred)

print("model accuracy:", accuracy)

# Save the trained model
joblib.dump(model, "loan_model.pkl")

# Save model accuracy
joblib.dump(accuracy, "accuracy.pkl")

print("Model trained and saved successfully!")
print("Files created successfully!")