import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("./creditcard/creditcard.csv")

# Select features
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Plot original quantity of samples per category
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(['0', '1'], [(y == 0).sum(), (y == 1).sum()], color=['blue', 'red'])
ax.set_title('Original Distribution')
ax.set_ylabel('Number of Samples')
plt.savefig("./results/before_undersampling.png")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

clf = make_pipeline(StandardScaler(), SVC())
clf.fit(X_train, y_train)

# Calculate the score
print("Before undersampling application")
score_value = clf.score(X_test, y_test) * 100
print(f"Score value: {score_value:.2f} %")


print("#################")
####Applying the Random UnderSampling
# Reduce the quantity of samples of '0' type to be equal to '1'
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Plot distribution after undersampling
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(['0', '1'], [(y_resampled == 0).sum(), (y_resampled == 1).sum()], color=['blue', 'red'])
ax.set_title('Distribution After Random Undersampling')
ax.set_ylabel('Number of Samples')
plt.savefig("./results/after_undersampling.png")
plt.show()

#Applying thr SVM model to detect the anomaly 

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=123)

clf = make_pipeline(StandardScaler(), SVC())
clf.fit(X_train, y_train)

# Calculate the score
print("After undersampling application")
score_value = clf.score(X_test, y_test) * 100
print(f"Score value: {score_value:.2f} %")
