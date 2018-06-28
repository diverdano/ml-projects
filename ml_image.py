#!/bin/env python3
import imageio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load images
al = imageio.imread('~/Documents/abelincoln.jpg')
bl = imageio.imread('~/Documents/bangalorelake.jpg')

# setup training data with images and labels
train_data = {'X': [al, bl], 'y': ['face', 'place']}

# setup classifier
clf = RandomForestClassifier()
print(clf)

# train/test split & fit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

# setup predictions
prediction = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, prediction))
