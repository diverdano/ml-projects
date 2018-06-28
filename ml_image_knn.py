#!/bin/env python3

import imageio
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

import logging
# == set logging ==
#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# load our dataset
logger.info('load data')
train_data = scipy.io.loadmat('projects/image_class/extra_32x32.mat')
# extract the images and labels from the dictionary object
X = train_data['X']
y = train_data['y']
img_index = 25      # view an image (e.g. 25) and print its corresponding label
# plt.imshow(X[:,:,:,img_index])
# plt.show()
logger.info('label for image index: {}'.format(y[img_index]))

#vectorizer
logger.info('vectorize')
X = X.reshape(X.shape[0]*X.shape[1]*X.shape[2],X.shape[3]).T
y = y.reshape(y.shape[0],)
X, y = shuffle(X, y, random_state=42)

# setup classifier
clf = RandomForestClassifier()
logger.info('classifier: {0}'.format(clf))

# train/test split & fit
logger.info('split')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logger.info('fit')
clf.fit(X_train, y_train)

# setup predictions
logger.info('setup predictions')
prediction = clf.predict(X_test)
logger.info("Accuracy: {0:.3f}".format(accuracy_score(y_test, prediction)))
