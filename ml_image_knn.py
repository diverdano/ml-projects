#!/bin/env python3

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
# from sklearn.metrics import accuracy_score
import logging

# == set logging ==
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def image_to_feature_vector(image, size=(32, 32)):
    '''resize the image to a fixed size, then flatten the image into a list of raw pixel intensities'''
    return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
    '''extract a 3D color histogram from the HSV color space using the supplied number of `bins` per channel'''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    if imutils.is_cv2():   hist = cv2.normalize(hist)   # handle normalizing the histogram if we are using OpenCV 2.4.X
    else:                  cv2.normalize(hist, hist)    # otherwise, perform "in place" normalization in OpenCV 3
    return hist.flatten()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",      default='projects/image_class/knn-classifier/kaggle_dogs_vs_cats/train/',   help="path to input dataset")
ap.add_argument("-k", "--neighbors",    type=int, default=1,                    help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs",         type=int, default=-1,                   help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
logger.info('describing images')
imagePaths = list(paths.list_images(args["dataset"]))
# initialize the raw pixel intensities matrix, the features matrix, and labels list
rawImages   = []
features    = []
labels      = []
# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
    # load the image and extract the class label (assuming that our path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image       = cv2.imread(imagePath)
    label       = imagePath.split(os.path.sep)[-1].split(".")[0]
    # extract raw pixel intensity "features", followed by a color histogram to characterize the color distribution of the pixels in the image
    pixels      = image_to_feature_vector(image)
    hist        = extract_color_histogram(image)
    # update the raw images, features, and labels matricies, respectively
    rawImages.append(pixels)
    features.append(hist)
    labels.append(label)
    # show an update every 1,000 images
    if i > 0 and i % 5000 == 0: logger.info("processed {0}/{1}".format(i, len(imagePaths)))

# show some information on the memory consumed by the raw images matrix and features matrix
rawImages       = np.array(rawImages)
features        = np.array(features)
labels          = np.array(labels)
logger.info('pixels matrix: {:.2f}MB'.format(rawImages.nbytes / (1024 * 1000.0)))
logger.info('features matrix: {:.2f}MB'.format(features.nbytes / (1024 * 1000.0)))

# split train/test - 75% for train, 25% for test - can we remove the ()?
trainRI, testRI, trainRL, testRL                = train_test_split(rawImages, labels, test_size=0.25, random_state=42)
trainFeat, testFeat, trainLabels, testLabels    = train_test_split(features, labels, test_size=0.25, random_state=42)

# train and evaluate a k-NN classifer on the raw pixel intensities
logger.info('evaluating raw pixel accuracy...')
model           = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
logger.info('fitting training raw images/labels to classifier')
model.fit(trainRI, trainRL)
logger.info('scoring model')
acc             = model.score(testRI, testRL)
logger.info('raw pixel accuracy: {:.2f}%'.format(acc * 100))

# representations
logger.info('evaluating histogram accuracy...')
model           = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
logger.info('fitting training features/labels to classifier')
model.fit(trainFeat, trainLabels)
logger.info('scoring model')
acc             = model.score(testFeat, testLabels)
logger.info('histogram accuracy: {:.2f}%'.format(acc * 100))
