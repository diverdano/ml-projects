#!/bin/env python3

import tensorflow as tf
from imutils import paths
import argparse
import imutils
import cv2
import numpy as np
import os
import logging

# == set logging ==
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",      default='projects/image_class/knn-classifier/kaggle_dogs_vs_cats/train/',   help="path to input dataset")
ap.add_argument("-k", "--neighbors",    type=int, default=1,                    help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs",         type=int, default=-1,                   help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
logger.info('describing images')
imagePaths = list(paths.list_images(args["dataset"]))




# get shape
a               = tf.truncated_normal([16,128,128,3])
sess            = tf.Session()
sess.run(tf.initialize_all_variables())
sess.run(tf.shape(a))
# reshape
b               = tf.reshape(a,[16,49152])
sess.run(tf.shape(b))

classes         = ['dogs', 'cats']
num_classes     = len(classes)
train_path      = 'training_data'

# validation split
validation_size = 0.2
# batch size
batch_size      = 16
data            = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)
    ## Creating the convolutional layer
    layer           = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer           += biases
    ## We shall be using max-pooling.
    layer           = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)
    return layer

def create_flatten_layer(layer):
    layer_shape     = layer.get_shape()
    num_features    = layer_shape[1:4].num_elements()
    layer           = tf.reshape(layer, [-1, num_features])
    return layer

def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    #Let's define trainable weights and biases.
    weights         = create_weights(shape=[num_inputs, num_outputs])
    biases          = create_biases(num_outputs)
    layer           = tf.matmul(input, weights) + biases
    if use_relu:
        layer       = tf.nn.relu(layer)
    return layer

x               = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')
y_true          = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls      = tf.argmax(y_true, dimension=1)

layer_conv1     = create_convolutional_layer(input=x,
                num_input_channels=num_channels,
                conv_filter_size=filter_size_conv1,
                num_filters=num_filters_conv1)

layer_conv2     = create_convolutional_layer(input=layer_conv1,
                num_input_channels=num_filters_conv1,
                conv_filter_size=filter_size_conv2,
                num_filters=num_filters_conv2)

layer_conv3     = create_convolutional_layer(input=layer_conv2,
                num_input_channels=num_filters_conv2,
                conv_filter_size=filter_size_conv3,
                num_filters=num_filters_conv3)

layer_flat      = create_flatten_layer(layer_conv3)

layer_fc1       = create_fc_layer(input=layer_flat,
                num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                num_outputs=fc_layer_size,
                use_relu=True)

layer_fc2       = create_fc_layer(input=layer_fc1,
                num_inputs=fc_layer_size,
                num_outputs=num_classes,
                use_relu=False)

cross_entropy   = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost            = tf.reduce_mean(cross_entropy)

batch_size      = 16
x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
feed_dict_train = {x: x_batch, y_true: y_true_batch}
session.run(optimizer, feed_dict=feed_dict_tr)

x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)
feed_dict_val   = {x: x_valid_batch, y_true: y_valid_batch}
val_loss        = session.run(cost, feed_dict=feed_dict_val)

correct_prediction  = tf.equal(y_pred_cls, y_true_cls)
accuracy            = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver.save(session, 'dogs-cats-model')

def train(num_iteration):
    global total_iterations
    for i in range(total_iterations, total_iterations + num_iteration):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)
        feed_dict_tr = {x: x_batch, y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}
        session.run(optimizer, feed_dict=feed_dict_tr)
        if i % int(data.train.num_examples/batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, 'dogs-cats-model')
    total_iterations += num_iteration

saver       = tf.train.import_meta_graph('flowers-model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

image           = cv2.imread(filename)
# Resizing the image to our desired size and preprocessing will be done exactly as done during training
image           = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
images.append(image)
images          = np.array(images, dtype=np.uint8)
images          = images.astype('float32')
images          = np.multiply(images, 1.0/255.0)
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch         = images.reshape(1, image_size,image_size,num_channels)
graph           = tf.get_default_graph()
y_pred          = graph.get_tensor_by_name("y_pred:0")
## Let's feed the images to the input placeholders
x               = graph.get_tensor_by_name("x:0")
y_true          = graph.get_tensor_by_name("y_true:0")
y_test_images   = np.zeros((1, 2))

feed_dict_testing = {x: x_batch, y_true: y_test_images}
result          = sess.run(y_pred, feed_dict=feed_dict_testing)

#
#
#
# # initialize the raw pixel intensities matrix, the features matrix, and labels list
# rawImages   = []
# features    = []
# labels      = []
# # loop over the input images
# for (i, imagePath) in enumerate(imagePaths):
#     # load the image and extract the class label (assuming that our path as the format: /path/to/dataset/{class}.{image_num}.jpg
#     image       = cv2.imread(imagePath)
#     label       = imagePath.split(os.path.sep)[-1].split(".")[0]
#     # extract raw pixel intensity "features", followed by a color histogram to characterize the color distribution of the pixels in the image
#     pixels      = image_to_feature_vector(image)
#     hist        = extract_color_histogram(image)
#     # update the raw images, features, and labels matricies, respectively
#     rawImages.append(pixels)
#     features.append(hist)
#     labels.append(label)
#     # show an update every 1,000 images
#     if i > 0 and i % 5000 == 0: logger.info("processed {0}/{1}".format(i, len(imagePaths)))
#
# # show some information on the memory consumed by the raw images matrix and features matrix
# rawImages       = np.array(rawImages)
# features        = np.array(features)
# labels          = np.array(labels)
# logger.info('pixels matrix: {:.2f}MB'.format(rawImages.nbytes / (1024 * 1000.0)))
# logger.info('features matrix: {:.2f}MB'.format(features.nbytes / (1024 * 1000.0)))
#
# # split train/test - 75% for train, 25% for test - can we remove the ()?
# trainRI, testRI, trainRL, testRL                = train_test_split(rawImages, labels, test_size=0.25, random_state=42)
# trainFeat, testFeat, trainLabels, testLabels    = train_test_split(features, labels, test_size=0.25, random_state=42)
#
# # train and evaluate a k-NN classifer on the raw pixel intensities
# logger.info('evaluating raw pixel accuracy...')
# model           = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
# logger.info('fitting training raw images/labels to classifier')
# model.fit(trainRI, trainRL)
# logger.info('scoring model')
# acc             = model.score(testRI, testRL)
# logger.info('raw pixel accuracy: {:.2f}%'.format(acc * 100))
#
# # representations
# logger.info('evaluating histogram accuracy...')
# model           = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
# logger.info('fitting training features/labels to classifier')
# model.fit(trainFeat, trainLabels)
# logger.info('scoring model')
# acc             = model.score(testFeat, testLabels)
# logger.info('histogram accuracy: {:.2f}%'.format(acc * 100))
