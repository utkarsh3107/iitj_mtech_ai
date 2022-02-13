from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import decode_predictions
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pickle import dump
from sklearn.metrics import classification_report

from tensorflow.keras import layers
from tensorflow.keras import activations

# other imports
from xml.etree import ElementTree
import numpy as np
import glob
import h5py
import os
import json
import datetime
import time


class Q1:

    def __init__(self, img_dim_x=224, img_dim_y=224, img_dim_rgb=3):
        self.img_dim_x = img_dim_x
        self.img_dim_y = img_dim_y
        self.img_dim_rgb = img_dim_rgb
        self.img_size = (self.img_dim_x, self.img_dim_y)

    """
        def part0(self):
        # load model without classifier layers
        model_vgg19 = VGG19()

        # load an image from file
        img = load_img('/Users/utkarsh/Desktop/study/iitj/sem2/ml2/assignment/VOCdevkit/VOC2012/JPEGImages1/2007_000027.jpg', target_size=(224, 224))

        # convert the image pixels to a numpy array
        img = img_to_array(img)

        # reshape data for the model
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

        # prepare the image for the VGG model
        img = preprocess_input(img)

        # predict the probability across all output classes
        yhat = model_vgg19.predict(img)
        # convert the probabilities to class labels
        label = decode_predictions(yhat)
        # retrieve the most likely result, e.g. highest probability
        label = label[0][0]
        # print the classification
        print('%s (%.2f%%)' % (label[1], label[2] * 100))
    """

    def part01(self):
        # load model without classifier layers
        base_model = VGG19(weights='imagenet',
                           include_top=False,
                           input_shape=(self.img_dim_x, self.img_dim_y, self.img_dim_rgb))
        # add new classifier layers
        #flat1 = Flatten()(base_model.layers[-1].output)
        #print(flat1)
        #class1 = Dense(1024, activation=activations.relu)(flat1)
        #output = Dense(20, activation=activations.softmax)(class1)
        # define new model
        #pre_trained_model = Model(inputs=base_model.inputs, outputs=flat1)
        # summarize
        base_model.summary()

    def part1(self):
        # load model without classifier layers
        model_vgg19 = VGG19(weights='imagenet',
                            include_top=False,
                            input_shape=(self.img_dim_x, self.img_dim_y, self.img_dim_rgb))
        flat_result = Flatten()(model_vgg19.layers[-1].output)
        print(flat_result)

        # add new classifier layers
        dense_layer = Dense(1024, activations.relu)(flat_result)
        output = Dense(20, activations.tanh)(dense_layer)

        # define new model
        pre_trained_model = Model(inputs=model_vgg19.inputs,
                                  outputs=output)

        # summarize
        pre_trained_model.summary()

        return pre_trained_model

    def part2(self):
        pre_trained_model = self.part1()

        print("encoding labels.")
        le = LabelEncoder()

        # variables to hold features and labels
        features = []
        labels = []

        # loop over all the labels in the folder
        count = 1
        for i, file in enumerate(
                os.listdir("/Users/utkarsh/Desktop/study/iitj/sem2/ml2/assignment/VOCdevkit/VOC2012/JPEGImages1")):
            name = file.split(".")
            image_path = "/Users/utkarsh/Desktop/study/iitj/sem2/ml2/assignment/VOCdevkit/VOC2012/JPEGImages1/" + name[
                0] + ".jpg"
            annotation_path = "/Users/utkarsh/Desktop/study/iitj/sem2/ml2/assignment/VOCdevkit/VOC2012/Annotations/" + \
                              name[0] + ".xml"
            img = image.load_img(image_path, target_size=self.img_size)
            # Converts a PIL Image to 3D Numy Array
            # Adding the fouth dimension, for number of images
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = pre_trained_model.predict(x)
            flat = feature.flatten()
            features.append(flat)
            tree = ElementTree.parse(annotation_path)
            # get the root of the document
            root = tree.getroot()
            objects = root.find('.//object')
            label = objects.find('name').text
            labels.append(label)
            count += 1

        print("features: {}".format(features))
        print("labels: {}".format(labels))
        print("unique labels: {}".format(list(set(labels))))

        le_labels = le.fit_transform(labels)

        # get the shape of training labels
        print("encoded labels: {}".format(le_labels))
        print("features shape: {}".format(np.array(features).shape))
        print("encoded labels shape: {}".format(le_labels.shape))

        (train_data, test_data, train_labels, test_labels) = train_test_split(np.array(features),
                                                                              np.array(le_labels),
                                                                              test_size=0.3,
                                                                              random_state=100)

        print("splitted data...")
        print("train data  : {}".format(train_data.shape))
        print("test data   : {}".format(test_data.shape))
        print("train labels: {}".format(train_labels.shape))
        print("test labels : {}".format(test_labels.shape))

        # use logistic regression as the model
        model = OneVsRestClassifier(SVC())
        model.fit(np.array(features), np.array(le_labels))
        print("created SVM model")
        y = model.predict(x)
        label = decode_predictions(y)
        label = label[0][0]
        print('%s (%.2f%%)' % (label[1], label[2] * 100))


q1 = Q1()
q1.part01()