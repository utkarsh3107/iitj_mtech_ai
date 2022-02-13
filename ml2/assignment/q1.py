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

# other imports
from xml.etree import ElementTree
import numpy as np
import glob
import h5py
import os
import json
import datetime
import time

# Default input size for VGG16
img_width = 224
img_height = 224
no_of_channels_in_rgb = 3
image_size = (224, 224)
# load model without classifier layers
base_model = VGG19(weights='imagenet',include_top=False, input_shape=(img_width, img_height, no_of_channels_in_rgb))
# add new classifier layers
flat1 = Flatten()(base_model.layers[-1].output)
print(flat1)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(20, activation='softmax')(class1)
# define new model
pre_trained_model = Model(inputs=base_model.inputs, outputs=flat1)
# summarize
pre_trained_model.summary()

#train_labels = [['person',1],['aeroplane',2],['tvmonitor',3],['train',4],['boat',5],['dog',6],['chair',7],['bird',8],['bicycle',9],['bottle',10],['diningtable',11],['sheep','horse',12],['sofa',13],['cat',14]]
#train_labels = ['person','aeroplane','tvmonitor','train','boat','dog','chair','bird','bicycle','bottle','diningtable','sheep','horse','sofa','cat']
#train_labels = ['person','aeroplane','tvmonitor','train','boat','dog','chair','bird','bicycle','bottle']

# encode the labels
print("encoding labels.")
le = LabelEncoder()
#le_labels = le.fit_transform(train_labels)

# variables to hold features and labels
features = []
labels   = []

# loop over all the labels in the folder
count = 1
for i, file in enumerate(os.listdir("/Users/utkarsh/Desktop/study/iitj/sem2/ml2/assignment/VOCdevkit/VOC2012/JPEGImages1")):
  name = file.split(".")
  image_path = "/Users/utkarsh/Desktop/study/iitj/sem2/ml2/assignment/VOCdevkit/VOC2012/JPEGImages1/" + name[0]+".jpg"
  annotation_path = "/Users/utkarsh/Desktop/study/iitj/sem2/ml2/assignment/VOCdevkit/VOC2012/Annotations/" + name[0]+".xml"
  #print(file)
  #print(annotation_path)
  #print('image_path')
  #print(image_path)
  img = image.load_img(image_path, target_size=image_size)
  # Converts a PIL Image to 3D Numy Array
  # Adding the fouth dimension, for number of images
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  feature = pre_trained_model.predict(x)
  flat = feature.flatten()
  features.append(flat)
  tree = ElementTree.parse(annotation_path)
  #get the root of the document
  root = tree.getroot()
  objects=root.find('.//object')
  label=objects.find('name').text
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
y=model.predict(x)
label = decode_predictions(y)
label = label[0][0]
print('%s (%.2f%%)' % (label[1], label[2]*100))


#y = np.array([0, 0, 1, 1, 2, 2])
#clf = OneVsRestClassifier(SVC()).fit(, le_labels)
#clf.predict([[-19, -20], [9, 9], [-5, 5]])


