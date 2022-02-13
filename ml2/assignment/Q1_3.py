from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


import numpy as np
import sys
import os
import glob
from pathlib import Path
from xml.etree import ElementTree


class Q1_3:

    def __init__(self, img_dim_x=224, img_dim_y=224, img_rgb=3, img_folder='assignment/VOCdevkit/VOC2012/'):
        self.img_dim_x = img_dim_x
        self.img_dim_y = img_dim_y
        self.img_rgb = img_rgb
        self.img_dim = [img_dim_x, img_dim_y]
        self.img_folder = img_folder

    def execute(self):
        vgg19 = VGG19(weights='imagenet')
        vgg19.summary()

        model = Model(inputs=vgg19.input,
                      outputs=vgg19.get_layer('fc2').output)

        # useful for getting number of output classes
        abs_path = sys.path[0]
        base_name = os.path.dirname(abs_path)
        img_path = os.path.join(base_name, 'assignment/VOCdevkit/VOC2012/JPEGImages1/2007_000027.jpg')

        img = image.load_img(img_path, target_size=(self.img_dim_x, self.img_dim_y))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        block_pool_features = model.predict(x)
        print(block_pool_features.shape)
        print(block_pool_features)

    def execute_many(self):
        vgg19 = VGG19(weights='imagenet')
        vgg19.summary()

        model = Model(inputs=vgg19.input,
                      outputs=vgg19.get_layer('fc2').output)

        labels = []
        features = []
        # useful for getting number of output classes
        img_dict = self.read()
        img_paths = img_dict['img_path']
        xml_paths = img_dict['xml_path']
        for index in range(0, len(img_paths)):
            img = image.load_img(img_paths[index], target_size=(self.img_dim_x, self.img_dim_y))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            block_pool_features = model.predict(x)
            flat_pool_features = block_pool_features.flatten()
            features.append(flat_pool_features)
            labels.append(ElementTree.parse(xml_paths[index]).getroot().find('.//object').find('name').text)
            #objects = ElementTree.parse(xml_paths[index]).getroot().findall('.//object')
            #label = []
            #[label.append(each.find('name').text) for each in objects]
            #labels.append(list(dict.fromkeys(label)))
        # print(labels)
        # print(len(features))
        #flat_list = [item for sublist in labels for item in sublist]

        print(len(features[0]))
        print("features: {}".format(features))
        print("labels: {}".format(labels))
        print("unique labels: {}".format(list(set(labels))))

        le_labels = LabelEncoder().fit_transform(labels)

        # get the shape of training labels
        print("encoded labels: {}".format(le_labels))
        print("features shape: {}".format(np.array(features).shape))
        print("encoded labels shape: {}".format(le_labels.shape))

        (X_train, X_test, y_train, y_test) = train_test_split(np.array(features),
                                                                              np.array(le_labels),
                                                                              test_size=0.3,
                                                                              random_state=100)

        print("splitted data...")
        print("train data  : {}".format(X_train.shape))
        print("test data   : {}".format(X_test.shape))
        print("train labels: {}".format(y_train.shape))
        print("test labels : {}".format(y_test.shape))

        # use logistic regression as the model
        #model = OneVsRestClassifier(SVC())
        #model.fit(np.array(features), np.array(le_labels))
        #print("created SVM model")

        # Creating the SVM model
        model = OneVsRestClassifier(SVC())

        # Fitting the model with training data
        model.fit(X_train, y_train)

        # Making a prediction on the test set
        prediction = model.predict(X_test)

        # Evaluating the model
        print("Test Set Accuracy : {}".format(accuracy_score(y_test, prediction)))
        print("Classification Report : {}".format(classification_report(y_test, prediction)))

    def read(self):
        img_dict = {'img_path': self.read_image(), 'xml_path': self.read_annotation()}
        return img_dict

    def read_annotation(self):
        abs_path = sys.path[0]
        base_name = os.path.dirname(abs_path)
        img_files = self.read_image()
        xml_files = []
        for file in img_files:
            filename = Path(file).stem
            xml_dir = os.path.join(base_name, self.img_folder + 'Annotations/')
            xml_file = xml_dir + filename + ".xml"
            xml_files.append(xml_file)

        return xml_files

    def read_path(self):
        files = self.read_image()
        for file in files:
            print(Path(file).stem)

    def read_image(self):
        abs_path = sys.path[0]
        base_name = os.path.dirname(abs_path)
        img_dir = os.path.join(base_name, self.img_folder + 'JPEGImages/')
        ext = ['png', 'jpg', 'gif']
        files = []
        [files.extend(glob.glob(img_dir + '*.' + e)) for e in ext]
        return files


q1 = Q1_3()
q1.execute_many()
