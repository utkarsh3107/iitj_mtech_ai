import sys
import os
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras import activations
import numpy as np
from glob import glob


class Assignment1:

    def __init__(self, img_dim_x=224, img_dim_y=224, img_rgb=3):
        self.img_dim_x = img_dim_x
        self.img_dim_y = img_dim_y
        self.img_rgb = img_rgb

    def model(self):
        # Import the VGG19 library as shown below and add preprocessing layer to the front of VGG
        # Here we will be using imagenet weights

        vgg19 = VGG19(input_shape=[self.img_dim_x, self.img_dim_y, self.img_rgb],
                      weights='imagenet',
                      include_top=False)

        # don't train existing weights
        for layer in vgg19.layers:
            layer.trainable = False

        # useful for getting number of output classes
        abs_path = sys.path[0]
        base_name = os.path.dirname(abs_path)
        print(base_name)
        resources_path = os.path.join(base_name, "assignment/VOCdevkit/VOC2012/JPEGImages1")
        folders = glob(resources_path)
        print(folders)

        # our layers - you can add more if you want
        x = Flatten()(vgg19.output)
        prediction = Dense(20, activation=activations.tanh)(x)

        # create a model object
        model = Model(inputs=vgg19.input, outputs=prediction)

        # view the structure of the model
        model.summary()


a1 = Assignment1()
a1.model()
