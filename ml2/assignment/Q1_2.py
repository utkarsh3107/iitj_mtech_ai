from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model

import numpy as np
import sys
import os

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

# useful for getting number of output classes
abs_path = sys.path[0]
base_name = os.path.dirname(abs_path)
img_path = os.path.join(base_name, 'assignment/VOCdevkit/VOC2012/JPEGImages1/2007_000027.jpg')
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block_pool_features = model.predict(x)
print(block_pool_features.shape)
print(block_pool_features)

