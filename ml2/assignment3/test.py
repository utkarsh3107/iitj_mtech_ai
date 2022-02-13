import gzip
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size

f = gzip.open('data/train-images-idx3-ubyte.gz','r')

image_size = 28
num_images = 5

f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)

print(size(data))
image = np.asarray(data[1]).squeeze()
plt.imshow(image)
plt.show()
