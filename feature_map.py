import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model, Model
import numpy as np
import matplotlib.pyplot as plt
'''
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# reshape and normalize data (60000, 28, 28, 1)
train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1)
train_images = train_images / 255.0
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)
test_images = test_images / 255.0

model = load_model('./mnist.h5')
img = test_images[51].reshape(1,28,28,1)
'''

model = load_model('./cifar10.h5')

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images, test_images = train_images / 255.0, test_images / 255.0

# redefine model to output right after the first hidden layer
ixs = [0, 2, 4]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)
img = test_images[51].reshape(1,32,32,3)

fig = plt.figure(figsize=(5,5))
plt.imshow(img[0,:,:,0],cmap="gray")
plt.axis('off')
plt.show()

feature_maps = model.predict(img)

square = 3
for map in feature_maps:
	ix = 1
	for i in range(square):
		for j in range(square):
			ax = plt.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			plt.imshow(map[0, :, :, ix-1])
			ix += 1
	
	plt.show()
    

