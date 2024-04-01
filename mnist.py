import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# reshape and normalize data (60000, 28, 28, 1)
train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1)
train_images = train_images / 255.0
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)
test_images = test_images / 255.0

# one-hot encode label
train_labels = tf.one_hot(train_labels.astype(np.int32), depth=10)
test_labels = tf.one_hot(test_labels.astype(np.int32), depth=10)

class_num = 10
epochs = 5

# strides = 1
model = models.Sequential()
model.add(layers.Conv2D(32, (5,5), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=tf.keras.regularizers.l2(0.01))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))) 
# last output layer
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(class_num, activation='softmax'))

# strides = 2
'''
model = models.Sequential()
model.add(layers.Conv2D(16, (5,5), strides=2, activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (2, 2), activation='relu')) 
# last output layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(class_num, activation='softmax'))
'''

#model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))

_, test_acc = model.evaluate(test_images, test_labels)
'''
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
e = range(epochs)

# Accuracy
plt.plot(train_acc, label='train_accuracy')
plt.plot(val_acc, label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Loss
plt.plot(train_loss, label='train_loss')
plt.plot(val_loss, label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.show()

# draw distribution of weights and biases
conv1_weights = model.layers[0].get_weights()[0]
conv1_biases = model.layers[0].get_weights()[1]
conv2_weights = model.layers[4].get_weights()[0]
conv2_biases = model.layers[4].get_weights()[1]
dense1_weights = model.layers[6].get_weights()[0]
dense1_biases = model.layers[6].get_weights()[1]
output_weights = model.layers[7].get_weights()[0]
output_biases = model.layers[7].get_weights()[1]

conv1 = np.concatenate([conv1_weights.ravel(), conv1_biases])
conv2 = np.concatenate([conv2_weights.ravel(), conv2_biases])
dense1 = np.concatenate([dense1_weights.ravel(), dense1_biases])
output = np.concatenate([output_weights.ravel(), output_biases])

plt.hist(conv1)
plt.xlabel('Value')
plt.ylabel('Number')
plt.title('conv1')
plt.show()

plt.hist(conv2)
plt.xlabel('Value')
plt.ylabel('Number')
plt.title('conv2')
plt.show()

plt.hist(dense1)
plt.xlabel('Value')
plt.ylabel('Number')
plt.title('dense1')
plt.show()

plt.hist(output)
plt.xlabel('Value')
plt.ylabel('Number')
plt.title('output')
plt.show()
'''

# save model
#model.save('./mnist.h5')

# load model
#model = load_model('./mnist.h5')

# Predict the values from the testing dataset
#pred = model.predict(test_images)
    
# draw predicted images            
'''
for i in range(321, 322):
  plt.imshow(test_images[i], cmap='gray')
  plt.title(f"Label: {np.argmax(test_labels[i])}\nPrediction: {np.argmax(pred[i])}")
  plt.show()
'''
