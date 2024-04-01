import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images, test_images = train_images / 255.0, test_images / 255.0

# one-hot encode label
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# data augmentation
gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
gen_train = gen.flow(train_images, train_labels, batch_size=64)

steps = int(train_images.shape[0] / 64)
class_num = 10
epochs = 50

# stride = 1
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# last output layer
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(class_num, activation='softmax'))
'''
# stride = 2
model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), strides=2, activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(32, (3, 3), strides=2, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(64, (1, 1), strides=2, activation='relu'))
# last output layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))
'''
#model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(gen_train, steps_per_epoch=steps, epochs=epochs, validation_data=(test_images, test_labels))
'''
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

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
plt.legend(loc='upper right')
plt.show()

#_, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#print(test_acc)

# draw distribution of weights and biases
conv1_weights = model.layers[0].get_weights()[0]
conv1_biases = model.layers[0].get_weights()[1]
conv2_weights = model.layers[6].get_weights()[0]
conv2_biases = model.layers[6].get_weights()[1]
dense1_weights = model.layers[8].get_weights()[0]
dense1_biases = model.layers[8].get_weights()[1]
output_weights = model.layers[10].get_weights()[0]
output_biases = model.layers[10].get_weights()[1]

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
#model.save('./cifar10.h5')
'''
# load model
model = load_model('./cifar10.h5')

# Predict the values from the testing dataset
pred = model.predict(test_images)

# draw predicted images            
for i in range(20):
  plt.imshow(test_images[i], cmap='gray')
  plt.title(f"Label: {np.argmax(test_labels[i])}\nPrediction: {np.argmax(pred[i])}")
  plt.show()
'''