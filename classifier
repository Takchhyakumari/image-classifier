import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

## dataset search
tfds.list_builders()

## info about data
builder = tfds.builder("rock_paper_scissors")
info = builder.info
info

## data prep
ds_train = tfds.load(name="rock_paper_scissors", split="train")
ds_test = tfds.load(name="rock_paper_scissors", split="test")

## show examples
fig = tfds.show_examples(info, ds_train)

## additional data prep
train_images = np.array([example['image'].numpy()[:,:,0] for example in ds_train])
train_labels = np.array([example['label'].numpy() for example in ds_train])
test_images = np.array([example['image'].numpy()[:,:,0] for example in ds_test])
test_labels = np.array([example['label'].numpy() for example in ds_test])

test_images.shape

## processing
train_images = train_images.reshape(2520, 300, 300, 1)
test_images = test_images.reshape(372, 300, 300, 1)

## int values change to float so that they are b/w 0 and 1
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

train_images[0]

## neural network
model = keras.Sequential([
  keras.layers.Flatten(),
  keras.layers.Dense(512, activation='relu'),
  keras.layers.Dense(256, activation='relu'),
  keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', 
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']
              )
model.fit(train_images, train_labels, epochs=5, batch_size=32)

## CNN (using cnn cause nn isnt that useful ryt now as the image size is huge)
model = keras.Sequential([
  ##3X3 matrix for diff size pass in 3X5 so that wud be rectangular
  keras.layers.Conv2D(64, 3, activation='relu', input_shape = (300, 300, 1)),
  keras.layers.Conv2D(32, 3, activation='relu'), 
  keras.layers.Flatten(),
  keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', 
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']
              )
model.fit(train_images, train_labels, epochs=5, batch_size=32)

## improved cnn
## reduce input image size cause its really big and we dont need it so we take an average layer
model = keras.Sequential([
  ##3X3 matrix for diff size pass in 3X5 so that wud be rectangular
  keras.layers.AveragePooling2D(6, 3, input_shape=(300,300,1)),
  keras.layers.Conv2D(64, 3, activation='relu'),
  keras.layers.Conv2D(32, 3, activation='relu'),
  keras.layers.MaxPool2D(2,2),
  keras.layers.Dropout(0.5),
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', 
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']
              )
model.fit(train_images, train_labels, epochs=5, batch_size=32)

model.evaluate(test_images, test_labels)

## keras tuner automatic tuner
!pip install -U keras-tuner

from kerastuner.tuners import RandomSearch

## improved cnn using keras tuner
def build_model(hp):
  model = keras.Sequential()
  model.add(keras.layers.AveragePooling2D(6,3,input_shape=(300,300,1)))
  for i in range(hp.Int("Conv layers", min_value=1, max_value=3)):
    model.add(keras.layers.Conv2D(hp.Choice(f'layer_{i}_filters',[32, 64, 128]),3,activation='relu'))
  model.add(keras.layers.MaxPool2D(2,2))
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(hp.Choice('Dense layer', [512, 1024]), activation='relu'))
  model.add(keras.layers.Dense(3,activation='softmax'))
  model.compile(optimizer='adam', 
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']
              )
  return model
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=32,
    )
tuner.search(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10, batch_size=32)

best_model = tuner.get_best_models()[0]

best_model.evaluate(test_images, test_labels)

best_model.summary()

tuner.results_summary()
