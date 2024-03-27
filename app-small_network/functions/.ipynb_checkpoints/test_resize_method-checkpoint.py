# import packages
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import time

from tensorflow.keras import datasets, layers, models

# Datasets
train_paths = [
    '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_AREA/train',
    '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_BICUBIC/train',
    '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_BILINEAR/train',
    '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_GAUSSIAN/train',
    '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_LANCZOS3/train',
    '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_LANCZOS5/train',
    '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_MITCHELLCUBIC/train',
    '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_NEAREST_NEIGHBOR/train'   
]
test_paths = [
    '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_AREA/test',
    '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_BICUBIC/test',
    '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_BILINEAR/test',
    '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_GAUSSIAN/test',
    '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_LANCZOS3/test',
    '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_LANCZOS5/test',
    '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_MITCHELLCUBIC/test',
    '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_NEAREST_NEIGHBOR/test'   
]
csv_names = [
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/AREA.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/BICUBIC.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/BILINEAR.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/GAUSSIAN.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/LANCZOS3.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/LANCZOS5.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/MITCHELLCUBIC.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/NEAREST_NEIGHBOR.csv'
]

#train = tf.data.Dataset.load('/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16/train')
#test  = tf.data.Dataset.load('/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16/test')

def preprocess(example):
    image = example['image']
    image.set_shape([16, 16, 1])
    label = example['label']
    label = tf.one_hot(label, depth=100)  # One-hot encode the labels
    return image, label

def format_set(train_set, test_set):
    # format and cache
    train_set_formatted = train_set.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_set_formatted = train_set_formatted.cache()
    train_set_formatted = train_set_formatted.batch(512)
    train_set_formatted = train_set_formatted.prefetch(tf.data.AUTOTUNE)
    test_set_formatted = test_set.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_set_formatted = test_set_formatted.batch(512)
    test_set_formatted = test_set_formatted.cache()
    test_set_formatted = test_set_formatted.prefetch(tf.data.AUTOTUNE)
    return train_set_formatted, test_set_formatted

# Keeping time
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()

# Model taken from example (https://www.tensorflow.org/tutorials/images/cnn)
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of classes
    return model
    
def compile_model(model):
    model.compile(
        optimizer='adam',
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['accuracy']
    )
    return model

#train = tf.data.Dataset.load('/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16/train')
#test  = tf.data.Dataset.load('/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16/test')

for i in range(len(csv_names)):
    # fetch datasets
    train = tf.data.Dataset.load(train_paths[i])
    test  = tf.data.Dataset.load(test_paths[i])
    train, test = format_set(train, test)

    # create model
    model = create_model()
    model = compile_model(model)

    # Train
    history = model.fit(train, epochs=250, validation_data=test, callbacks=[time_callback])

    # Convert the history.history dict to a pandas DataFrame
    hist_df = pd.DataFrame(history.history)

    # Add epoch times
    hist_df['time'] = time_callback.times

    # Save to csv
    hist_df.to_csv(csv_names[i])