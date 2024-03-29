#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import time

from tensorflow.keras import datasets, layers, models


# In[2]:


# Datasets
train_path = '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_LANCZOS3/train'
test_path =  '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_LANCZOS3/test'
csv_names = [
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/loss/BinaryCrossentropy.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/loss/BinaryFocalCrossentropy.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/loss/CategoricalCrossentropy.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/loss/CategoricalFocalCrossentropy.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/loss/CategoricalHinge.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/loss/CosineSimilarity.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/loss/Hinge.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/loss/Huber.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/loss/KLDivergence.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/loss/LogCosh.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/loss/MeanAbsolutePercentageError.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/loss/MeanSquaredError.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/loss/MeanSquaredLogarithmicError.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/loss/SquaredHinge.csv'
]

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


# In[3]:


# Keeping time
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()


# In[4]:


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
    
def compile_model(model, loss_function):
    model.compile(
        optimizer='adamax',
        loss=loss_function,
        metrics=['accuracy']
    )
    return model


# In[5]:


# List of optimizers to iterate over
losses = (
    tf.keras.losses.BinaryCrossentropy(),
    tf.keras.losses.BinaryFocalCrossentropy(),
    tf.keras.losses.CategoricalCrossentropy(),
    tf.keras.losses.CategoricalFocalCrossentropy(),
    tf.keras.losses.CategoricalHinge(),
    tf.keras.losses.CosineSimilarity(),
    tf.keras.losses.Hinge(),
    tf.keras.losses.Huber(),
    tf.keras.losses.KLDivergence(),
    tf.keras.losses.LogCosh(),
    tf.keras.losses.MeanAbsolutePercentageError(),
    tf.keras.losses.MeanSquaredError(),
    tf.keras.losses.MeanSquaredLogarithmicError(),
    tf.keras.losses.SquaredHinge()
)

for i in range(len(csv_names)):
    print("-------------------\n")
    print(losses[i])
    print(csv_names[i])
    print("-------------------\n")
    
    # fetch datasets
    train = tf.data.Dataset.load(train_path)
    test  = tf.data.Dataset.load(test_path)
    train, test = format_set(train, test)

    # create model
    model = create_model()
    model = compile_model(model, losses[i])

    # Train
    history = model.fit(train, epochs=250, validation_data=test, callbacks=[time_callback])

    # Convert the history.history dict to a pandas DataFrame
    hist_df = pd.DataFrame(history.history)

    # Add epoch times
    hist_df['time'] = time_callback.times

    # Save to csv
    hist_df.to_csv(csv_names[i])


# In[ ]:




