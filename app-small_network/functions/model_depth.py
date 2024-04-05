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

from contextlib import redirect_stdout
from tensorflow.keras import datasets, layers, models


# In[2]:


# Datasets
train_path = '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_LANCZOS3/train'
test_path =  '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_LANCZOS3/test'
csv_names = [
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_depth/base_model.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_depth/oneofeach_model.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_depth/twoofeach_model.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_depth/threeofeach_model.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_depth/fourofeach_model.csv'
]
summary_names = [
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_depth/base_model_summary.txt',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_depth/oneofeach_model_summary.txt',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_depth/twoofeach_model_summary.txt',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_depth/threeofeach_model_summary.txt',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_depth/fourofeach_model_summary.txt'
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
base_model = models.Sequential()
base_model.add(layers.Conv2D(32, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
base_model.add(layers.MaxPooling2D((2, 2)))
base_model.add(layers.Conv2D(64, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
base_model.add(layers.MaxPooling2D((2, 2)))
base_model.add(layers.Conv2D(64, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
base_model.add(layers.Flatten())
base_model.add(layers.Dense(64, activation='relu'))
base_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of classes

# 1 of each
oneofeach_model = models.Sequential()
oneofeach_model.add(layers.Conv2D(32, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
oneofeach_model.add(layers.MaxPooling2D((2, 2)))
oneofeach_model.add(layers.Flatten())
oneofeach_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of classes

# 2 of each
twoofeach_model = models.Sequential()
twoofeach_model.add(layers.Conv2D(32, (3, 3), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
twoofeach_model.add(layers.MaxPooling2D((2, 2)))
twoofeach_model.add(layers.Conv2D(64, (3, 3), activation='relu')) # (3, 3) -> (2, 2)
twoofeach_model.add(layers.MaxPooling2D((2, 2)))
twoofeach_model.add(layers.Flatten())
twoofeach_model.add(layers.Dense(64, activation='relu'))
twoofeach_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of classes

# 3 of each
threeofeach_model = models.Sequential()
threeofeach_model.add(layers.Conv2D(32, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
threeofeach_model.add(layers.MaxPooling2D((2, 2)))
threeofeach_model.add(layers.Conv2D(64, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
threeofeach_model.add(layers.MaxPooling2D((2, 2)))
threeofeach_model.add(layers.Conv2D(64, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
threeofeach_model.add(layers.MaxPooling2D((2, 2)))
threeofeach_model.add(layers.Flatten())
threeofeach_model.add(layers.Dense(64, activation='relu'))
threeofeach_model.add(layers.Dense(64, activation='relu'))
threeofeach_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of classes

# 4 of each
fourofeach_model = models.Sequential()
fourofeach_model.add(layers.Conv2D(32, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
fourofeach_model.add(layers.MaxPooling2D((2, 2)))
fourofeach_model.add(layers.Conv2D(64, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
fourofeach_model.add(layers.MaxPooling2D((2, 2)))
fourofeach_model.add(layers.Conv2D(64, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
fourofeach_model.add(layers.MaxPooling2D((2, 2)))
fourofeach_model.add(layers.Conv2D(64, (1, 1), activation='relu')) # (3, 3) -> (2, 2)
fourofeach_model.add(layers.MaxPooling2D((1, 1)))
fourofeach_model.add(layers.Flatten())
fourofeach_model.add(layers.Dense(64, activation='relu'))
fourofeach_model.add(layers.Dense(64, activation='relu'))
fourofeach_model.add(layers.Dense(64, activation='relu'))
fourofeach_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of classes


dnn_models = [
    base_model,
    oneofeach_model,
    twoofeach_model,
    threeofeach_model,
    fourofeach_model
]
    


# In[5]:


def compile_model(model):
    model.compile(
        optimizer='adamax',
        loss=tf.keras.losses.BinaryFocalCrossentropy(),
        metrics=['accuracy']
    )
    return model

def myprint(s):
    with open('modelsummary.txt','a') as f:
        print(s, file=f)


# In[11]:


# List of optimizers to iterate over
for i in range(len(csv_names)):    
    # fetch datasets
    train = tf.data.Dataset.load(train_path)
    test  = tf.data.Dataset.load(test_path)
    train, test = format_set(train, test)

    # create model
    model = dnn_models[i]
    model = compile_model(model)
    model.build((None, 16, 16, 1))

    # Train
    history = model.fit(train, epochs=250, validation_data=test, callbacks=[time_callback])

    # Convert the history.history dict to a pandas DataFrame
    hist_df = pd.DataFrame(history.history)

    # Add epoch times
    hist_df['time'] = time_callback.times

    # Save model summary
    with open(summary_names[i], 'w') as f:
        for layer in model.layers:
            print(type(layer).__name__, file=f)

        # print params
        total_params         = model.count_params()
        trainable_params     = sum([tf.size(w_matrix).numpy() for w_matrix in model.trainable_weights])
        non_trainable_params = sum([tf.size(w_matrix).numpy() for w_matrix in model.non_trainable_weights])
        #optimizer_params     = sum([tf.size(w_matrix).numpy() for w_matrix in model.optimizer.weights])
        print(f'Total params: {total_params}', file=f)
        print(f'Trainable params: {trainable_params}', file=f)
        print(f'Non-trainable params: {non_trainable_params}', file=f)
        # print(f'Optimizer params: {optimizer_params} \n', file=f)
    
    # Save to csv
    hist_df.to_csv(csv_names[i])


# In[ ]:




