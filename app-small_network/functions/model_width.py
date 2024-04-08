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
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/base_model.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n16_model.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n24_model.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n32_model.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n40_model.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n48_model.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n56_model.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n64_model.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n72_model.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n80_model.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n88_model.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n96_model.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n104_model.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n112_model.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n120_model.csv',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n128_model.csv'
]
summary_names = [
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/base_model.txt',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n16_model.txt',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n24_model.txt',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n32_model.txt',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n40_model.txt',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n48_model.txt',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n56_model.txt',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n64_model.txt',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n72_model.txt',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n80_model.txt',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n88_model.txt',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n96_model.txt',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n104_model.txt',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n112_model.txt',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n120_model.txt',
    '/home/ubuntu/approximate_computing_in_CNN/app-small_network/results/model_width/n128_model.txt'
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

n16_model = models.Sequential()
n16_model.add(layers.Conv2D(16, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
n16_model.add(layers.MaxPooling2D((2, 2)))
n16_model.add(layers.Conv2D(16, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n16_model.add(layers.MaxPooling2D((2, 2)))
n16_model.add(layers.Conv2D(16, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n16_model.add(layers.Flatten())
n16_model.add(layers.Dense(16, activation='relu'))
n16_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of 

n24_model = models.Sequential()
n24_model.add(layers.Conv2D(24, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
n24_model.add(layers.MaxPooling2D((2, 2)))
n24_model.add(layers.Conv2D(24, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n24_model.add(layers.MaxPooling2D((2, 2)))
n24_model.add(layers.Conv2D(24, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n24_model.add(layers.Flatten())
n24_model.add(layers.Dense(24, activation='relu'))
n24_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of 

n32_model = models.Sequential()
n32_model.add(layers.Conv2D(32, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
n32_model.add(layers.MaxPooling2D((2, 2)))
n32_model.add(layers.Conv2D(32, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n32_model.add(layers.MaxPooling2D((2, 2)))
n32_model.add(layers.Conv2D(32, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n32_model.add(layers.Flatten())
n32_model.add(layers.Dense(32, activation='relu'))
n32_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of 

n40_model = models.Sequential()
n40_model.add(layers.Conv2D(40, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
n40_model.add(layers.MaxPooling2D((2, 2)))
n40_model.add(layers.Conv2D(40, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n40_model.add(layers.MaxPooling2D((2, 2)))
n40_model.add(layers.Conv2D(40, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n40_model.add(layers.Flatten())
n40_model.add(layers.Dense(40, activation='relu'))
n40_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of classes

n48_model = models.Sequential()
n48_model.add(layers.Conv2D(48, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
n48_model.add(layers.MaxPooling2D((2, 2)))
n48_model.add(layers.Conv2D(48, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n48_model.add(layers.MaxPooling2D((2, 2)))
n48_model.add(layers.Conv2D(48, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n48_model.add(layers.Flatten())
n48_model.add(layers.Dense(48, activation='relu'))
n48_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of classes

n56_model = models.Sequential()
n56_model.add(layers.Conv2D(56, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
n56_model.add(layers.MaxPooling2D((2, 2)))
n56_model.add(layers.Conv2D(56, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n56_model.add(layers.MaxPooling2D((2, 2)))
n56_model.add(layers.Conv2D(56, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n56_model.add(layers.Flatten())
n56_model.add(layers.Dense(56, activation='relu'))
n56_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of classes

n64_model = models.Sequential()
n64_model.add(layers.Conv2D(64, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
n64_model.add(layers.MaxPooling2D((2, 2)))
n64_model.add(layers.Conv2D(64, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n64_model.add(layers.MaxPooling2D((2, 2)))
n64_model.add(layers.Conv2D(64, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n64_model.add(layers.Flatten())
n64_model.add(layers.Dense(64, activation='relu'))
n64_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of classes

n72_model = models.Sequential()
n72_model.add(layers.Conv2D(72, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
n72_model.add(layers.MaxPooling2D((2, 2)))
n72_model.add(layers.Conv2D(72, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n72_model.add(layers.MaxPooling2D((2, 2)))
n72_model.add(layers.Conv2D(72, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n72_model.add(layers.Flatten())
n72_model.add(layers.Dense(72, activation='relu'))
n72_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of classes

n80_model = models.Sequential()
n80_model.add(layers.Conv2D(80, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
n80_model.add(layers.MaxPooling2D((2, 2)))
n80_model.add(layers.Conv2D(80, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n80_model.add(layers.MaxPooling2D((2, 2)))
n80_model.add(layers.Conv2D(80, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n80_model.add(layers.Flatten())
n80_model.add(layers.Dense(80, activation='relu'))
n80_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of classes

n88_model = models.Sequential()
n88_model.add(layers.Conv2D(88, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
n88_model.add(layers.MaxPooling2D((2, 2)))
n88_model.add(layers.Conv2D(88, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n88_model.add(layers.MaxPooling2D((2, 2)))
n88_model.add(layers.Conv2D(88, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n88_model.add(layers.Flatten())
n88_model.add(layers.Dense(88, activation='relu'))
n88_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of classes

n96_model = models.Sequential()
n96_model.add(layers.Conv2D(96, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
n96_model.add(layers.MaxPooling2D((2, 2)))
n96_model.add(layers.Conv2D(96, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n96_model.add(layers.MaxPooling2D((2, 2)))
n96_model.add(layers.Conv2D(96, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n96_model.add(layers.Flatten())
n96_model.add(layers.Dense(96, activation='relu'))
n96_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of classes

n104_model = models.Sequential()
n104_model.add(layers.Conv2D(104, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
n104_model.add(layers.MaxPooling2D((2, 2)))
n104_model.add(layers.Conv2D(104, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n104_model.add(layers.MaxPooling2D((2, 2)))
n104_model.add(layers.Conv2D(104, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n104_model.add(layers.Flatten())
n104_model.add(layers.Dense(104, activation='relu'))
n104_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of classes

n112_model = models.Sequential()
n112_model.add(layers.Conv2D(112, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
n112_model.add(layers.MaxPooling2D((2, 2)))
n112_model.add(layers.Conv2D(112, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n112_model.add(layers.MaxPooling2D((2, 2)))
n112_model.add(layers.Conv2D(112, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n112_model.add(layers.Flatten())
n112_model.add(layers.Dense(112, activation='relu'))
n112_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of classes

n120_model = models.Sequential()
n120_model.add(layers.Conv2D(120, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
n120_model.add(layers.MaxPooling2D((2, 2)))
n120_model.add(layers.Conv2D(120, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n120_model.add(layers.MaxPooling2D((2, 2)))
n120_model.add(layers.Conv2D(120, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n120_model.add(layers.Flatten())
n120_model.add(layers.Dense(120, activation='relu'))
n120_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of classes

n128_model = models.Sequential()
n128_model.add(layers.Conv2D(128, (2, 2), activation='relu')) # input_shape removed, (3, 3) -> (2, 2)
n128_model.add(layers.MaxPooling2D((2, 2)))
n128_model.add(layers.Conv2D(128, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n128_model.add(layers.MaxPooling2D((2, 2)))
n128_model.add(layers.Conv2D(128, (2, 2), activation='relu')) # (3, 3) -> (2, 2)
n128_model.add(layers.Flatten())
n128_model.add(layers.Dense(128, activation='relu'))
n128_model.add(layers.Dense(100))  # changed from 10 to 100, due to amount of classes

dnn_models = [
    base_model,
    n16_model,
    n24_model,
    n32_model,
    n40_model,
    n48_model,
    n56_model,
    n64_model,
    n72_model,
    n80_model,
    n88_model,
    n96_model,
    n104_model,
    n112_model,
    n120_model,
    n128_model
]
    


# In[5]:


def compile_model(model):
    model.compile(
        optimizer='adamax',
        loss=tf.keras.losses.BinaryFocalCrossentropy(),
        metrics=['accuracy']
    )
    return model


# In[6]:


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




