# %%
# Import packages
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import subprocess
import csv
import tqdm
import time

from contextlib import redirect_stdout
from tensorflow.keras import datasets, layers, models

# import custom functions
import utils

#%% Datasets
# Classes
num_classes = 10
classes_to_keep = range(num_classes)

# which classes are left?
cifar100, info = tfds.load('cifar100', with_info=True)
label_names = info.features['label'].names
print(label_names[0:num_classes])

# import local datasets and preprocess them
train_path = '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_LANCZOS3/train'
test_path =  '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_LANCZOS3/test'
train, test = utils.dataset_manipulation.get_datasets(train_path, test_path, classes_to_keep)

#%% 
class ZeroBias(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return tf.zeros_like(w)
    
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()

#%%
lambda_values = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]

l1_paths = []
l2_paths = []
l1l2_paths = []
for lambda_value in lambda_values:
    l1_paths.append(f'runs/regularization/l1/lambda_{lambda_value}.csv')
    l2_paths.append(f'runs/regularization/l2/lambda_{lambda_value}.csv')
    l1l2_paths.append(f'runs/regularization/l1l2/lambda_{lambda_value}.csv')

for i, lambda_value in enumerate(lambda_values):
    model_l1 = models.Sequential([
        layers.Conv2D(40, (2, 2), activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l1(lambda_value)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(40, (2, 2), activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l1(lambda_value)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(40, (2, 2), activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l1(lambda_value)),
        layers.Flatten(),
        layers.Dense(40, activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l1(lambda_value)),
        layers.Dense(num_classes, activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l1(lambda_value)),  # OBS!!! last layer will be changed to accommodate no of classes
    ])
    model_l2 = models.Sequential([
        layers.Conv2D(40, (2, 2), activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(40, (2, 2), activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(40, (2, 2), activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
        layers.Flatten(),
        layers.Dense(40, activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
        layers.Dense(num_classes, activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),  # OBS!!! last layer will be changed to accommodate no of classes
    ])
    model_l1l2 = models.Sequential([
        layers.Conv2D(40, (2, 2), activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l1_l2(l1=lambda_value, l2=lambda_value)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(40, (2, 2), activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l1_l2(l1=lambda_value, l2=lambda_value)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(40, (2, 2), activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l1_l2(l1=lambda_value, l2=lambda_value)),
        layers.Flatten(),
        layers.Dense(40, activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l1_l2(l1=lambda_value, l2=lambda_value)),
        layers.Dense(num_classes, activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l1_l2(l1=lambda_value, l2=lambda_value)),  # OBS!!! last layer will be changed to accommodate no of classes
    ])

    models_reg = [model_l1, model_l2, model_l1l2]
    for j, model in enumerate(models_reg):
        model = utils.model_manipulation.compile_model(model)
        model.build((None, 16, 16, 1))

        history = model.fit(train, epochs=250, validation_data=test, callbacks=[time_callback])

        # Convert the history.history dict to a pandas DataFrame
        hist_df = pd.DataFrame(history.history)

        # Add epoch times
        hist_df['time'] = time_callback.times

        hist_df.to_csv()

        if j == 0:
            hist_df.to_csv(l1_paths[i])
        elif j == 1:
            hist_df.to_csv(l2_paths[i])
        elif j == 2:
            hist_df.to_csv(l1l2_paths[i])
        else:
            raise ValueError("Unknown regularization type.")
# %%
