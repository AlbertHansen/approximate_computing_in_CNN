# %%
# Import packages
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import subprocess
import csv
import time
from tensorflow.keras import datasets, layers, models

# import custom functions
import utils
import Conv2D_approx
import Dense_approx

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

#%% Model
lambda_value = 0.0002
model = models.Sequential([
    layers.Conv2D(40, (2, 2), activation='relu', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(40, (2, 2), activation='relu', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(40, (2, 2), activation='relu', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
    layers.Flatten(),
    layers.Dense(40, activation='relu', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
    layers.Dense(num_classes, activation='relu', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),  # OBS!!! last layer will be changed to accommodate no of classes
])
model_approx = models.Sequential([
    Conv2D_approx.MyConv2DLayer(num_filters=40, kernel_size=(2, 2)),
    layers.MaxPooling2D((2, 2)),
    Conv2D_approx.MyConv2DLayer(num_filters=40, kernel_size=(2, 2)),
    layers.MaxPooling2D((2, 2)),
    Conv2D_approx.MyConv2DLayer(num_filters=40, kernel_size=(2, 2)),
    layers.Flatten(),
    Dense_approx.MyDenseLayer(num_outputs=40),
    Dense_approx.MyDenseLayer(num_outputs=num_classes),  # OBS!!! last layer will be changed to accommodate no of classes
])

model = utils.model_manipulation.compile_model(model)
model_approx = utils.model_manipulation.compile_model(model_approx)
model.build((None, 16, 16, 1))
model_approx.build((32, 16, 16, 1))
model.summary()
model_approx.summary()

# Copy weights from exact model to approximate model
weights = model.get_weights()
model_approx.set_weights(weights)

#evaluate_approx()
#%%

for i, batch in enumerate(train):
    if i != 0:
        break
    x, y = batch
    y_predicted    = model(x)
    y_approximated = model_approx(x)
    utils.my_csv.tensor_to_csv(y_predicted, 'y_predicted.csv')
    utils.my_csv.tensor_to_csv(y_approximated, 'y_approximated.csv')
