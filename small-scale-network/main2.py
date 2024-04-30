# %%
# Import packages
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import subprocess
import csv

from print_versions import print_versions
from contextlib import redirect_stdout
from tensorflow.keras import datasets, layers, models

# import custom functions
import utils

#%% Datasets
# Classes
num_classes = 100
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
model = models.Sequential([
    layers.Conv2D(40, (2, 2), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(40, (2, 2), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(40, (2, 2), activation='relu'),
    layers.Flatten(),
    layers.Dense(40, activation='relu'),
    layers.Dense(num_classes, activation='relu'),       # OBS!!! last layer will be changed to accommodate no of classes
])

model = utils.model_manipulation.compile_model(model)
model.build((None, 16, 16, 1))
model.summary()

    
for i in range(10):
    utils.train.epoch(model, train)
    accuracy = utils.train.evaluate(model, test)
    print(f'Accuracy: {accuracy * 100:.2f}%')

#%% testing area
