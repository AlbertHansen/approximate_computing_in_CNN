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

    

#%% testing area
import random

# Define the number of rows and columns
rows = 32
columns = 20

for i, batch in enumerate(train):
    if i != 0:
        continue
    
    # take first batch and save
    utils.my_csv.batch_to_csv(batch, 'forward_pass_test/batch_test')
    utils.my_csv.weights_to_csv(model, 'forward_pass_test')
    images, labels = batch
    labels_predicted = model(images)
    utils.my_csv.tensor_to_csv(labels_predicted, 'forward_pass_test/labels_predicted')
    
    #utils.train.iteration(model, batch)
    #utils.my_csv.weights_to_csv(model, 'after')



# %%
