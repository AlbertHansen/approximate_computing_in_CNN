# %%
# Import packages
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import subprocess
import csv
import tqdm

from print_versions import print_versions
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
# model.summary()

# Train
accuracy     = []
accuracy_val = []
for i in range(100):
    utils.train.epoch(model, train)
    acc = utils.train.evaluate_model(model, train)
    acc_val = utils.train.evaluate_model(model, test)
    accuracy.append(acc)
    accuracy_val.append(acc_val)

with open('runs/gradient_test/eval.csv', 'w') as file:
    writer = csv.writer(file)
    for acc, acc_val in zip(accuracy, accuracy_val):
        writer.writerow([acc, acc_val])

# Save model summary
with open('runs/gradient_test/summary.txt', 'w') as f:
    for layer in model.layers:
        print(type(layer).__name__, file=f)

    # print params
    total_params         = model.count_params()
    trainable_params     = sum([tf.size(w_matrix).numpy() for w_matrix in model.trainable_weights])
    non_trainable_params = sum([tf.size(w_matrix).numpy() for w_matrix in model.non_trainable_weights])
    print(f'Total params: {total_params}', file=f)
    print(f'Trainable params: {trainable_params}', file=f)
    print(f'Non-trainable params: {non_trainable_params}', file=f)
