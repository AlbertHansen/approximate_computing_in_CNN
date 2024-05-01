# %%
# Import packages
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import subprocess
import csv
import tqdm

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

#%% Functions
def evaluate_approx(dataset):
    path = "forward_pass_test/batch.csv"
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)

        for images, labels in dataset:

            # save all images in a file (to be processed by the c++ network)
            for i in range(images.shape[0]):            # lines in csv
                line = []
                for j in range(images.shape[-1]):       #
                    for k in range(images.shape[2]):    # columns
                        for l in range(images.shape[1]):# rows
                            line.append(images[i, l, k, j].numpy())
                writer.writerow(line)

            # save all


#%%
for i in range(5):
    for j, batch in enumerate(train):
        if j != 0:
            continue
        images, labels = batch

        utils.my_csv.weights_to_csv(model, f'runs/weight_increments_test/iteration_{i}')
        labels_approx, labels_predicted = utils.train.iteration_approx(model, batch)
        utils.my_csv.tensor_to_csv(images, f'runs/weight_increments_test/images_{i}')
        utils.my_csv.tensor_to_csv(labels, f'runs/weight_increments_test/labels_{i}')
        utils.my_csv.tensor_to_csv(labels_approx, f'runs/weight_increments_test/labels_approx_{i}')
        utils.my_csv.tensor_to_csv(labels_predicted, f'runs/weight_increments_test/labels_predicted_{i}')


#%%
'''
# Train
accuracy     = []
accuracy_val = []
for i in range(5):
    utils.train.epoch_approx(model, train)
    acc = utils.train.evaluate_model(model, train)
    acc_val = utils.train.evaluate_model(model, test)
    accuracy.append(acc)
    accuracy_val.append(acc_val)

with open('runs/diff_test/eval.csv', 'w') as file:
    writer = csv.writer(file)
    for acc, acc_val in zip(accuracy, accuracy_val):
        writer.writerow([acc, acc_val])

# Save model summary
with open('runs/diff_test/summary.txt', 'w') as f:
    for layer in model.layers:
        print(type(layer).__name__, file=f)

    # print params
    total_params         = model.count_params()
    trainable_params     = sum([tf.size(w_matrix).numpy() for w_matrix in model.trainable_weights])
    non_trainable_params = sum([tf.size(w_matrix).numpy() for w_matrix in model.non_trainable_weights])
    print(f'Total params: {total_params}', file=f)
    print(f'Trainable params: {trainable_params}', file=f)
    print(f'Non-trainable params: {non_trainable_params}', file=f)
'''
