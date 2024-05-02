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

#%% Model
class ZeroBias(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return tf.zeros_like(w)

model = models.Sequential([
    layers.Conv2D(40, (2, 2), activation='relu', bias_constraint=ZeroBias()),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(40, (2, 2), activation='relu', bias_constraint=ZeroBias()),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(40, (2, 2), activation='relu', bias_constraint=ZeroBias()),
    layers.Flatten(),
    layers.Dense(40, activation='relu', bias_constraint=ZeroBias()),
    layers.Dense(num_classes, activation='relu', bias_constraint=ZeroBias()),  # OBS!!! last layer will be changed to accommodate no of classes
])


model = utils.model_manipulation.compile_model(model)
model.build((None, 16, 16, 1))
# model.summary()

#%%
def compare_max_indices(file1, file2):
    df1 = pd.read_csv(file1, header=None)
    df2 = pd.read_csv(file2, header=None)

    if df1.shape != df2.shape:
        print('Shapes do not match')
        return
    
    max_indices_df1 = df1.idxmax(axis=1)
    max_indices_df2 = df2.idxmax(axis=1)

    matches = sum(max_indices_df1 == max_indices_df2)

    print(f'Matches: {matches}/{df1.shape[0]}')

    return matches/df1.shape[0]


def evaluate_approx():
    subprocess.check_call(['cp forward_pass_test/train_images.csv forward_pass_test/batch.csv'], shell=True)
    subprocess.check_call(['/home/ubuntu/approximate_computing_in_CNN/small-scale-network/AC_FF'])

    acc = compare_max_indices('forward_pass_test/train_labels.csv', 'forward_pass_test/output.csv')

    # Call c++ network
    subprocess.check_call(['cp forward_pass_test/test_images.csv forward_pass_test/batch.csv'], shell=True)
    subprocess.check_call(['/home/ubuntu/approximate_computing_in_CNN/small-scale-network/AC_FF'])

    acc_val = compare_max_indices('forward_pass_test/test_labels.csv', 'forward_pass_test/output.csv')
    
    return acc, acc_val

#evaluate_approx()
#%%
for i in range(5):
    print(f"----- Epoch {i} -----")
    utils.train.epoch_approx(model, train)
    acc, acc_val = evaluate_approx()
    print(f'Accuracy: {acc}, Accuracy_val: {acc_val}')


#%%
'''
for i, batch in enumerate(train):
    if i != 0:
        continue
    
    # take first batch and save
    # utils.my_csv.batch_to_csv(batch, 'forward_pass_test/batch_test')
    




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
