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

# Keeping time
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()

class ZeroBias(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return tf.zeros_like(w)

lambda_value = 0.0002
model = models.Sequential([
    layers.Conv2D(40, (2, 2), activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(40, (2, 2), activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(40, (2, 2), activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
    layers.Flatten(),
    layers.Dense(40, activation='relu', bias_constraint=ZeroBias()),
    layers.Dense(num_classes, activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),  # OBS!!! last layer will be changed to accommodate no of classes
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
    print(f"From within evaluate_approx: acc = {acc}")

    # Call c++ network
    subprocess.check_call(['cp forward_pass_test/test_images.csv forward_pass_test/batch.csv'], shell=True)
    subprocess.check_call(['/home/ubuntu/approximate_computing_in_CNN/small-scale-network/AC_FF'])

    acc_val = compare_max_indices('forward_pass_test/test_labels.csv', 'forward_pass_test/output.csv')
    print(f"From within evaluate_approx: acc_val = {acc_val}")
    
    return acc, acc_val

#evaluate_approx()
#%%
# Train
history = model.fit(train, epochs=250, validation_data=test, callbacks=[time_callback])

# Convert the history.history dict to a pandas DataFrame
hist_df = pd.DataFrame(history.history)

# Add epoch times
hist_df['time'] = time_callback.times

hist_df.to_csv('runs/pre_train_250_epochs/tensorflow_model.csv')
utils.my_csv.weights_to_csv(model, 'runs/pre_train_250_epochs/tensorflow_model_weights')
    