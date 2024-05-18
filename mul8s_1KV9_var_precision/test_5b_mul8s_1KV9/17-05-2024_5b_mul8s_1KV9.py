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

lambda_value = 0.0002
model = models.Sequential([
    layers.Conv2D(40, (2, 2), activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(2, (2, 2), activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(40, (2, 2), activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
    layers.Flatten(),
    layers.Dense(40, activation='relu', bias_constraint=ZeroBias()),
    layers.Dense(num_classes, activation='relu', bias_constraint=ZeroBias(), kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),  # OBS!!! last layer will be changed to accommodate no of classes
])


model = utils.model_manipulation.compile_model(model)
model.build((None, 16, 16, 1))
model.summary()



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
    subprocess.check_call(['cp weights/train_images.csv weights/batch.csv'], shell=True)
    subprocess.check_call(['./AC_FF_5b_mul8s_1KV9'])

    acc = compare_max_indices('weights/train_labels.csv', 'weights/output.csv')
    print(f"From within evaluate_approx: acc = {acc}")

    # Call c++ network
    subprocess.check_call(['cp weights/test_images.csv weights/batch.csv'], shell=True)
    subprocess.check_call(['./AC_FF_5b_mul8s_1KV9'])

    acc_val = compare_max_indices('weights/test_labels.csv', 'weights/output.csv')
    print(f"From within evaluate_approx: acc_val = {acc_val}")
    
    return acc, acc_val

def find_max_weight_and_val():
    input = tf.ones(shape=(1, 16, 16, 1))

    max_values = []
    max_weights = []
    
    for i, layer in enumerate(model.layers):
        if i == 0:
            x = layer(input)
        else:
            x = layer(x)
        
        max_values.append(tf.reduce_max(x).numpy())
        try: 
            max_weights.append(tf.reduce_max(layer.get_weights()[0]).numpy())
        except IndexError:
            max_weights.append(0.0)

    max_values = np.array(max_values)
    max_weights = np.array(max_weights)

    print('Max value: ', max(max_values.flatten()))
    print('Max weight:', max(max_weights.flatten()))

#evaluate_approx()
#%%
for i in range(5):
    utils.my_csv.csv_to_weights(model, '2_kernels_45_epochs_start')
    subprocess.check_call(['cp -r 2_kernels_45_epochs_start/* weights/'], shell=True)

    with open(f'run_{i}_5b_mul8s_1KV9.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'accuracy', 'accuracy_val', 'time'])

        start_epoch = time.time()
        acc, acc_val = evaluate_approx()
        epoch_time = time.time() - start_epoch
        print(f'Accuracy: {acc}, Accuracy_val: {acc_val}, Time: {epoch_time}')
        writer.writerow([44, acc, acc_val, epoch_time])

        for i in range(10):
            print(f"----- Epoch {i+45} -----")

            start_epoch = time.time()
            utils.train.epoch_approx(model, train)     
            epoch_time = time.time() - start_epoch

            acc, acc_val = evaluate_approx()

            print("\n---------------------------------------------------------------------------\n")
            print(f'Accuracy: {acc}, Accuracy_val: {acc_val}, Time: {epoch_time}')
            find_max_weight_and_val()
            print("\n---------------------------------------------------------------------------\n")

            writer.writerow([i+45, acc, acc_val, epoch_time])
