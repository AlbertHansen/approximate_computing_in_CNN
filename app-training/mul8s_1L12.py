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

#%% Settings:
lambda_value = 0.0002
sgd_learning_rate = 0.00005
epochs = 1
weights_path = 'weights'
pretrained_weights_path = 'tensorflow_model_weights'
executable_path = 'AC_FF_mul8s_1L12'

#%% Model
class ZeroBias(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return tf.zeros_like(w)

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


# model = utils.model_manipulation.compile_model(model)
models = model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=sgd_learning_rate, momentum=0.0),
        loss=tf.keras.losses.BinaryFocalCrossentropy(),
        metrics=['accuracy']
    )

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
    # Call c++ network on train set
    subprocess.check_call([f'cp {weights_path}/train_images.csv {weights_path}/batch.csv'], shell=True)
    subprocess.check_call([f'./{executable_path}'], shell=True)

    acc = compare_max_indices(f'{weights_path}/train_labels.csv', f'{weights_path}/output.csv')

    # Call c++ network on test set
    subprocess.check_call([f'cp {weights_path}/test_images.csv {weights_path}/batch.csv'], shell=True)
    subprocess.check_call([f'./{executable_path}'], shell=True)

    acc_val = compare_max_indices(f'{weights_path}/test_labels.csv', f'{weights_path}/output.csv')
    
    return acc, acc_val

def find_best_start(model) -> str:
    # Find the best starting point
    best_start = ''
    best_acc = 0
    for i in range(50, 251, 50):
        print(f"----- Pretrained Epochs: {i} -----")
        subprocess.check_call([f'cp -r {pretrained_weights_path}/tf_model_weights_{i}/* {weights_path}/'], shell=True)
        acc, acc_val = evaluate_approx()
        print(f'Accuracy: {acc}, Accuracy_val: {acc_val}')
        if acc + acc_val > best_acc:
            best_acc = acc + acc_val
            best_start = f'tf_model_weights_{i}'

    print("Best starting point: ", best_start)
    return best_start


#evaluate_approx()
#%%
'''
with open('mul8s_1KV9_inference_on_weights_intervals.csv', 'w') as file:
    writer = csv.writer(file)

    writer.writerow(['pretrained_epochs', 'accuracy', 'accuracy_val', 'time'])  

    epochs = [50, 100, 150, 200, 250]
    for i in epochs:
        print(f"----- Pretrained Epochs: {i} -----")
        subprocess.check_call([f'cp -r tensorflow_model_weights/tf_model_weights_{i}/* weights0/'], shell=True)
        acc, acc_val = evaluate_approx()
        print(f'Accuracy: {acc}, Accuracy_val: {acc_val}')
        writer.writerow([i, acc, acc_val])

'''
# %%
find_best_start(model)