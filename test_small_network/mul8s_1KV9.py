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
    subprocess.check_call(['cp weights2/train_images.csv weights2/batch.csv'], shell=True)
    subprocess.check_call(['/home/ubuntu/approximate_computing_in_CNN/app-training_approx_network/AC_FF_2'])

    acc = compare_max_indices('weights2/train_labels.csv', 'weights2/output.csv')
    print(f"From within evaluate_approx: acc = {acc}")

    # Call c++ network
    subprocess.check_call(['cp weights2/test_images.csv weights2/batch.csv'], shell=True)
    subprocess.check_call(['/home/ubuntu/approximate_computing_in_CNN/app-training_approx_network/AC_FF_2'])

    acc_val = compare_max_indices('weights2/test_labels.csv', 'weights2/output.csv')
    print(f"From within evaluate_approx: acc_val = {acc_val}")
    
    return acc, acc_val

#evaluate_approx()
#%%
with open('mul8s_1KV9.csv', 'w') as file:
    writer = csv.writer(file)

    writer.writerow(['accuracy', 'accuracy_val', 'time'])

    # 5 training epochs with approximate training (STE)
    for i in range(50):
        print(f"----- Epoch {i} -----")
        start_epoch = time.time()
        
        utils.train_2.epoch_approx(model, train)
        acc, acc_val = evaluate_approx()
        epoch_time = time.time() - start_epoch
        print(f'Accuracy: {acc}, Accuracy_val: {acc_val}, Time: {epoch_time}')
        writer.writerow([acc, acc_val, epoch_time])
# %%
'''
157it [20:23,  6.10s/it]2024-05-10 04:38:14.854164: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
157it [20:23,  7.79s/it]
Matches: 513/5000
From within evaluate_approx: acc = 0.1026
Matches: 110/1000
From within evaluate_approx: acc_val = 0.11
Accuracy: 0.1026, Accuracy_val: 0.11, Time: 2430.3744671344757
----- Epoch 16 -----
157it [20:24,  6.09s/it]2024-05-10 05:18:46.130008: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
157it [20:24,  7.80s/it]
Matches: 496/5000
From within evaluate_approx: acc = 0.0992
Matches: 95/1000
From within evaluate_approx: acc_val = 0.095
Accuracy: 0.0992, Accuracy_val: 0.095, Time: 2430.854722261429
----- Epoch 17 -----
157it [20:24,  6.09s/it]2024-05-10 05:59:16.799054: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
157it [20:24,  7.80s/it]
Matches: 516/5000
From within evaluate_approx: acc = 0.1032
Matches: 93/1000
From within evaluate_approx: acc_val = 0.093
Accuracy: 0.1032, Accuracy_val: 0.093, Time: 2427.411835193634
----- Epoch 18 -----
157it [20:22,  6.07s/it]2024-05-10 06:39:42.911299: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
157it [20:22,  7.79s/it]
Matches: 513/5000
From within evaluate_approx: acc = 0.1026
Matches: 104/1000
From within evaluate_approx: acc_val = 0.104
Accuracy: 0.1026, Accuracy_val: 0.104, Time: 2426.7044241428375
----- Epoch 19 -----
157it [20:25,  6.09s/it]2024-05-10 07:20:12.241278: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
157it [20:25,  7.81s/it]
Matches: 483/5000
From within evaluate_approx: acc = 0.0966
Matches: 100/1000
From within evaluate_approx: acc_val = 0.1
Accuracy: 0.0966, Accuracy_val: 0.1, Time: 2430.4040961265564
----- Epoch 20 -----
157it [20:26,  6.10s/it]2024-05-10 08:00:43.410502: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
157it [20:26,  7.81s/it]
Matches: 539/5000
From within evaluate_approx: acc = 0.1078
Matches: 114/1000
From within evaluate_approx: acc_val = 0.114
Accuracy: 0.1078, Accuracy_val: 0.114, Time: 2430.5839323997498
----- Epoch 21 -----
157it [20:29,  6.13s/it]2024-05-10 08:41:17.497624: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
157it [20:29,  7.83s/it]
'''