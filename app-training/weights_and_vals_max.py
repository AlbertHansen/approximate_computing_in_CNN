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
import utils.train_mul8s_1L12

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
epochs = 10
weights_path = 'weights'
pretrained_weights_paths = [
    'tensorflow_model_weights/tf_model_weights_50',
    'tensorflow_model_weights/tf_model_weights_100',
    'tensorflow_model_weights/tf_model_weights_150',
    'tensorflow_model_weights/tf_model_weights_200',
    'tensorflow_model_weights/tf_model_weights_250'
]
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
model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=sgd_learning_rate, momentum=0.0),
        loss=tf.keras.losses.BinaryFocalCrossentropy(),
        metrics=['accuracy']
    )

model.build((None, 16, 16, 1))
# model.summary()

#%%%%%%%%%%%%%%%%%%%%%%% Max Values %%%%%%%%%%%%%%%%%%%%%%%%

input = tf.ones(shape=(1, 16, 16, 1))

max_values = []
max_weights = []

for path in pretrained_weights_paths:
    utils.my_csv.csv_to_weights(model, path)
    
    layer_max_values = []
    layer_max_weights = []
    for i, layer in enumerate(model.layers):
        if i == 0:
            x = layer(input)
        else:
            x = layer(x)
        
        layer_max_values.append(tf.reduce_max(x).numpy())
        try: 
            layer_max_weights.append(tf.reduce_max(layer.get_weights()[0]).numpy())
        except IndexError:
            layer_max_weights.append(0.0)

    max_values.append(layer_max_values)
    max_weights.append(layer_max_weights)

max_values = np.array(max_values)
max_weights = np.array(max_weights)

for column in max_values:
    print(column)

for column in max_weights:
    print(column)


print('Max value: ', max(max_values.flatten()))
print('Max weight:', max(max_weights.flatten()))