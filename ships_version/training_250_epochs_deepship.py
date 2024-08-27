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

import os
import scipy.io
import numpy as np

#%% Datasets

def load_mat_file(file_path):
    data = scipy.io.loadmat(file_path)  # This loads the .mat file
    return data['your_variable_name']   # Replace 'your_variable_name' with the actual key in your .mat file

def load_data_from_directory(base_dir):
    classes = sorted(os.listdir(base_dir))  # Get class labels from directory names
    data = []
    labels = []
    
    for label_index, class_name in enumerate(classes):
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith(".mat"):
                    file_path = os.path.join(class_dir, file_name)
                    data.append(load_mat_file(file_path))  # Load the .mat file
                    labels.append(label_index)  # Use the class index as the label

    return np.array(data), np.array(labels)

# Load training and testing data
train_data, train_labels = load_data_from_directory("/path/to/data/train")
test_data, test_labels = load_data_from_directory("/path/to/data/test")

train_data = train_data.astype(np.float32) / 255.0  # Normalize if needed
test_data = test_data.astype(np.float32) / 255.0    # Normalize if needed

train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels))

# Shuffle, batch, and prefetch the data
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

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

# Example using a global pooling layer
model = models.Sequential([
    layers.Input(shape=(None, None, 1)),  # None allows for variable dimensions
    layers.Conv2D(32, (3, 3), activation='relu',bias_constraint=ZeroBias()),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu',bias_constraint=ZeroBias()),
    layers.GlobalAveragePooling2D(),  # Global pooling reduces the spatial dimensions
    layers.Dense(64, activation='relu',bias_constraint=ZeroBias()),
    layers.Dense(4, activation='softmax',bias_constraint=ZeroBias())
])



model = utils.model_manipulation.compile_model(model)
model.build((None, 16, 16, 1))
# model.summary()

#evaluate_approx()
#%%

for i in range(1, 6):

    # Train
    history = model.fit(train, epochs=50, validation_data=test, callbacks=[time_callback])

    # Convert the history.history dict to a pandas DataFrame
    hist_df = pd.DataFrame(history.history)

    # Add epoch times
    hist_df['time'] = time_callback.times

    hist_df.to_csv(f'tensorflow_model_weights/tf_model_{i*50}.csv')
    utils.my_csv.weights_to_csv(model, f'tensorflow_model_weights/tf_model_weights_{i*50}')
    