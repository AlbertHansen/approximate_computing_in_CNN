# %%
# Import packages
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import subprocess

from print_versions import print_versions
from contextlib import redirect_stdout
from tensorflow.keras import datasets, layers, models

# %%
# Keeping time
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()

# %%
# Classes
num_classes = 20

# remove classes 
classes_to_keep = range(num_classes)
def filter_fn(example):
    # Extract the label from the example
    label = example['label']  # replace 'label' with the actual key for the label in your dataset
    return tf.reduce_any(tf.equal(tf.cast(classes_to_keep, tf.int32), tf.cast(label, tf.int32)))

# which classes are left?
cifar100, info = tfds.load('cifar100', with_info=True)
label_names = info.features['label'].names
for i in classes_to_keep:
    print(label_names[i])

# %%
# transform labels to one-hot encoding
def preprocess(example):
    image = example['image']
    image.set_shape([16, 16, 1])
    label = example['label']
    label = tf.one_hot(label, depth=num_classes)  # One-hot encode the labels
    return image, label

# format set into batches
def format_set(set):
    # format and cache
    set_formatted = set.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    set_formatted = set_formatted.cache()
    set_formatted = set_formatted.batch(32)     # Changed from 512
    set_formatted = set_formatted.prefetch(tf.data.AUTOTUNE)
    return set_formatted

#%% C++ interactions

def tensor_to_csv(tensor, file):
    numpy_array = tensor.numpy()
    np.savetxt(file, numpy_array, delimiter=",")

def csv_to_tensor(path):
    print("csv")

#%%
# Datasets
train_path = '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_LANCZOS3/train'
test_path =  '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_LANCZOS3/test'

# fetch datasets
train = tf.data.Dataset.load(train_path)
test  = tf.data.Dataset.load(test_path)

# Use the filter function to remove examples not in the list
filtered_train = train.filter(filter_fn)
filtered_test = test.filter(filter_fn)

# format sets
train = format_set(filtered_train) 
test  = format_set(filtered_test)

#%%
# Model
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

def compile_model(model):
    model.compile(
        optimizer='adamax',
        loss=tf.keras.losses.BinaryFocalCrossentropy(),
        metrics=['accuracy']
    )
    return model

model = compile_model(model)
model.build((None, 16, 16, 1))
# model.summary()

# %%
# Train
# history = model.fit(train, epochs=50, validation_data=test, callbacks=[time_callback])

def iteration(model, batch, labels_approximated):

    # unpack batch
    images, labels = batch

    # Use GradientTape() for auto differentiation, FORWARD PASS(ES)
    with tf.GradientTape() as tape:     # OBS! tape will not be destroyed when exiting this scope
        labels_predicted = model(images)
        loss_value       = model.compute_loss(labels, labels_predicted)
    
    # Overwrite loss (from approx prediction)
    loss_value = model.compute_loss(labels, labels_approximated)

    # Perform gradient descent, BACKWARD PASS(ES)
    grads = tape.gradient(loss_value, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))


def epoch(model, dataset):
    for batch in dataset:
        iteration(model, batch, )
        

#%% test
# epoch(model, train)
batch = train
for image, label in batch:
    tensor_to_csv(image, 'test.txt')


# %%
# Save information

# Convert the history.history dict to a pandas DataFrame
hist_df = pd.DataFrame(history.history)

# Add epoch times
hist_df['time'] = time_callback.times

# Save model summary
with open(summary_names[i], 'w') as f:
    for layer in model.layers:
        print(type(layer).__name__, file=f)

    # print params
    total_params         = model.count_params()
    trainable_params     = sum([tf.size(w_matrix).numpy() for w_matrix in model.trainable_weights])
    non_trainable_params = sum([tf.size(w_matrix).numpy() for w_matrix in model.non_trainable_weights])
    #optimizer_params     = sum([tf.size(w_matrix).numpy() for w_matrix in model.optimizer.weights])
    print(f'Total params: {total_params}', file=f)
    print(f'Trainable params: {trainable_params}', file=f)
    print(f'Non-trainable params: {non_trainable_params}', file=f)
    # print(f'Optimizer params: {optimizer_params} \n', file=f)

# Save to csv
# hist_df.to_csv(csv_names[i])

print_versions(globals())


