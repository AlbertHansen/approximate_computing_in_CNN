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

from print_versions import print_versions
from contextlib import redirect_stdout
from tensorflow.keras import datasets, layers, models

# import custom functions
import utils

#%%
# Keeping time
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()

summary_names = [
    'runs/classes_no_bias/5_classes_summary.txt',
    'runs/classes_no_bias/10_classes_summary.txt',
    'runs/classes_no_bias/15_classes_summary.txt',
    'runs/classes_no_bias/20_classes_summary.txt',
    'runs/classes_no_bias/25_classes_summary.txt',
    'runs/classes_no_bias/30_classes_summary.txt',
    'runs/classes_no_bias/35_classes_summary.txt',
    'runs/classes_no_bias/40_classes_summary.txt',
    'runs/classes_no_bias/45_classes_summary.txt',
    'runs/classes_no_bias/50_classes_summary.txt',
    'runs/classes_no_bias/55_classes_summary.txt',
    'runs/classes_no_bias/60_classes_summary.txt',
    'runs/classes_no_bias/65_classes_summary.txt',
    'runs/classes_no_bias/70_classes_summary.txt',
    'runs/classes_no_bias/75_classes_summary.txt',
    'runs/classes_no_bias/80_classes_summary.txt',
    'runs/classes_no_bias/85_classes_summary.txt',
    'runs/classes_no_bias/90_classes_summary.txt',
    'runs/classes_no_bias/95_classes_summary.txt',
    'runs/classes_no_bias/100_classes_summary.txt'
]

csv_names = [
    'runs/classes_no_bias/5_classes.csv',
    'runs/classes_no_bias/10_classes.csv',
    'runs/classes_no_bias/15_classes.csv',
    'runs/classes_no_bias/20_classes.csv',
    'runs/classes_no_bias/25_classes.csv',
    'runs/classes_no_bias/30_classes.csv',
    'runs/classes_no_bias/35_classes.csv',
    'runs/classes_no_bias/40_classes.csv',
    'runs/classes_no_bias/45_classes.csv',
    'runs/classes_no_bias/50_classes.csv',
    'runs/classes_no_bias/55_classes.csv',
    'runs/classes_no_bias/60_classes.csv',
    'runs/classes_no_bias/65_classes.csv',
    'runs/classes_no_bias/70_classes.csv',
    'runs/classes_no_bias/75_classes.csv',
    'runs/classes_no_bias/80_classes.csv',
    'runs/classes_no_bias/85_classes.csv',
    'runs/classes_no_bias/90_classes.csv',
    'runs/classes_no_bias/95_classes.csv',
    'runs/classes_no_bias/100_classes.csv'
]

#%% Datasets
class ZeroBias(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return tf.zeros_like(w)
# Classes
classes = list(range(5,101,5))
for i, number_of_classes in enumerate(classes):

    num_classes = number_of_classes
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

    # Train
    history = model.fit(train, epochs=250, validation_data=test, callbacks=[time_callback])

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
    hist_df.to_csv(csv_names[i])


    


# %%
'''
#%% testing area
    train_accs = []

    for i in range(25):
        utils.train.epoch_approx(model, train)
        acc = utils.train.evaluate_model(model, test)
        train_accs.append(acc)

    with open('forward_pass_test/acc.txt', 'w') as file:
        for acc in train_accs:
            file.write(f'{acc}\n')

for i, batch in enumerate(train):
    if i != 0:
        continue
    
    # take first batch and save
    # utils.my_csv.batch_to_csv(batch, 'forward_pass_test/batch_test')
    utils.my_csv.weights_to_csv(model, 'forward_pass_test')
    images, labels = batch
    image = images[0]
    utils.my_csv.tensor_to_csv(image, 'forward_pass_test/image')
    image = image[None, ...]    # add batch dimension
    for i, layer in enumerate(model.layers):
        if i == 0:
            x = layer(image)
        else:
            x = layer(x)
        print(f'For layer {i}: {x.shape}')
        utils.my_csv.tensor_to_csv(x, f'forward_pass_test/layer_{i}')
    
    #labels_predicted = model(images)
    #utils.my_csv.tensor_to_csv(labels_predicted, 'forward_pass_test/labels_predicted')
    
    #utils.train.iteration(model, batch)
    #utils.my_csv.weights_to_csv(model, 'after')
'''