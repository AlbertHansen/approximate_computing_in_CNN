# %%
# Import packages
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import subprocess
import csv

from print_versions import print_versions
from contextlib import redirect_stdout
from tensorflow.keras import datasets, layers, models

# import custom functions
import utils

#%% Datasets
# Classes
num_classes = 20
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
    layers.Dense(20, activation='relu'),
    layers.Dense(num_classes, activation='relu'),       # OBS!!! last layer will be changed to accommodate no of classes
])

model = utils.model_manipulation.compile_model(model)
model.build((None, 16, 16, 1))
model.summary()

    
#%% test
def weights_to_csv(model, path):
    for i, layer in enumerate(model.layers):
        if layer.trainable:
            weights = layer.get_weights()
            if weights:  # Check if weights is not empty
                path_weight = f"{path}/layer_{i}/weights.csv"
                with open(path_weight, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(np.asarray(weights[0]))
                path_bias = f"{path}/layer_{i}/biases.csv"
                with open(path_bias, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(weights[1])


weights_to_csv(model, 'weights')


#%%

'''
for i, layer in enumerate(model.layers):
    print(f"---------- LAYER {i} ----------")
    if i == 0:
        weights = layer.get_weights()
        for l, weight in enumerate(weights):
            if l == 0:
                print(weight.shape)
                for k in range(weight.shape[2]):
                    for j in range(weight.shape[3]):
                        np.savetxt(f"weights/weight_{k}_{j}.csv", weight[:, :, k, j], delimiter=",")
        # fprint(weight)
    
    #biases = weights[1]
    #weights = weights[0]
    #print(biases)
    #print(weights)

# utils.csv.weights_to_csv(model, 'weights/weights')
# utils.train.epoch(model, train)
# acc = utils.train.evaluate_model(model, test)
'''


# %%
# Save information
# Train
# time_callback = TimeHistory()
# history = model.fit(train, epochs=50, validation_data=test, callbacks=[time_callback])


'''
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
'''
# %%
