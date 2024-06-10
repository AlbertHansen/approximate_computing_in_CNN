#%% Dependencies
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm as tqdm
import pandas as pd
import math
import matplotlib.pyplot as plt
import csv
from NoisyLayers import * 
from dataset_manipulation import *
# from test_custom_layers import * 
from collections import defaultdict
from tensorflow.keras import layers, models
from my_csv import weights_to_csv, csv_to_weights
from sklearn.metrics import accuracy_score
import subprocess


#%% Functions
def make_pmfs(filename: str) -> dict:

    # Read the CSV file
    data = pd.read_csv(filename, header=None)

    # Dictionary to store PMFs for each row (each b value)
    pmf_dict = {}

    # Iterate over each row to calculate and store the PMF
    for index, row in data.iterrows():
        error_distributions = row.values
        
        # Calculate unique values and frequencies
        unique_values, value_counts = np.unique(error_distributions, return_counts=True)
        freq = value_counts / len(error_distributions)
        
        # Combine unique values and frequencies into a list of tuples
        pmf = dict(zip(unique_values, freq))  # Store as dictionary for easier manipulation
        pmf_dict[index] = pmf

    return pmf_dict

#%% Custom training and evaluation functions
#def iteration(model, batch, labels_approximated):
def iteration(model, batch):
    """
    Perform one iteration of training.

    Args:
        model: The neural network model.
        batch: The batch of data containing images and labels.

    Returns:
        None
    """
    # unpack batch
    images, labels = batch

    # Use GradientTape() for auto differentiation, FORWARD PASS(ES)
    with tf.GradientTape() as tape:     # OBS! tape will not be destroyed when exiting this scope
        labels_predicted = model(images)
        loss_value       = model.compute_loss(images, labels, labels_predicted)

    # Perform gradient descent, BACKWARD PASS(ES)
    grads = tape.gradient(loss_value, model.trainable_weights)
    # print(grads)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

def epoch(model, dataset):
    """
    Perform one epoch of training.

    Args:
        model: The neural network model.
        dataset: The dataset to train on.

    Returns:
        None
    """
    # Iterate over each batch in the dataset
    for i, batch in enumerate(dataset):
        print(f'it {i}')
        iteration(model, batch)

def obscure_tensor(tensor):
    """
    Converts a TensorFlow tensor to a NumPy array and then converts it back to a TensorFlow tensor.

    Args:
        tensor (tf.Tensor): The input TensorFlow tensor.

    Returns:
        tf.Tensor: The converted TensorFlow tensor.

    """
    numpy_array = tensor.numpy()
    return tf.convert_to_tensor(numpy_array, dtype=tf.float32)


def noisy_iteration(noisy_model, model, batch):
    """
    Perform one iteration of training.

    Args:
        model: The neural network model.
        batch: The batch of data containing images and labels.

    Returns:
        None
    """
    # unpack batch
    images, labels = batch

    # noise model weights -> model weights
    for noisy_layer, layer in zip(noisy_model.layers, model.layers):
        layer.set_weights(noisy_layer.get_weights())

    # noisy predictions
    labels_noisy = noisy_model(images)

    # Use GradientTape() for auto differentiation, FORWARD PASS(ES)
    with tf.GradientTape() as tape:     # OBS! tape will not be destroyed when exiting this scope
        labels_predicted = model(images)
        diff = labels_predicted - labels_noisy
        diff = obscure_tensor(diff) # remove dependencies from weights to diff
        labels_predicted = labels_predicted - diff
        loss_value       = model.compute_loss(images, labels, labels_predicted)

    # Perform gradient descent, BACKWARD PASS(ES)
    grads = tape.gradient(loss_value, model.trainable_weights)
    # print(grads)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # model weights -> noisy model weights
    for noisy_layer, layer in zip(noisy_model.layers, model.layers):
        noisy_layer.set_weights(layer.get_weights())

def noisy_epoch(noisy_model, model, dataset):
    """
    Perform one epoch of training.

    Args:
        model: The neural network model.
        dataset: The dataset to train on.

    Returns:
        None
    """
    # Iterate over each batch in the dataset
    for i, batch in enumerate(tqdm.tqdm(dataset)):
        # print(f'noisy - it {i}')
        noisy_iteration(noisy_model, model, batch)

def debatch_dataset(dataset):
    """
    Debatch a dataset.

    Args:
        dataset (tf.data.Dataset): The dataset to debatch.

    Returns:
        list: A list of batches.
    """
    all_images = []
    all_labels = []

    for batch in dataset:
        images, labels = batch
        all_images.append(images)
        all_labels.append(labels)

    merged_images = np.concatenate(all_images, axis=0)
    merged_labels = np.concatenate(all_labels, axis=0)

    return merged_images, merged_labels

def evaluate_model(model, dataset):
    """
    Evaluate a TensorFlow Keras Sequential model on a dataset without using model.evaluate().

    Args:
        model (tf.keras.Sequential): The Keras Sequential model to evaluate.
        dataset (tf.data.Dataset): The dataset to evaluate the model on.

    Returns:
        float: The accuracy of the model on the dataset.
    """
    # Debatch the dataset
    merged_images, merged_labels = debatch_dataset(dataset)

    # Predict labels using the model
    predictions = model(merged_images)

    # Convert labels to one-hot encoding if needed
    if len(merged_labels.shape) == 1:
        merged_labels = tf.one_hot(merged_labels, depth=10)

    correct_predictions = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(merged_labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    print(f'Accuracy: {accuracy.numpy()}')
    return accuracy.numpy()

def flatten(lst):
    flattened_list = []
    for element in lst:
        if isinstance(element, list):
            flattened_list.extend(flatten(element))
        else:
            flattened_list.append(element)
    return flattened_list

##################### For the approximate model #########################
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

def evaluate_approx(executable):
    subprocess.check_call(['cp weights/train_images.csv weights/batch.csv'], shell=True)
    subprocess.check_call([f'./{executable}'])

    acc = compare_max_indices('weights/train_labels.csv', 'weights/output.csv')
    #print(f"From within evaluate_approx: acc = {acc}")

    # Call c++ network
    subprocess.check_call(['cp weights/test_images.csv weights/batch.csv'], shell=True)
    subprocess.check_call([f'./{executable}'])

    acc_val = compare_max_indices('weights/test_labels.csv', 'weights/output.csv')
    # print(f"From within evaluate_approx: acc_val = {acc_val}")
    
    return acc, acc_val
##################### For the approximate model #########################


def train_stat_model(num_classes, train, test, weight_path, pmfs):
    # instantiate model
    i = 6
    print(f"----------- {i} bits for precision -----------")
    lambda_value = 0.0002
    noisy_model = models.Sequential([
        NoisyConv2D(40, (2, 2), error_pmfs=pmfs, precision_bits=i, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
        layers.MaxPooling2D((2, 2)),
        NoisyConv2D(2, (2, 2), error_pmfs=pmfs, precision_bits=i, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
        layers.MaxPooling2D((2, 2)),
        NoisyConv2D(40, (2, 2), error_pmfs=pmfs, precision_bits=i, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
        layers.Flatten(),
        NoisyDense(40, error_pmfs=pmfs, precision_bits=i, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
        NoisyDense(num_classes, error_pmfs=pmfs, precision_bits=i, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),  # OBS!!! last layer will be changed to accommodate no of classes
    ])
    noisy_model.compile(
        optimizer='adamax',
        loss=tf.keras.losses.BinaryFocalCrossentropy(),
        metrics=['accuracy']
    )
    noisy_model.build((None, 16, 16, 1))

    model = models.Sequential([
        layers.Conv2D(40, (2, 2), use_bias=False, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(2, (2, 2), use_bias=False, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(40, (2, 2),  use_bias=False, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
        layers.Flatten(),
        layers.Dense(40,  use_bias=False, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
        layers.Dense(num_classes,  use_bias=False, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),  # OBS!!! last layer will be changed to accommodate no of classes
    ])
    model.compile(
        optimizer='adamax',
        loss=tf.keras.losses.BinaryFocalCrossentropy(),
        metrics=['accuracy']
    )
    model.build((None, 16, 16, 1))

    csv_to_weights(model, '2_kernels_45_epochs_start')
    csv_to_weights(noisy_model, '2_kernels_45_epochs_start')

    # TRAINING
    with open(f'{weight_path}_accuracy.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'accuracy', 'val_accuracy'])

        for i in range(41):
            noisy_epoch(noisy_model, model, train)
            accuracy = evaluate_model(noisy_model, train)
            accuracy_val = evaluate_model(noisy_model, test)
            
            writer.writerow([45 + i, accuracy, accuracy_val])

            if i % 5 == 0:
                weights_to_csv(noisy_model, f'{weight_path}/save_{i+45}')



def infer_on_approx(weight_path, executable):
    ###########################################################################
    # C++ model - With Approximate Multipliers
    ###########################################################################
    with open(f'{weight_path}_approx_accuracy.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'accuracy', 'val_accuracy'])

        for i in range(9):
            subprocess.check_call([f'cp -r {weight_path}/save_{45+i*5}/* weights/'], shell=True)
            try:
                accuracy, accuracy_val = evaluate_approx(executable)
                writer.writerow([45+i*5, accuracy, accuracy_val])
            except:
                writer.writerow([45+i*5, 'NaN', 'NaN'])





    # subprocess.check_call([f'cp input.csv weights/batch.csv'], shell=True)
    # subprocess.check_call([f'./{executable}'], shell=True)
    # subprocess.check_call([f'cp weights/output.csv approximate_predictions/{folder}/save_{epoch}.csv'], shell=True)


#%% Main
def main() -> None:
    # Classes
    num_classes = 10
    classes_to_keep = range(num_classes)

    # which classes are left?
    cifar100, info = tfds.load('cifar100', with_info=True)
    del cifar100
    label_names = info.features['label'].names
    print(label_names[0:num_classes])

    # import local datasets and preprocess them
    train_path = '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_LANCZOS3/train'
    test_path =  '/home/ubuntu/tensorflow_datasets/cifar100_grey_16x16_LANCZOS3/test'
    train, test = get_datasets(train_path, test_path, classes_to_keep, 32)

    # Error Distributions
    error_files = [
        'error_mul8s_1kv8.csv',
        'error_mul8s_1kv9.csv'
    ]
    error_pmfs = [make_pmfs(filename) for filename in error_files]

    weight_paths = [
        '1KV8_weights_stats',
        '1KV9_weights_stats'
    ]

    executables = [
        'AC_FF_6b_mul8s_1KV8',
        'AC_FF_6b_mul8s_1KV9'
    ]

    for weight_path, pmfs, executable in zip(weight_paths, error_pmfs, executables):
        train_stat_model(num_classes, train, test, weight_path, pmfs)
        infer_on_approx(weight_path, executable)

    


if __name__ == "__main__":
    main()
# %%
