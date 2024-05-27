#%% Dependencies
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm as tqdm
import pandas as pd
import math
import matplotlib.pyplot as plt
import csv
import time as time
from NoisyLayers import * 
from dataset_manipulation import *
# from test_custom_layers import * 
from collections import defaultdict
from tensorflow.keras import layers, models
from my_csv import weights_to_csv, csv_to_weights
from sklearn.metrics import accuracy_score


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



#%% Main
def main() -> None:
        
    # Classes
    num_classes = 10
    classes_to_keep = range(num_classes)

    # which classes are left?
    cifar10, info = tfds.load('cifar10', with_info=True)
    del cifar10
    label_names = info.features['label'].names
    print(label_names[0:num_classes])

    # import local datasets and preprocess them
    train_path = '/home/ubuntu/tensorflow_datasets/cifar10/train'
    test_path =  '/home/ubuntu/tensorflow_datasets/cifar10/test'
    train, test = get_datasets(train_path, test_path, classes_to_keep, 256)

    
    # Error Distributions
    error_file = 'error_mul8s_1kv8.csv'
    pmfs = make_pmfs(error_file)

    # instantiate model, without bias...
    model = models.Sequential([
            layers.Conv2D(32, (3, 3), use_bias=False, activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), use_bias=False, activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), use_bias=False, activation='relu'),
            layers.Flatten(),
            layers.Dense(64, use_bias=False, activation='relu'),
            layers.Dense(10, use_bias=False)
    ])
    model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
    )
    model.build((None, 32, 32, 3))
    model.summary()

    history = model.fit(train, epochs=100, validation_data=test)


    '''
    # instantiate noisy model
    start = time.time()
    noisy_model = models.Sequential([
            NoisyConv2D(32, (3, 3), error_pmfs=pmfs, precision_bits=6, activation='relu'),
            layers.MaxPooling2D((2, 2)),
            NoisyConv2D(64, (3, 3), error_pmfs=pmfs, precision_bits=6, activation='relu'),
            layers.MaxPooling2D((2, 2)),
            NoisyConv2D(64, (3, 3), error_pmfs=pmfs, precision_bits=6, activation='relu'),
            layers.Flatten(),
            NoisyDense(64, error_pmfs=pmfs, precision_bits=6, activation='relu'),
            NoisyDense(10, error_pmfs=pmfs, precision_bits=6)
    ])
    noisy_model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
    )
    noisy_model.build((None, 32, 32, 3))
    noisy_model.summary()
    print(f'Building Noisy Model took - {time.time() - start} s')

    
    # Train the model
    history = model.fit(train, epochs=1, validation_data=test)
    noisy_model.set_weights(model.get_weights())

    # save history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('95_model.csv', index=False)

    # last 5 epochs
    with open('5_model.csv', 'w') as file, open('5_noisy_model.csv', 'w') as file2:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Accuracy', 'Accuracy Val'])

        writer2 = csv.writer(file2)
        writer2.writerow(['Epoch', 'Accuracy', 'Accuracy Val'])

        for j in range(1):
            print(f'----- Epoch {95 + j} -----')
            noisy_epoch(noisy_model, model, train)
            accuracy = evaluate_model(model, train)
            accuracy_val = evaluate_model(model, test)
            writer.writerow([95 + j, accuracy, accuracy_val])

            for k in range(25):
                noisy_accuracy = evaluate_model(noisy_model, train)
                noisy_accuracy_val = evaluate_model(noisy_model, test)
                writer.writerow([95 + j, noisy_accuracy, noisy_accuracy_val])
    '''
                
if __name__ == "__main__":
    main()
# %%
