#%% Dependencies
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm as tqdm
import pandas as pd
import csv
import subprocess
from NoisyLayers import * 
from dataset_manipulation import *
# from test_custom_layers import * 
from collections import defaultdict
from tensorflow.keras import layers, models
from my_csv import weights_to_csv, csv_to_weights, tensor_to_csv, csv_to_tensor
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
    for batch in dataset:
        iteration(model, batch)

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
    # installed_packages = [d.project_name for d in pkg_resources.working_set]
    # print_versions(globals())
    
    # settings
    pb = 6
    lambda_value = 0.0002
    number_of_images = 1000
    samplesize = 1000

    ###########################################################################
    # Generate random inputs for the models
    ###########################################################################
    # images = tf.random.uniform((number_of_images, 16, 16, 1))
    # tensor_to_csv(images, 'input')
    images = csv_to_tensor('input.csv')
    tensor_to_csv(images, 'read_input')


    ###########################################################################
    # Loop over weights
    ###########################################################################
    weight_folders = [
        '1KV8_weights',
        '1KV9_weights'
    ]
    approximate_multiplier_executables = [
        'AC_FF_6b_mul8s_1KV8',
        'AC_FF_6b_mul8s_1KV9'
    ]
    error_files = [
        'error_mul8s_1kv8.csv',
        'error_mul8s_1kv9.csv'
    ]
    error_pmfs = [make_pmfs(filename) for filename in error_files]

    epochs = [45, 50, 55, 60, 65, 70, 75, 80, 85]
    for folder, executable, pmfs in zip(weight_folders, approximate_multiplier_executables, error_pmfs):
        for epoch in epochs:
            print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(f'Folder: {folder}, Epoch: {epoch}')
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
            '''
            ###########################################################################
            # TensorFlow model - No noise
            ###########################################################################
            model = models.Sequential([
                layers.Conv2D(40, (2, 2), activation='relu', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(2, (2, 2), activation='relu', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(40, (2, 2), activation='relu', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
                layers.Flatten(),
                layers.Dense(40, activation='relu', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
                layers.Dense(10, activation='relu', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),  # OBS!!! last layer will be changed to accommodate no of classes
            ])

            model.compile(
                optimizer='adamax',
                loss=tf.keras.losses.BinaryFocalCrossentropy(),
                metrics=['accuracy']
            )

            model.build((None, 16, 16, 1))
            # model.summary()

            csv_to_weights(model, f'{folder}/save_{epoch}')
            predictions = model(images)
            tensor_to_csv(predictions, f'accurate_predictions/{folder}/save_{epoch}.csv')

            ###########################################################################
            # C++ model - With Approximate Multipliers
            ###########################################################################
            subprocess.check_call([f'cp -r {folder}/save_{epoch}/* weights/'], shell=True)
            subprocess.check_call([f'cp input.csv weights/batch.csv'], shell=True)
            subprocess.check_call([f'./{executable}'], shell=True)
            subprocess.check_call([f'cp weights/output.csv approximate_predictions/{folder}/save_{epoch}.csv'], shell=True)
            '''
            ###########################################################################
            # TensorFlow model - With Noise
            ###########################################################################    

            # create model
            model_noise = models.Sequential([
                NoisyConv2D(40, (2, 2), error_pmfs=pmfs, precision_bits=pb, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
                layers.MaxPooling2D((2, 2)),
                NoisyConv2D(2, (2, 2), error_pmfs=pmfs, precision_bits=pb, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
                layers.MaxPooling2D((2, 2)),
                NoisyConv2D(40, (2, 2), error_pmfs=pmfs, precision_bits=pb, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
                layers.Flatten(),
                NoisyDense(40, error_pmfs=pmfs, precision_bits=pb, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),
                NoisyDense(10, error_pmfs=pmfs, precision_bits=pb, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_value)),  # OBS!!! last layer will be changed to accommodate no of classes
            ])

            model_noise.compile(
                optimizer='adamax',
                loss=tf.keras.losses.BinaryFocalCrossentropy(),
                metrics=['accuracy']
            )

            model_noise.build((None, 16, 16, 1))
            # model_noise.summary()

            csv_to_weights(model_noise, f'{folder}/save_{epoch}')
            
            # loop over each images
            for i in range(number_of_images):
                print(f'image {i}')
                image = images[i, :, :, :]                      # original image in tensor, however, this removes a dimension 
                image = tf.expand_dims(image, axis=0)           # add the dimension back
                image = tf.tile(image, [samplesize, 1, 1, 1])   # repeat the image samplesize times

                predictions = model_noise(image)
                tensor_to_csv(predictions, f'statistical_predictions/{folder}/save_{epoch}/image_{i}')

if __name__ == "__main__":
    main()
# %%
