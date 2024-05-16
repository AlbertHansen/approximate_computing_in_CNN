import tensorflow as tf
import numpy as np
from utils import my_csv
from sklearn.metrics import accuracy_score
import subprocess
import tqdm
import csv

def iteration_gradient_test(model, batch):
    """
    Perform one iteration of approximate computing in a convolutional neural network.

    Args:
        model (tf.keras.Model): The CNN model.
        batch (tuple): A tuple containing the input images and corresponding labels.
        labels_approximated (tf.Tensor): The approximated labels.

    Returns:
        None
    """
    # unpack batch
    images, labels = batch

    # Use GradientTape() for auto differentiation, FORWARD PASS(ES)
    with tf.GradientTape() as tape:     # OBS! tape will not be destroyed when exiting this scope
        labels_predicted = model(images)
        diff             = gen_zeromean_tensor(labels_predicted.shape)
        labels_predicted = labels_predicted - diff
        loss_value       = model.compute_loss(images, labels, labels_predicted)

    # Perform gradient descent, BACKWARD PASS(ES)
    grads = tape.gradient(loss_value, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

def epoch_gradient_test(model, dataset):
    """
    Executes a single epoch of training on the given model using the provided dataset and the approximated network.

    Args:
        model (object): The model to train.
        dataset (object): The dataset to use for training.

    Returns:
        None
    """
    for batch in tqdm.tqdm(dataset):
        iteration_gradient_test(model, batch)

def iteration_approx(model, batch):
    """
    Perform one iteration of approximate computing in a convolutional neural network.

    Args:
        model (tf.keras.Model): The CNN model.
        batch (tuple): A tuple containing the input images and corresponding labels.
        labels_approximated (tf.Tensor): The approximated labels.

    Returns:
        None
    """
    # unpack batch
    images, labels = batch

    # save batch and weights
    my_csv.tensor_to_csv(images, 'weights/batch')
    my_csv.weights_to_csv(model, 'weights')

    # Call c++ network
    subprocess.check_call(['/home/ubuntu/approximate_computing_in_CNN/test_small_network/AC_FF_5b_multiply'])
    labels_approximated = my_csv.csv_to_tensor('weights/output.csv')

    # Use GradientTape() for auto differentiation, FORWARD PASS(ES)
    with tf.GradientTape() as tape:     # OBS! tape will not be destroyed when exiting this scope
        labels_predicted = model(images)
        labels_predicted_untouched = labels_predicted
        diff             = labels_predicted - labels_approximated
        diff             = obscure_tensor(diff) # remove dependencies from weights to diff
        labels_predicted = labels_predicted - diff
        loss_value       = model.compute_loss(images, labels, labels_predicted)

    # Perform gradient descent, BACKWARD PASS(ES)
    grads = tape.gradient(loss_value, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return labels_approximated, labels_predicted_untouched

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

    # Save weights, for approx model evaluation
    my_csv.weights_to_csv(model, 'weights')

    # Use GradientTape() for auto differentiation, FORWARD PASS(ES)
    with tf.GradientTape() as tape:     # OBS! tape will not be destroyed when exiting this scope
        labels_predicted = model(images)
        loss_value       = model.compute_loss(images, labels, labels_predicted)

    # Perform gradient descent, BACKWARD PASS(ES)
    grads = tape.gradient(loss_value, model.trainable_weights)
    # print(grads)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

def evaluate_model(model, dataset):
    """
    Evaluates the performance of a given model on a dataset.

    Args:
        model (tf.keras.Model): The model to be evaluated.
        dataset (tf.data.Dataset): The dataset to evaluate the model on.

    Returns:
        float: The accuracy of the model on the dataset.
    """
    # Initialize lists to store actual and predicted labels
    y_true = []
    y_pred = []

    # Iterate over each batch in the dataset
    for batch in dataset:
        images, labels = batch

        # Perform inference
        predictions = model(images)

        # Convert predictions to class labels
        predicted_labels = np.argmax(predictions, axis=1)

        # Convert one-hot encoded labels to class labels
        true_labels = np.argmax(labels.numpy(), axis=1)

        # Append to lists
        y_true.extend(true_labels)
        y_pred.extend(predicted_labels)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    print(f'Accuracy: {accuracy * 100:.2f}%')

    return accuracy

def epoch(model, dataset):
    """
    Executes a single epoch of training on the given model using the provided dataset.

    Args:
        model (object): The model to train.
        dataset (object): The dataset to use for training.

    Returns:
        None
    """
    for batch in tqdm.tqdm(dataset):
        iteration(model, batch)

def epoch_approx(model, dataset):
    """
    Executes a single epoch of training on the given model using the provided dataset and the approximated network.

    Args:
        model (object): The model to train.
        dataset (object): The dataset to use for training.

    Returns:
        None
    """
    for batch in tqdm.tqdm(dataset):
        iteration_approx(model, batch)
        #my_csv.tensor_to_csv(diff, f'runs/labels_test/labels_approx_{i}')

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

def gen_zeromean_tensor(shape):
    # Create a tensor with values from a normal distribution
    tensor = tf.random.normal(shape, stddev=0.01)

    # Subtract the mean to make the tensor have approximately zero mean
    tensor_zero_mean = tensor - tf.reduce_mean(tensor)

    return tensor_zero_mean
