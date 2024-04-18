import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

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
        # diff = labels_predicted - labels_approximated # HVAD HVIS DEN SET AT labels_predicted BLIVER TRUKKET FRA labels_predicted?????????
        # labels_predicted = labels_predicted - diff
        loss_value       = model.compute_loss(images, labels, labels_predicted)

    # Perform gradient descent, BACKWARD PASS(ES)
    grads = tape.gradient(loss_value, model.trainable_weights)
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
    for batch in dataset:
        iteration(model, batch)