import numpy as np
import tensorflow as tf
import csv

def tensor_to_csv(tensor, file):
    """
    Converts a tensor to a CSV file.

    Args:
        tensor (torch.Tensor): The tensor to be converted.
        file (str): The file path to save the CSV file.

    Returns:
        None
    """
    path = f"{file}.csv"
    print(tensor.shape)


    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        if len(tensor.shape) == 2:
            for i in range(tensor.shape[-1]):
                flat_tensor = tf.reshape(tensor[:, i], [-1])
                writer.writerow(flat_tensor.numpy())
        elif len(tensor.shape) == 4:
            for i in range(tensor.shape[-1]):
                flat_tensor = tf.reshape(tensor[:, :, :, i], [-1])
                writer.writerow(flat_tensor.numpy())
        else:
            print(f"The tensor has an unexpected shape: {tensor.shape}. Albert Fix Det!")

def csv_to_tensor(path):
    """
    Reads a CSV file and returns a tensor with dtype float32.

    Args:
        path (str): The file path of the CSV file.

    Returns:
        tf.Tensor: The tensor with dtype float32.
    """
    numpy_array = np.loadtxt(path, delimiter=",", dtype=np.float32)
    tensor = tf.convert_to_tensor(numpy_array, dtype=tf.float32)
    # tensor = tf.expand_dims(tensor, axis=0)
    return tensor

def batch_to_csv(batch, path):
    """
    Converts a batch of images to CSV files and saves them to the specified path.

    Args:
        batch (numpy.ndarray): The batch of images to convert to CSV files.
        path (str): The path where the CSV files will be saved.

    Returns:
        None
    """
    images, labels = batch

    # 1 CSV file per image
    #images = images.numpy()    
    #for i in range(batch.shape[0]):
    #    np.savetxt(f"{path}_{i}.csv", batch[i, :, :, 0], delimiter=",")
    images = images.numpy()

    # 1 csv file per batch
    path = f"{path}.csv"
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        for j in range(images.shape[0]):    # (batch_size, 16, 16, 1)
            writer.writerow(images[j, :, :, :].flatten())

def weights_to_csv(model, path):
    """
    Save the weights and biases of the trainable layers in the model to CSV files.

    Args:
        model (tf.keras.Model): The model whose weights and biases are to be saved.
        path (str): The path to the directory where the CSV files will be saved.

    Returns:
        None
    """
    for i, layer in enumerate(model.layers):
        if layer.trainable:
            weights = layer.get_weights()
            if not weights:  # Check if weights is not empty
                continue
            
            path_weight = f"{path}/layer_{i}/weights.csv"
            with open(path_weight, 'w', newline='') as file:
                writer = csv.writer(file)
                if len(weights[0].shape) == 4:
                    for j in range(weights[0].shape[-2]):
                        temp = weights[0][:, :, j, :]
                        writer.writerow(temp.flatten())
                elif len(weights[0].shape) == 2:
                    for j in range(weights[0].shape[-1]):
                        writer.writerow(weights[0][:, j])
                else:
                    print("The weights have an unexpected shape.")

            # Save the biases
            path_bias = f"{path}/layer_{i}/biases.csv"
            with open(path_bias, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(weights[1].flatten())

''' Original weights_to_csv (WORKS)
def weights_to_csv(model, path):
    """
    Save the weights and biases of the trainable layers in the model to CSV files.

    Args:
        model (tf.keras.Model): The model whose weights and biases are to be saved.
        path (str): The path to the directory where the CSV files will be saved.

    Returns:
        None
    """
    for i, layer in enumerate(model.layers):
        if layer.trainable:
            weights = layer.get_weights()
            if not weights:  # Check if weights is not empty
                continue
            
            path_weight = f"{path}/layer_{i}/weights.csv"
            with open(path_weight, 'w', newline='') as file:
                writer = csv.writer(file)
                if len(weights[0].shape) == 4:
                    for j in range(weights[0].shape[-1]):
                        temp = weights[0][:, :, :, j]
                        writer.writerow(temp.flatten())
                elif len(weights[0].shape) == 2:
                    for j in range(weights[0].shape[-1]):
                        writer.writerow(weights[0][:, j])
                else:
                    print("The weights have an unexpected shape.")

            # Save the biases
            path_bias = f"{path}/layer_{i}/biases.csv"
            with open(path_bias, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(weights[1].flatten())
'''
                
def get_approximate_predictions(path):
    """
    Reads the CSV file in the specified path and returns the approximated label predictions.

    Args:
        path (str): The path where the CSV files are saved.

    Returns:
        tf.Tensor: The approximate predictions.
    """
    predictions = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        temp = []
        for row in reader:
            temp.append([float(val) for val in row])
        predictions.append(temp)
    return tf.convert_to_tensor(predictions, dtype=tf.float32)