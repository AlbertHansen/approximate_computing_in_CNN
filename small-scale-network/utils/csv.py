import numpy as np
import tensorflow as tf

def tensor_to_csv(tensor, file):
    """
    Converts a tensor to a CSV file.

    Args:
        tensor (torch.Tensor): The tensor to be converted.
        file (str): The file path to save the CSV file.

    Returns:
        None
    """
    numpy_array = tensor.numpy()
    np.savetxt(file, numpy_array, delimiter=",")

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
    batch = batch.numpy()    
    for i in range(batch.shape[0]):
        np.savetxt(f"{path}_{i}.csv", batch[i, :, :, 0], delimiter=",")

'''
def weights_to_csv(model, path):
    """
    Converts the weights of a model to CSV files and saves them to the specified path.

    Args:
        model (tf.keras.Model): The model whose weights will be converted to CSV files.
        path (str): The path where the CSV files will be saved.

    Returns:
        None
    """
    for i, layer in enumerate(model.layers):
        if layer.trainable:
            weights = layer.get_weights()
            print("-----------------")
            print(weights)
            #for j, weight in enumerate(weights):
            #    np.savetxt(f"{path}_layer_{i}_weight_{j}.csv", weight, delimiter=",")
'''