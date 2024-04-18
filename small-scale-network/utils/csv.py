import numpy as np

def tensor_to_csv(tensor, file):
    numpy_array = tensor.numpy()
    np.savetxt(file, numpy_array, delimiter=",")

def csv_to_tensor(path):
    print("csv")

def batch_to_csv(batch, path):
    batch = batch.numpy()    
    for i in range(batch.shape[0]):
        np.savetxt(f"{path}_{i}.csv", batch[i, :, :, 0], delimiter=",")