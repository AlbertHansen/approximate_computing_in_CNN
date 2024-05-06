import tensorflow as tf
import numpy as np
import tqdm
from multiprocessing import Pool
import time
import test
import tensorflow.keras.backend as K

'''
def worker(args):
    i, inputs, kernel = args
    output = np.zeros((inputs.shape[1] - kernel.shape[0] + 1, inputs.shape[2] - kernel.shape[1] + 1, kernel.shape[-1]))
    for j in range(inputs.shape[1] - kernel.shape[0] + 1):      # Column
        for k in range(inputs.shape[2] - kernel.shape[1] + 1):  # Row
            for l in range(kernel.shape[-1]):                   # Filter
                for m in range(kernel.shape[0]):                # filter width
                    for n in range(kernel.shape[1]):            # filter height
                        for o in range(inputs.shape[-1]):       # input channels
                            output[j, k, l] += inputs[i, j+m, k+n, o] * kernel[m, n, o, l]
    return output

def worker(args):
    i, inputs, kernel = args
    inputs = np.array(inputs, dtype=np.float64)
    kernel = np.array(K.get_value(kernel), dtype=np.float64)
    output = test.worker(i, inputs, kernel)
    return output

    '''



    
def conv2d_manual(inputs, kernel):
    # Get the height and width of the input tensor and the kernel
    input_shape = tf.shape(inputs)
    kernel_shape = tf.shape(kernel)
    input_shape_np = tf.get_static_value(input_shape)
    kernel_shape_np = tf.get_static_value(kernel_shape)
    input_height, input_width = input_shape_np[1], input_shape_np[2]
    kernel_height, kernel_width = kernel_shape_np[0], kernel_shape_np[1]

    # Calculate the dimensions of the output tensor
    batch_size        = input_shape[0]
    output_height     = input_height - kernel_height + 1
    output_width      = input_width - kernel_width + 1
    filter_count      = kernel_shape[-1]
    output_shape      = [batch_size, output_height, output_width, filter_count]

    # Initialize the output tensor
    output = np.zeros((inputs.shape[0], inputs.shape[1] - kernel_height + 1, inputs.shape[2] - kernel_width + 1, kernel.shape[-1]), dtype=np.float32)

    '''
    # Perform the convolution
    with Pool() as p:
        results = p.map(worker, [(i, inputs, kernel) for i in range(inputs.shape[0])])

    # Combine results into a single output array
    output = np.stack(results)
    '''
    output = test.worker(inputs, kernel)

    return output

class MyConv2DLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, **kwargs):
        super(MyConv2DLayer, self).__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        # Create a weight variable with appropriate initialization
        kernel_shape = (self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.num_filters)
        self.kernel = self.add_weight(name='kernel', 
                                      shape=kernel_shape, 
                                      initializer='glorot_uniform', 
                                      regularizer=tf.keras.regularizers.l2(0.0002), 
                                      trainable=True)

    def call(self, inputs):
        start = time.time()

        # Define the forward pass
        try: 
            print("Using the approximation")
            output = conv2d_manual(inputs, self.kernel)
        except Exception as e:
            print(f"Error in the loop: \n\t{e}")
            output = tf.nn.conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='VALID')

        # Apply an activation function
        output = tf.nn.relu(output)

        end = time.time()
        print(f"Time taken: {end-start}")
        return output