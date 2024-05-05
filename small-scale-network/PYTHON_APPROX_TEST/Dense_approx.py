import tensorflow as tf
import numpy as np

def matmul(inputs, kernel):     # (32, 160), (160, 40) and (32, 40), (40, 10)
    # Initialize an empty list to store the result
    result = np.zeros((inputs.shape[0], kernel.shape[-1]))
    
    # Perform the matrix multiplication
    for i in range(inputs.shape[0]):
        for j in range(kernel.shape[0]):
            for k in range(kernel.shape[-1]):
                result[i][k] += inputs[i][j] * kernel[j][k]

    return result

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, **kwargs):
        super(MyDenseLayer, self).__init__(**kwargs)
        self.num_outputs = num_outputs

    def build(self, input_shape):
        # Initialize the weights with L2 regularization
        kernel_shape = (input_shape[-1], self.num_outputs)
        self.kernel = self.add_weight(name='kernel',
                                    shape=kernel_shape,
                                    initializer='glorot_uniform',
                                    regularizer=tf.keras.regularizers.l2(0.0002),
                                    trainable=True)

    def call(self, inputs):
        # Define the forward pass
        ''' THIS ACTUALLY WORKS!
        try:
            print("Using the approximation")
            output = matmul(inputs, self.kernel)
        except Exception as e:
            print(f"Error in the loop: \n\t{e}")
            output = tf.matmul(inputs, self.kernel)
        '''
        # Apply an activation function
        output = tf.matmul(inputs, self.kernel)
        output = tf.nn.relu(output)
        print(output.shape)

        return output

    

