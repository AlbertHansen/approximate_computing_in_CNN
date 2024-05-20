import tensorflow as tf
import numpy as np

class NoisyConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation=None, kernel_regularizer=None, strides=(1, 1),  padding="valid", **kwargs):
        super(NoisyConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = tf.keras.activations.get(activation)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters),
                                      initializer='uniform',
                                      trainable=True)

    def call(self, inputs):
        # perform convolution
        outputs = tf.nn.conv2d(
                inputs,
                self.kernel,
                strides=[1, *self.strides, 1],
                padding=self.padding.upper()
        )
        
        # Add noise


        # Apply activation function
        if self.activation is not None:
            return self.activation(outputs)
        else:
            return outputs
        
class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units, precision_bits=6, activation=None, kernel_regularizer=None, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.precision_bits = precision_bits
        self.activation = tf.keras.activations.get(activation)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      trainable=True)
        '''No Bias
        self.bias = self.add_weight(name='bias', 
                                    shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)
        '''
                                    
    def call(self, inputs):
        # perform dense operation
        outputs = tf.matmul(inputs, self.kernel) # + self.bias
        
        # Add noise
            # kernel.shape = (inputs, perceptrons)
        for perceptron in range(self.units):
            weights = self.kernel[:, perceptron]
            print(weights.numpy())
            weights = weights * (2 ** self.precision_bits)
            weights = weights + 128                 # Shift to positive
            weights = tf.cast(weights, tf.int32)
            print(weights.numpy())

        # Apply activation function
        if self.activation is not None:
            return self.activation(outputs)
        else:
            return outputs