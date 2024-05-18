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
        outputs = tf.nn.conv2d(inputs,
                            self.kernel,
                            strides=[1, *self.strides, 1],
                            padding=self.padding.upper())
        
        if self.activation is not None:
            return self.activation(outputs)
        else:
            return outputs
        