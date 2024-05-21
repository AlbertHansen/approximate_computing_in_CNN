import tensorflow as tf
import numpy as np
import scipy.stats as stats
import time

class NoisyConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, mean, variance, precision_bits=6, activation=None, kernel_regularizer=None, strides=(1, 1),  padding="valid", **kwargs):
        """
        Custom layer for performing noisy convolution on 2D inputs.

        Args:
            filters (int): The number of output filters in the convolution.
            kernel_size (tuple): The size of the convolutional kernel.
            activation (str or callable): The activation function to use. If None, no activation is applied.
            kernel_regularizer (str or callable): The regularization function to use for the kernel weights. If None, no regularization is applied.
            strides (tuple): The strides of the convolution along the height and width.
            padding (str): The padding mode to use during convolution.

        Returns:
            Tensor: The output tensor after applying the noisy convolution.
        """
        super(NoisyConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.mean = mean
        self.variance = variance
        self.precision_bits = precision_bits
        self.activation = tf.keras.activations.get(activation)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        """
        Builds the layer by creating the trainable weight variable for the convolutional kernel.

        Args:
            input_shape (tuple): The shape of the input tensor.

        Returns:
            None
        """
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.filters),
                                      initializer='uniform',
                                      trainable=True)

    def call(self, inputs):
        """
        Performs the noisy convolution on the input tensor.

        Args:
            inputs (Tensor): The input tensor to be convolved.

        Returns:
            Tensor: The output tensor after applying the noisy convolution.
        """
        print("Call NoisyConv2D")

        # perform convolution
        outputs = tf.nn.conv2d( 
                inputs,
                self.kernel,    # size = (kernel_height, kernel_width, nr_input_channels, nr_filters)
                strides=[1, *self.strides, 1],
                padding=self.padding.upper()
        ) # size = (nr_images, height-kernel_height+1, width-kernel_width+1, nr_filters)

        try:
            # Generate noise
            noise = np.zeros(shape=outputs.shape)
            
            for filt in range(self.filters):
                # sample from distribution
                filter_noise = stats.norm.rvs(*(self.mean, self.variance), size=outputs.shape[0:-1]) / (2 ** self.precision_bits)
                noise[:, :, :, filt] = filter_noise

            # Add noise
            outputs = outputs + tf.convert_to_tensor(noise, dtype=tf.float32)
        except TypeError as e:
            print(e)
        except AttributeError as e:
            print(e)

        # Apply activation function
        if self.activation is not None:
            return self.activation(outputs)
        else:
            return outputs
            
    
class NoisyDense(tf.keras.layers.Layer):
    """
    Custom layer that applies noise to the outputs of a dense layer.

    Args:
        units (int): Number of units in the layer.
        error_pmfs (list): List of error probability mass functions (PMFs) for each weight.
        precision_bits (int, optional): Number of bits used for precision. Defaults to 6.
        activation (str or callable, optional): Activation function to use. Defaults to None.
        kernel_regularizer (str or callable, optional): Regularizer function applied to the kernel weights. Defaults to None.
        **kwargs: Additional keyword arguments passed to the base class.

    Attributes:
        units (int): Number of units in the layer.
        error_pmfs (list): List of error probability mass functions (PMFs) for each weight.
        precision_bits (int): Number of bits used for precision.
        activation (callable): Activation function to use.
        kernel_regularizer (callable): Regularizer function applied to the kernel weights.
        kernel (tf.Variable): Trainable weight variable for the layer.

    """

    def __init__(self, units, mean, variance, precision_bits=6, activation=None, kernel_regularizer=None, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.mean = mean
        self.variance = variance
        self.precision_bits = precision_bits
        self.activation = tf.keras.activations.get(activation)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        """
        Builds the layer by creating a trainable weight variable.

        Args:
            input_shape (tuple): Shape of the input tensor.

        """
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      trainable=True)

                                    
    def call(self, inputs):
        """
        Applies the layer operation to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor after applying the layer operation.

        """
        print("Call NoisyDense")
        # perform dense operation
        outputs = tf.matmul(inputs, self.kernel) # + self.bias
        
        try:
            # Generate noise
            noise = np.zeros(shape=outputs.shape)
            
            for perceptron in range(self.units):

                # sample from distribution
                if self.perceptron_error_distributions:
                    noise[:, perceptron] = stats.norm.rvs(*(self.mean, self.variance), size=outputs.shape[0]) / (2 ** self.precision_bits)

            # Add noise
            outputs = outputs + tf.convert_to_tensor(noise, dtype=tf.float32)

        except TypeError as e:
            print(e)
        except AttributeError as e:
            tf.print(e)

        # Apply activation function
        if self.activation is not None:
            return self.activation(outputs)
        else:
            return outputs

    def compute_output_shape(self):
        return (None, self.units)