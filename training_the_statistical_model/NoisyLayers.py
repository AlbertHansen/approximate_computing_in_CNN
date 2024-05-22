import tensorflow as tf
import numpy as np
import scipy.stats as stats
import time

def convolve_pmfs(pmf_dict, b_indexes) -> dict:
    """
    Convolve the probability mass functions (PMFs) for the given b-indexes.

    Parameters:
    pmf_dict (dict): A dictionary containing the PMFs for each b-index.
    b_indexes (list): A list of b-indexes for which the PMFs need to be convolved.

    Returns:
    dict: The combined PMF after convolving the PMFs for the given b-indexes.
    """
    # Initialize the combined PMF with the PMFs of the first two b-indexes
    combined_pmf = pmf_dict[b_indexes[0]].copy()
    
    # Convolve the PMFs for the remaining b-indexes
    for b_index in b_indexes[1:]:

        current_pmf = pmf_dict[b_index]
        
        # Convolve the current PMF with the combined PMF
        temp_combined_pmf = {}
        for error_sum, prob_sum in combined_pmf.items():
            for error, prob in current_pmf.items():
                new_error_sum = error_sum + error
                new_prob_sum = prob_sum * prob
                
                if new_error_sum in temp_combined_pmf:
                    temp_combined_pmf[new_error_sum] += new_prob_sum
                else:
                    temp_combined_pmf[new_error_sum] = new_prob_sum
        
        # Update the combined PMF with the convolved PMF
        combined_pmf = temp_combined_pmf
    
    return combined_pmf

def fit_pmfs(convolved_pmf) -> tuple:
    """
    Fits a probability mass function (PMF) to the given convolved PMF.

    Parameters:
    convolved_pmf (dict): A dictionary representing the convolved PMF, where the keys are error sums and the values are probability sums.

    Returns:
    tuple: A tuple containing the fitted distribution and its parameters.
    """

    samples = []
    for error_sum, prob_sum in convolved_pmf.items():
        num_samples = int(prob_sum * 1000000)  # Scale to get a sizable sample
        samples.extend([error_sum] * num_samples)
    samples = np.array(samples)

    # Fit each distribution to the sample data
    params = stats.norm.fit(samples)

    return stats.norm, params



class NoisyConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, error_pmfs, precision_bits=6, activation=None, kernel_regularizer=None, strides=(1, 1),  padding="valid", **kwargs):
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
        self.error_pmfs = error_pmfs
        self.precision_bits = precision_bits
        self.activation = tf.keras.activations.get(activation)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.strides = strides
        self.padding = padding
        self.filter_error_distributions = {}
        self.filter_error_params = {}

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
        try:
            self.calculate_filter_error_fit()
        except AttributeError as e:
            print(e)

    def call(self, inputs):
        """
        Performs the noisy convolution on the input tensor.

        Args:
            inputs (Tensor): The input tensor to be convolved.

        Returns:
            Tensor: The output tensor after applying the noisy convolution.
        """

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
            self.calculate_filter_error_fit()
            
            for filt in range(self.filters):
                # sample from distribution
                # filter_noise = self.filter_error_distributions[filt].rvs(*self.filter_error_params[filt], size=outputs.shape[0:-1]) / (2 ** self.precision_bits)
                filter_noise = self.filter_error_distributions[filt].rvs(*self.filter_error_params[filt], size=outputs.shape[0:-1]) / (2 ** (2 * self.precision_bits))
                # filter_noise = self.filter_error_distributions[filt].rvs(*self.filter_error_params[filt], size=outputs.shape[0:-1]) / (2 ** 13)
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

    def calculate_filter_error_fit(self):
        """
        Calculates the filter error fit for each filter in the layer.

        This method performs the following steps:
        1. Retrieves the weights for each perceptron in the filter.
        2. Flattens the weights to a 1D array.
        3. Scales the weights to integers using the specified precision bits.
        4. Shifts the weights to positive values.
        5. Quantizes the weights to integers.
        6. Fits the filter error probability mass function (PMF) to a distribution.
        7. Stores the fitted distribution and parameters for each filter.

        Returns:
            None
        """
        for filt in range(self.filters):
            weights = self.kernel[:, :, :, filt]            # Get weights for perceptron, kernel.shape = (kernel_height, kernel_width, nr_input_channels, nr_filters)
            weights = tf.reshape(weights, [-1])             # Flatten to 1D
            weights = weights * (2 ** 7)  # Scale to integer
            weights = weights + 128                         # Shift to positive
            weights = tf.cast(weights, tf.int32)            # Quantize

            # Fit to distribution
            filter_error_pmf = convolve_pmfs(self.error_pmfs, weights.numpy())
            distribution, fit_params = fit_pmfs(filter_error_pmf)
            
            '''
            # emulate 8 bits right-shift
            mean, variance = fit_params
            mean = mean / (2 ** self.precision_bits)
            variance = variance / (2 ** self.precision_bits)
            fit_params = (mean, variance)   
            '''      

            # Store distribution and fit parameters
            self.filter_error_distributions[filt] = distribution
            self.filter_error_params[filt] = fit_params

    def set_weights(self, weights):
            """
            Sets the weights of the layer. Overrides the base class method to recalculate the filter error fit.

            Args:
                weights (list): A list of numpy arrays representing the weights.

            Returns:
                None
            """
            super(NoisyConv2D, self).set_weights(weights)
            self.calculate_filter_error_fit()
            
    
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

    def __init__(self, units, error_pmfs, precision_bits=6, activation=None, kernel_regularizer=None, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.error_pmfs = error_pmfs
        self.precision_bits = precision_bits
        self.activation = tf.keras.activations.get(activation)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.perceptron_error_distributions = {}
        self.perceptron_error_params = {}

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
        '''No Bias
        self.bias = self.add_weight(name='bias', 
                                    shape=(self.units,),
                                    initializer='zeros',
                                    trainable=True)
        '''
        try:
            self.calculate_perceptron_error_fit()
        except AttributeError as e:
            print(e)

                                    
    def call(self, inputs):
        """
        Applies the layer operation to the input tensor.

        Args:
            inputs (tf.Tensor): Input tensor.

        Returns:
            tf.Tensor: Output tensor after applying the layer operation.

        """
        # perform dense operation
        outputs = tf.matmul(inputs, self.kernel) # + self.bias
        
        try:
            # Generate noise
            noise = np.zeros(shape=outputs.shape)
            self.calculate_perceptron_error_fit()
            
            for perceptron in range(self.units):

                # sample from distribution
                if self.perceptron_error_distributions:
                    # noise[:, perceptron] = self.perceptron_error_distributions[perceptron].rvs(*self.perceptron_error_params[perceptron], size=outputs.shape[0]) / (2 ** self.precision_bits)
                    noise[:, perceptron] = self.perceptron_error_distributions[perceptron].rvs(*self.perceptron_error_params[perceptron], size=outputs.shape[0]) / (2 ** (2 * self.precision_bits))
                    # noise[:, perceptron] = self.perceptron_error_distributions[perceptron].rvs(*self.perceptron_error_params[perceptron], size=outputs.shape[0]) / (2 ** 13)

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
            

    def calculate_perceptron_error_fit(self):
        """
        Calculates the error fit for each perceptron in the layer.

        This method performs the following steps:
        1. Retrieves the weights for each perceptron.
        2. Scales the weights to integers using the specified precision bits.
        3. Shifts the weights to positive values.
        4. Quantizes the weights to integers.
        5. Fits the perceptron error probability mass function (PMF) to a distribution.

        Returns:
            None
        """
        for perceptron in range(self.units):
            weights = self.kernel[:, perceptron]            # Get weights for perceptron, kernel.shape = (inputs, perceptrons)
            weights = weights * (2 ** 7)  # Scale to integer
            weights = weights + 128                         # Shift to positive
            weights = tf.cast(weights, tf.int32)            # Quantize

            # Fit to distribution
            perceptron_error_pmf = convolve_pmfs(self.error_pmfs, weights.numpy()) ############## HERE IT GOES WRONG ##################
            distribution, fit_params = fit_pmfs(perceptron_error_pmf)
            
            '''
            mean, variance = fit_params
            mean = mean / (2 ** self.precision_bits)
            variance = variance / (2 ** self.precision_bits)
            fit_params = (mean, variance)            
            '''
            
            # Store distribution and fit parameters
            self.perceptron_error_distributions[perceptron] = distribution
            self.perceptron_error_params[perceptron] = fit_params

    def set_weights(self, weights):
        """
        Sets the weights of the layer. Overrides the base class method to recalculate the filter error fit.

        Args:
            weights (list): A list of numpy arrays representing the weights.

        Returns:
            None
        """
        super(NoisyDense, self).set_weights(weights)
        self.calculate_perceptron_error_fit()

    def compute_output_shape(self):
        return (None, self.units)
    










############################################################################################################

''' --------- Fit PMFs to distributions ---------
# SHOULD BE USED, HOWEVER, IT IS VERY SLOW

# Evaluate goodness of fit
def evaluate_fit(distribution, params, data):
    # Generate a PDF from the fitted distribution
    pdf_fitted = distribution.pdf(data, *params[:-2], loc=params[-2], scale=params[-1])
    
    # Compute the K-S statistic
    D, p_value = stats.kstest(data, distribution.cdf, args=params)
    
    return D, p_value

def fit_pmfs(convolved_pmf) -> tuple:
    distributions = {
        'norm': stats.norm
    }

    samples = []
    for error_sum, prob_sum in convolved_pmf.items():
        num_samples = int(prob_sum * 1000000)  # Scale to get a sizable sample
        samples.extend([error_sum] * num_samples)
    samples = np.array(samples)

    # Fit each distribution to the sample data
    best_fit_p_value = 1
    best_fit_params = None  
    best_fit_distribution = None
    for distribution in distributions.values():
        params = distribution.fit(samples)

        start  = time.time()
        D, p_value = evaluate_fit(distribution, params, samples)
        del D
        
        if p_value < best_fit_p_value:
            best_fit_p_value = p_value
            best_fit_params  = params
            best_fit_distribution = distribution

    return best_fit_distribution, best_fit_params
'''