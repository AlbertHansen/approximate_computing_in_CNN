#%% Dependencies
import tensorflow as tf
import tqdm as tqdm
from NoisyLayers import * 
from convergence_of_stat_and_approx_model.mul8s_1kv9_stats_and_approx import make_pmfs

#%% functions

def test_noisy_conv2d_layer(pmfs, iterations=100, no_filter=8, kernel_dims=(3, 3), activation_function='relu'):
    non_zero_count = 0

    for iteration in tqdm.tqdm(range(iterations)):
        # generate random input
        input = tf.random.uniform(shape=(32, 16, 20, 2))

        # create layers and sync
        layer = tf.keras.layers.Conv2D(filters=no_filter, kernel_size=kernel_dims, activation=activation_function, input_shape=(16, 20, 2), use_bias=False)
        noisy_layer = NoisyConv2D(error_pmfs=pmfs, filters=no_filter, kernel_size=kernel_dims, activation=activation_function, input_shape=(16, 20, 2))
        
        # Call the layers on dummy data to initialize the weights
        _ = layer(tf.zeros((1, 16, 20, 2)))
        _ = noisy_layer(tf.zeros((1, 16, 20, 2)))
        noisy_layer.set_weights(layer.get_weights())

        # see the difference
        difference = layer(input) - noisy_layer(input)

        # Count non-zero elements
        non_zero_count += tf.math.count_nonzero(difference)

    # Check if there are any non-zero elements
    if non_zero_count > 0:
        print(f"Something went wrong. There are {non_zero_count} values over {iterations} iterations, which have been wrongly calculated.")
    else:
        print("Passed!")


def test_noisy_dense_layer(pmfs, iterations=100, perceptons=8, activation_function='relu'):
    non_zero_count = 0

    for iteration in tqdm.tqdm(range(iterations)):

        # generate random input
        input = tf.random.uniform(shape=(32, 16))

        # create layers and sync
        layer = tf.keras.layers.Dense(units=perceptons, activation=activation_function, use_bias=False)
        noisy_layer = NoisyDense(units=perceptons, error_pmfs=pmfs, activation=activation_function)
        
        # Call the layers on dummy data to initialize the weights
        _ = layer(tf.zeros((1, 16)))
        _ = noisy_layer(tf.zeros((1, 16)))
        noisy_layer.set_weights(layer.get_weights())

        # see the difference
        difference = layer(input) - noisy_layer(input)

        # Count non-zero elements
        non_zero_count += tf.math.count_nonzero(difference)

    # Check if there are any non-zero elements
    if non_zero_count > 0:
        print(f"Something went wrong. There are {non_zero_count} values over {iterations} iterations, which have been wrongly calculated.")
    else:
        print("Passed!")

#%% Main
def test_noisy_layers():
    pmfs = make_pmfs('256x256_zeros.csv')

    test_noisy_conv2d_layer(pmfs, iterations=20, no_filter=10, kernel_dims=(3, 3), activation_function='sigmoid') 
    test_noisy_dense_layer(pmfs, iterations=20, perceptons=10, activation_function='sigmoid')