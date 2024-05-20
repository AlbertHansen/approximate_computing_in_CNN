#%% Dependencies
import tensorflow as tf
import tqdm as tqdm
from NoisyLayers import * 

#%% functions

def test_noisy_conv2d_layer(iterations=200, no_filter=8, kernel_dims=(3, 3), activation_function='relu'):
    non_zero_count = 0

    for iteration in tqdm.tqdm(range(iterations)):
        # generate random input
        input = tf.random.uniform(shape=(3, 16, 20, 2))

        # create layers and sync
        layer = tf.keras.layers.Conv2D(filters=no_filter, kernel_size=kernel_dims, activation=activation_function, input_shape=(16, 20, 2), use_bias=False)
        noisy_layer = NoisyConv2D(filters=no_filter, kernel_size=kernel_dims, activation=activation_function, input_shape=(16, 20, 2))
        
        # Call the layers on dummy data to initialize the weights
        _ = layer(tf.zeros((3, 16, 20, 2)))
        _ = noisy_layer(tf.zeros((3, 16, 20, 2)))
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


def test_noisy_dense_layer(iterations=200, perceptons=8, activation_function='relu'):
    non_zero_count = 0

    for iteration in tqdm.tqdm(range(iterations)):

        # generate random input
        input = tf.random.uniform(shape=(1, 16))

        # create layers and sync
        layer = tf.keras.layers.Dense(units=perceptons, activation=activation_function, use_bias=False)
        noisy_layer = NoisyDense(units=perceptons, activation=activation_function)
        
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
    test_noisy_conv2d_layer(iterations=100, no_filter=8, kernel_dims=(3, 3), activation_function='sigmoid') 
    test_noisy_dense_layer(iterations=300, perceptons=50, activation_function='sigmoid')