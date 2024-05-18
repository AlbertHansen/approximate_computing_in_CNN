#%% Dependencies
import tensorflow as tf
from NoisyLayers import * 

#%% Generate input
input = tf.random(shape=(1, 16, 16, 1))

#%% Create layers
model_noisy = tf.keras.Sequential([
    NoisyConv2D(filters=8, kernel_size=(3, 3), activation='relu', input_shape=(16, 16, 1))
])

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', input_shape=(16, 16, 1), use_bias=False)
])
#%% Share weights
for layer, layer_noisy in zip(model.layers, model_noisy.layers):
    layer.set_weights(layer_noisy.get_weights())


#%% Compare outputs
output_noisy = model_noisy(input)
output = model(input)

