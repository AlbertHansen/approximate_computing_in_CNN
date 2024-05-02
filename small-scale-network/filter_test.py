#%% Import
import tensorflow as tf
import numpy as np

#%% Process
# Create a 3x3x3 image
image = np.random.rand(1, 3, 3, 3).astype(np.float32)
#print(image)

# Create a placeholder for the image
input_image = tf.convert_to_tensor(image, dtype=tf.float32)
#print(input_image)

# Create a Conv2D layer
conv2d_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(2, 2), activation='relu')

# Call the Conv2D layer on some data to build it
_ = conv2d_layer(input_image)

# Set the weights of the Conv2D layer to all 1's
weights = [np.ones((2, 2, 3, 1), dtype=np.float32), np.full((1,), 0.5, dtype=np.float32)]
conv2d_layer.set_weights(weights)
# conv2d_layer.set_bias()

# Apply the Conv2D layer to the image
output = conv2d_layer(input_image)
#print(output)

#%% Verification

index = np.array([[
    np.sum(image[0, 0:2, 0:2, :]),
    np.sum(image[0, 0:2, 1:3, :])
], [
    np.sum(image[0, 1:3, 0:2, :]),
    np.sum(image[0, 1:3, 1:3, :])
]])
print(index)
print(output.numpy().reshape(2, 2))

# %%
