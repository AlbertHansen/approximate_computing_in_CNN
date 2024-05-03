import tensorflow as tf

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
        # Define the forward pass
        print(inputs)
        output = tf.nn.conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='VALID')

        # Apply an activation function
        output = tf.nn.relu(output)

        return output
