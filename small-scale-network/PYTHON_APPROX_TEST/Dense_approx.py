import tensorflow as tf




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
        print(inputs)
        print(self.kernel)
        output = tf.matmul(inputs, self.kernel)

        # Apply an activation function
        output = tf.nn.relu(output)

        return output

    

