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
        output = tf.matmul(inputs, self.kernel)

        # Apply an activation function
        output = tf.nn.relu(output)

        return output
    
    def conv2d_manual(inputs, kernel):
        # Get the height and width of the input tensor and the kernel
        input_height, input_width = inputs.shape[0], inputs.shape[1]
        kernel_height, kernel_width = kernel.shape[0], kernel.shape[1]

        # Calculate the dimensions of the output tensor
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1

        # Initialize the output tensor with zeros
        output = tf.zeros((output_height, output_width))

        # Perform the convolution operation
        for i in range(output_height):
            for j in range(output_width):
                # Extract the current patch of the input tensor
                patch = inputs[i:i+kernel_height, j:j+kernel_width]
                # Perform element-wise multiplication between the patch and the kernel, and sum the result
                output[i, j] = tf.reduce_sum(patch * kernel)

        # Apply the ReLU activation function
        output = tf.nn.relu(output)

        return output

