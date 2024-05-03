import tensorflow as tf

def conv2d_manual(inputs, kernel):
    # Get the height and width of the input tensor and the kernel
    input_shape = tf.shape(inputs)
    kernel_shape = tf.shape(kernel)
    input_shape_np = tf.get_static_value(input_shape)
    kernel_shape_np = tf.get_static_value(kernel_shape)
    input_height, input_width = input_shape_np[1], input_shape_np[2]
    kernel_height, kernel_width = kernel_shape_np[0], kernel_shape_np[1]

    # Calculate the dimensions of the output tensor
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1

    # Initialize the output tensor array
    output_array = tf.TensorArray(tf.float32, size=output_height*output_width)

    # Perform the convolution operation
    for i in range(output_height):
        for j in range(output_width):
            # Extract the current patch of the input tensor
            patch = tf.expand_dims(inputs[:, i:i+kernel_height, j:j+kernel_width, :], -1)
            # Perform element-wise multiplication between the patch and the kernel, and sum the result
            output_slice = tf.reduce_sum(patch * kernel, axis=[1, 2, 3])
            # Store the output slice in the tensor array
            output_array = output_array.write(i*output_width + j, output_slice)

    # Stack the output slices together to form the final output tensor
    output = tf.reshape(output_array.stack(), (input_shape[0], output_height, output_width, kernel_shape[3]))

    return output



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
        # output = tf.nn.conv2d(inputs, self.kernel, strides=[1, 1, 1, 1], padding='VALID')
        output = conv2d_manual(inputs, self.kernel)

        # Apply an activation function
        output = tf.nn.relu(output)

        return output
