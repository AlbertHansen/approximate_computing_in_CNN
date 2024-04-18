import tensorflow as tf

def compile_model(model):
    """
    Compiles the given model with the specified optimizer, loss function, and metrics.

    Parameters:
        model (tf.keras.Model): The model to be compiled.

    Returns:
        tf.keras.Model: The compiled model.
    """
    model.compile(
        optimizer='adamax',
        loss=tf.keras.losses.BinaryFocalCrossentropy(),
        metrics=['accuracy']
    )
    return model