import tensorflow as tf

def filter_fn(classes_to_keep, example):
    """
    Filters examples based on the classes to keep.

    Args:
        classes_to_keep (list): A list of classes to keep.
        example (dict): An example containing a 'label' key.

    Returns:
        bool: True if the example's label is in the classes to keep, False otherwise.
    """
    # Extract the label from the example
    label = example['label']
    return tf.reduce_any(tf.equal(tf.cast(classes_to_keep, tf.int32), tf.cast(label, tf.int32)))

# transform labels to one-hot encoding
def preprocess(num_classes, example):
    """
    Preprocesses an example from the dataset. This function resizes the image to (16, 16, 1) and one-hot encodes the label.

    Args:
        num_classes (int): The number of classes in the dataset.
        example (dict): A dictionary containing the example data.

    Returns:
        tuple: A tuple containing the preprocessed image and label.

    Raises:
        None

    """
    image = example['image']
    image.set_shape([16, 16, 1])
    label = example['label']
    label = tf.one_hot(label, depth=num_classes)  # One-hot encode the labels
    return image, label

# format set into batches
def format_set(num_classes, set):
    """
    Formats the given dataset by preprocessing the examples, caching, batching, and prefetching.

    Args:
        num_classes (int): The number of classes in the dataset.
        set (tf.data.Dataset): The input dataset to be formatted.

    Returns:
        tf.data.Dataset: The formatted dataset.

    """
    # format and cache
    set_formatted = set.map(lambda example: preprocess(num_classes, example), num_parallel_calls=tf.data.AUTOTUNE)
    set_formatted = set_formatted.cache()
    set_formatted = set_formatted.batch(32)     # Changed from 512
    set_formatted = set_formatted.prefetch(tf.data.AUTOTUNE)
    return set_formatted

def get_datasets(train_path, test_path, classes_to_keep):
    """
    Load and filter datasets based on the provided class list.

    Args:
        train_path (str): The file path to the training dataset.
        test_path (str): The file path to the testing dataset.
        classes_to_keep (list): A list of classes to keep in the datasets.

    Returns:
        tuple: A tuple containing the filtered training and testing datasets.
    """
    # fetch datasets
    train = tf.data.Dataset.load(train_path)
    test  = tf.data.Dataset.load(test_path)

    # Use the filter function to remove examples not in the list
    train = train.filter(lambda example: filter_fn(classes_to_keep, example))
    test = test.filter(lambda example: filter_fn(classes_to_keep, example))

    # format sets
    train = format_set(len(classes_to_keep), train) 
    test  = format_set(len(classes_to_keep), test)

    return train, test