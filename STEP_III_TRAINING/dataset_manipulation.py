import tensorflow as tf
import tqdm as tqdm

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

# Transform labels to one-hot encoding
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
    image.set_shape([32, 32, 3])
    label = example['label']
    # label = tf.one_hot(label, depth=num_classes)  # One-hot encode the labels
    return image, label

# Format set into batches
def format_set(num_classes, dataset, batch_size):
    """
    Formats the given dataset by preprocessing the examples, caching, batching, and prefetching.

    Args:
        num_classes (int): The number of classes in the dataset.
        dataset (tf.data.Dataset): The input dataset to be formatted.
        batch_size (int): The size of the batches.

    Returns:
        tf.data.Dataset: The formatted dataset.

    """
    # Format and cache
    dataset_formatted = dataset.map(lambda example: preprocess(num_classes, example), num_parallel_calls=tf.data.AUTOTUNE)
    dataset_formatted = dataset_formatted.cache()
    dataset_formatted = dataset_formatted.batch(batch_size, drop_remainder=True)  # Ensure batch size consistency
    dataset_formatted = dataset_formatted.prefetch(tf.data.AUTOTUNE)
    return dataset_formatted

def get_datasets(train_path, test_path, classes_to_keep, batch_size):
    """
    Load and filter datasets based on the provided class list.

    Args:
        train_path (str): The file path to the training dataset.
        test_path (str): The file path to the testing dataset.
        classes_to_keep (list): A list of classes to keep in the datasets.
        batch_size (int): The size of the batches.

    Returns:
        tuple: A tuple containing the filtered training and testing datasets.
    """
    # Fetch datasets
    train = tf.data.Dataset.load(train_path)
    test = tf.data.Dataset.load(test_path)

    # Use the filter function to remove examples not in the list
    # train = train.filter(lambda example: filter_fn(classes_to_keep, example))
    # test = test.filter(lambda example: filter_fn(classes_to_keep, example))

    # Format sets
    train = format_set(len(classes_to_keep), train, batch_size)
    test = format_set(len(classes_to_keep), test, batch_size)

    return train, test

def get_datasets_no_batches(train_path, test_path, classes_to_keep):
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
    
    return train, test

def create_eval_sets(no_classes):
    train_eval, test_eval = utils.dataset_manipulation.get_datasets_no_batches(train_path, test_path, classes_to_keep)
    with open('forward_pass_test/train_images.csv', 'w') as image_file:
        writer_image = csv.writer(image_file)
        for batch in tqdm.tqdm(train_eval):
            image = batch['image']
            
            line = []
            for column in range(image.shape[1]):
                for row in range(image.shape[0]):
                    line.append(image[row, column, 0].numpy())
            writer_image.writerow(line)
    
        
    with open('forward_pass_test/train_labels.csv', 'w') as label_file:
        writer_label = csv.writer(label_file)
        for i, batch in enumerate(tqdm.tqdm(train_eval)):
            label = batch['label']
            label = tf.one_hot(label, no_classes)
            line = []
            for row in range(label.shape[0]):
                line.append(label[row].numpy())
            writer_label.writerow(line)
    
    with open('forward_pass_test/test_images.csv', 'w') as image_file:
        writer_image = csv.writer(image_file)
        for batch in tqdm.tqdm(test_eval):
            image = batch['image']
            
            line = []
            for column in range(image.shape[1]):
                for row in range(image.shape[0]):
                    line.append(image[row, column, 0].numpy())
            writer_image.writerow(line)
    
        
    with open('forward_pass_test/test_labels.csv', 'w') as label_file:
        writer_label = csv.writer(label_file)
        for i, batch in enumerate(tqdm.tqdm(test_eval)):
            label = batch['label']
            label = tf.one_hot(label, no_classes)
            line = []
            for row in range(label.shape[0]):
                line.append(label[row].numpy())
            writer_label.writerow(line)