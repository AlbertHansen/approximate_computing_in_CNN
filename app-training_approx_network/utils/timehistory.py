import tensorflow as tf
import time

#%% Keeping time
import tensorflow as tf
import time

class TimeHistory(tf.keras.callbacks.Callback):
    """
    Callback to record the time taken for each epoch during training.

    Attributes:
        times (list): List to store the time taken for each epoch.

    Methods:
        on_train_begin(logs={}): Called at the beginning of training.
        on_epoch_begin(batch, logs={}): Called at the beginning of each epoch.
        on_epoch_end(batch, logs={}): Called at the end of each epoch.
    """

    def on_train_begin(self, logs={}):
        """
        Called at the beginning of training.
        Initializes the times list.
        """
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        """
        Called at the beginning of each epoch.
        Records the start time of the epoch.
        """
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        """
        Called at the end of each epoch.
        Calculates the time taken for the epoch and appends it to the times list.
        """
        self.times.append(time.time() - self.epoch_time_start)