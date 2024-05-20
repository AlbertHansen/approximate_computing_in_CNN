#%% Dependencies
import tensorflow as tf
import tqdm as tqdm
from NoisyLayers import * 
from test_custom_layers import * 



#%% Main
def main() -> None:
    # test_noisy_layers()
    test_layer = NoisyDense(10, activation='relu')
    inputs = tf.random.normal((1, 20))
    outputs = test_layer(inputs)


if __name__ == "__main__":
    main()
# %%
