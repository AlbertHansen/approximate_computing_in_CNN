import numpy as np
import test
import sys 

print(sys.path)


i = 0
inputs = np.random.rand(5, 5, 5, 5)
kernel = np.random.rand(3, 3, 3, 3)
output = test.worker(i, inputs, kernel)

print(output)