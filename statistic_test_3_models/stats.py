import numpy as np
from energy import mv_two_sample

# Example data creation (replace with your actual data)
np.random.seed(0)
deterministic_vectors = np.random.randn(800, 10)
probabilistic_samples = np.random.randn(1000, 10)

# Perform the multivariate two-sample test
stat, p_value = mv_two_sample(deterministic_vectors, probabilistic_samples)

print(f"Test Statistic: {stat}")
print(f"P-Value: {p_value}")