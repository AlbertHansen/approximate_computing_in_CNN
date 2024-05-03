import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV files
actual = np.loadtxt("Actual.csv", delimiter=",")
expected = np.loadtxt("Expected.csv", delimiter=",")

# Calculate the difference
diff = expected - actual

# Define bin edges for histogram
bin_edges = np.linspace(-9.5, 9.5, 40)

# Create the histogram
plt.figure(figsize=(6, 3))
plt.hist(diff, bins=bin_edges, density=True)
plt.xlabel("Error [.]")
plt.ylabel("Probability [.]")
plt.grid(True)
plt.xlim(-2, 2)
plt.ylim(0, 1)

# Save the plot as PDF
plt.savefig("histogram.pdf", dpi=150)
plt.show()
