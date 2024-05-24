#%%
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Read the CSV file
filename = 'Error_mul8s_1KV9.csv'  # Replace with your CSV file name
data = pd.read_csv(filename, header=None)

# Dictionary to store PMFs for each row (each b value)
pmf_dict = {}

# Iterate over each row to calculate and store the PMF
for index, row in data.iterrows():
    error_distributions = row.values
    
    # Calculate unique values and frequencies
   # Initialize the dictionary to hold lists of indices for each unique value
    # Calculate unique values and frequencies
        # Initialize the dictionary to hold lists of indices for each unique value
    index_dict = {}

    input_pmf = {74: 0.0078125, 80: 0.00390625, 82: 0.01171875, 76: 0.0078125, 64: 0.00390625, 54: 0.015625, 45: 0.01171875, 40: 0.0078125, 37: 0.00390625, 36: 0.0234375, 26: 0.04296875, 27: 0.02734375, 28: 0.02734375, 83: 0.00390625, 71: 0.015625, 63: 0.0078125, 51: 0.01953125, 44: 0.01171875, 43: 0.01953125, 41: 0.00390625, 38: 0.00390625, 35: 0.00390625, 34: 0.015625, 25: 0.0390625, 66: 0.015625, 61: 0.0078125, 47: 0.015625, 46: 0.0234375, 39: 0.0234375, 23: 0.0234375, 18: 0.00390625, 52: 0.01171875, 50: 0.01171875, 65: 0.0078125, 79: 0.01171875, 68: 0.015625, 49: 0.015625, 48: 0.01953125, 22: 0.01171875, 53: 0.00390625, 69: 0.0078125, 62: 0.00390625, 55: 0.0078125, 75: 0.00390625, 19: 0.01171875, 30: 0.015625, 32: 0.0078125, 24: 0.03515625, 21: 0.01171875, 31: 0.0078125, 14: 0.00390625, 20: 0.00390625, 29: 0.01171875, 99: 0.015625, 97: 0.0078125, 87: 0.0078125, 104: 0.0078125, 91: 0.0078125, 102: 0.0078125, 67: 0.0078125, 78: 0.01171875, 33: 0.00390625, 59: 0.0078125, 111: 0.00390625, 106: 0.015625, 88: 0.0078125, 58: 0.00390625, 72: 0.01171875, 77: 0.00390625, 73: 0.01171875, 70: 0.0078125, 98: 0.0078125, 81: 0.015625, 101: 0.00390625, 92: 0.00390625, 90: 0.00390625, 94: 0.00390625, 115: 0.0078125, 117: 0.00390625, 113: 0.00390625, 116: 0.015625, 108: 0.01953125, 112: 0.0078125, 118: 0.01171875, 105: 0.0078125, 121: 0.00390625, 120: 0.00390625, 109: 0.00390625, 107: 0.00390625, 110: 0.00390625, 119: 0.0078125, 103: 0.00390625, 96: 0.00390625}
    #input_pmf = {74: 0.25, 82: 0.25, 76: 0.25, 80: 0.25}
# Populate the dictionary with indices
    for idx, value in enumerate(error_distributions):
        if value not in index_dict:
            if idx in input_pmf.keys():
                index_dict[value] = input_pmf[idx]
            else:
                index_dict[value] = 0
        if idx in input_pmf.keys():
            index_dict[value] = index_dict[value] + input_pmf[idx]

        # Combine unique values and frequencies into a list of tuples
        
    pmf_dict[index] = index_dict
    

# Function to convolve PMFs for a list of b-indexes
def convolve_pmfs(b_indexes):
    # Initialize the combined PMF with the PMFs of the first two b-indexes
    combined_pmf = pmf_dict[b_indexes[0]].copy()
    
    # Convolve the PMFs for the remaining b-indexes
    for b_index in b_indexes[1:]:
        current_pmf = pmf_dict[b_index]
        
        # Convolve the current PMF with the combined PMF
        temp_combined_pmf = {}
        for error_sum, prob_sum in combined_pmf.items():
            for error, prob in current_pmf.items():
                new_error_sum = error_sum + error
                new_prob_sum = prob_sum * prob
                
                if new_error_sum in temp_combined_pmf:
                    temp_combined_pmf[new_error_sum] += new_prob_sum
                else:
                    temp_combined_pmf[new_error_sum] = new_prob_sum
        
        # Update the combined PMF with the convolved PMF
        combined_pmf = temp_combined_pmf
    
    return combined_pmf

# Function to plot the convolved PMF and fitted distributions
def plot_fitted_distributions(convolved_pmf, fit_results):
    # Convert the convolved PMF to x and y values
    x_pmf = np.array(list(convolved_pmf.keys()))
    y_pmf = np.array(list(convolved_pmf.values()))

    # Plot the convolved PMF
    plt.figure(figsize=(10, 6))
    plt.bar(x_pmf, y_pmf, width=0.5, alpha=0.6, label='Convolved PMF', color='gray')

    # Plot each fitted distribution
    x_fit = np.linspace(min(x_pmf), max(x_pmf), 1000)
    for name, params in fit_results.items():
        if name == 'norm':
            y_fit = stats.norm.pdf(x_fit, *params)
        
        plt.plot(x_fit, y_fit, label=f'Fitted {name}', lw=2)
    
    plt.xlabel('Error sum')
    plt.ylabel('Probability')
    plt.legend()
    plt.title('Convolved PMF and Fitted Distributions')
    # plt.savefig('1KV9.pdf', bbox_inches='tight')
    plt.show()

# Example: Convolving PMFs for a list of b indexes
# Example: Convolving PMFs for a list of b indexes
b_indexes = pd.read_csv('weights.csv', header=None, nrows=1).to_numpy(dtype=np.float32)
#b_indexes = list(b_indexes)
b_indexes = b_indexes* (2 ** 6)
b_indexes = b_indexes + 128
b_indexes = b_indexes.astype(np.int16)
b_indexes = b_indexes.flatten().tolist()
print(b_indexes)
convolved_pmf = convolve_pmfs(b_indexes)

# Convert the convolved PMF to a sample dataset
samples = []
for error_sum, prob_sum in convolved_pmf.items():
    num_samples = int(prob_sum * 1000000)  # Scale to get a sizable sample
    samples.extend([error_sum] * num_samples)
samples = np.array(samples)

# Fit different parametric distributions
distributions = {
    'norm': stats.norm,
}

# Dictionary to store the results of the fits
fit_results = {}

# Fit each distribution to the sample data
for name, distribution in distributions.items():
    params = distribution.fit(samples)
    fit_results[name] = params

# Print the parameters for each fitted distribution
for name, params in fit_results.items():
    if name == 'norm':
        mu, sigma = params
        print(f"{name} distribution parameters: mu = {mu}, sigma² = {sigma**2}")

# Plot the convolved PMF and fitted distributions
plot_fitted_distributions(convolved_pmf, fit_results)

# %%
