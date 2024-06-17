#%%
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Read the CSV file
# Read the CSV file
data = pd.read_csv("Error_mul8s_1KV9.csv", header=None)

# Dictionary to store PMFs for each row (each b value)
pmf_dict = {}

    # Iterate over each row to calculate and store the PMF
for index, row in data.iterrows():
    error_distributions = row.values
        
        # Calculate unique values and frequencies
    unique_values, value_counts = np.unique(error_distributions, return_counts=True)
    freq = value_counts / len(error_distributions)
        
        # Combine unique values and frequencies into a list of tuples
    pmf = dict(zip(unique_values, freq))  # Store as dictionary for easier manipulation
    pmf_dict[index] = pmf
    

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
        
        plt.plot(x_fit, y_fit, label=f'Fitted Gaussian', lw=2)
    
    plt.xlabel('Error sum')
    
    plt.ylabel('Probability')
    plt.legend()
    plt.title('Convolved PMF and Fitted Distributions')
    plt.grid()
    plt.savefig('1KV9_gaussian_fit.pdf', bbox_inches='tight')
    plt.show()

# Example: Convolving PMFs for a list of b indexes
# Example: Convolving PMFs for a list of b indexes
b_indexes = pd.read_csv('weights.csv', header=None, nrows=1).to_numpy(dtype=np.float32)
#b_indexes = list(b_indexes)
b_indexes = b_indexes* (2 ** 7)
b_indexes = b_indexes + 128
b_indexes = b_indexes.astype(np.int16)
b_indexes = b_indexes.flatten().tolist()
#del b_indexes
#b_indexes = [147,155,110,125]
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
        print(f"{name} distribution parameters: mu = {mu}, sigma² = {sigma}")

# Plot the convolved PMF and fitted distributions
plot_fitted_distributions(convolved_pmf, fit_results)

# %%
