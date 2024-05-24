#%%
import numpy as np
import pandas as pd
import scipy.stats as sp
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
#%%


# Reading the CSV files
epoch_acc_pred = pd.read_csv('./accurate_predictions/1KV9_weights/save_45.csv.csv', header=None)
epoch_det_pred = pd.read_csv('./approximate_predictions/1KV9_weights/save_45.csv', header=None)

# Iterating over rows of epoch_det_pred
for image_index, image_det_pred in epoch_det_pred.iterrows():
    # Accessing image index and row data
    print('hej')
    image_acc_pred = epoch_acc_pred.iloc[:, image_index].values
    image_pro_pred = pd.read_csv(f'./statistical_predictions/1KV9_weights/save_45/image_{image_index}.csv', header=None)
    
    # Calculating mean of each column
    column_means = image_pro_pred.mean()

    # Your operations using image_acc_pred and image_pro_pred go here

'''
accurate_predictions_45_epochs = pd.read_csv('./accurate_predictions/1KV9_weights/save_65.csv.csv', header=None)
deterministic_predictions_45_epochs = pd.read_csv('./approximate_predictions/1KV9_weights/save_65.csv', header=None)
image_0_probabilistic_predictions = pd.read_csv('./statistical_predictions/1KV9_weights/save_65/image_0.csv', header=None)

accurate_predictions_45_epochs_image_0 = accurate_predictions_45_epochs.iloc[:,0].values
deterministic_predictions_45_epochs_image_0 = deterministic_predictions_45_epochs.iloc[0,:].values

deterministic_error_vector_image_0 = deterministic_predictions_45_epochs_image_0 - accurate_predictions_45_epochs_image_0
probabilistic_error_matrix_image_0 = image_0_probabilistic_predictions - accurate_predictions_45_epochs_image_0[:, np.newaxis]

error_of_first_entry_deterministic = deterministic_error_vector_image_0[7]
error_of_first_entry_probabilistic = probabilistic_error_matrix_image_0.iloc[7]

ecdf = ECDF(error_of_first_entry_probabilistic)
plt.step(ecdf.x, ecdf.y, where='post', label='EDF of Probabilistic Errors')
plt.hist(error_of_first_entry_probabilistic, bins=50, density=True, alpha=0.5, label='Deterministic Errors', color='red')
plt.axvline(error_of_first_entry_deterministic, color='r', linestyle='dashed', linewidth=2, label='Deterministic Error Magnitude')
plt.xlabel('Error')
plt.ylabel('Empirical CDF / Frequency')
plt.legend()
plt.title('Empirical Distribution Function of Error Vectors')
plt.show()'''
# %%

