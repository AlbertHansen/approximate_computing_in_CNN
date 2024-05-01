#%% Dependencies
import pandas as pd
import numpy as np

#%% Functions
def subtract_csv(file1, file2):
    df1 = pd.read_csv(file1, header=None)
    #print(df1.head())
    df2 = pd.read_csv(file2, header=None)
    #print(df2.head())
    df_diff = df1 - df2
    #print(df_diff.head())

    return df_diff

#%% Layer 0
paths = []
for i in range(5):
    paths.append(f'iteration_{i}/layer_0/weights.csv')

df = pd.DataFrame()
for i in range(1, 5):
    df_diff = subtract_csv(paths[i], paths[i-1])
    flat_array = df_diff.to_numpy().flatten()
    flat_array = pd.DataFrame(flat_array)
    #print(flat_array)
    df[f'diff_{i}_minus_{i-1}'] = df_diff.values.flatten()

print(df.head())

#%% Layer 2
paths = []
for i in range(5):
    paths.append(f'iteration_{i}/layer_2/weights.csv')

for i in range(1, 5):
    df_diff = subtract_csv(paths[i], paths[i-1])
    print(max(df_diff))
    #df_diff.to_csv(f'runs/weight_increments_test/weight_increment_{i}.csv', index=False)
#%% Layer 4
paths = []
for i in range(5):
    paths.append(f'iteration_{i}/layer_4/weights.csv')

for i in range(1, 5):
    df_diff = subtract_csv(paths[i], paths[i-1])
    print(max(df_diff))
    #df_diff.to_csv(f'runs/weight_increments_test/weight_increment_{i}.csv', index=False)
#%% Layer 6
paths = []
for i in range(5):
    paths.append(f'iteration_{i}/layer_6/weights.csv')

for i in range(1, 5):
    df_diff = subtract_csv(paths[i], paths[i-1])
    print(max(df_diff))
    #df_diff.to_csv(f'runs/weight_increments_test/weight_increment_{i}.csv', index=False)
#%% Layer 7
paths = []
for i in range(5):
    paths.append(f'iteration_{i}/layer_7/weights.csv')

for i in range(1, 5):
    df_diff = subtract_csv(paths[i], paths[i-1])
    print(max(df_diff))
    #df_diff.to_csv(f'runs/weight_increments_test/weight_increment_{i}.csv', index=False)