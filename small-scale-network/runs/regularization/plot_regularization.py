#%% Dependencies
import os
import importlib.util
import matplotlib
import warnings
import colorsys

if 'SUPPRESS_FIGURES' in os.environ:
    matplotlib.use('Agg')
    warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Import distinguishable colours
from distinguishable_colours import distinguishable_colors as dc

#%% Load data
lambda_values = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5]

l1_paths = []
l2_paths = []
l1l2_paths = []

labels = []
file_paths_l1 = []
file_paths_l2 = []
file_paths_l1l2 = []

for lambda_value in lambda_values:
    labels.append(f'lambda = {lambda_value}')
    file_paths_l1.append(f'l1/lambda_{lambda_value}.csv')
    file_paths_l2.append(f'l2/lambda_{lambda_value}.csv')
    file_paths_l1l2.append(f'l1l2/lambda_{lambda_value}.csv')

data_l1 = []
for file_path in file_paths_l1:
    data_l1.append(pd.read_csv(file_path))
data_l2 = []
for file_path in file_paths_l2:
    data_l2.append(pd.read_csv(file_path))
data_l1l2 = []
for file_path in file_paths_l1l2:
    data_l1l2.append(pd.read_csv(file_path))

if 'SURPRESS_FIGURES' not in os.environ:
    print(data_l1[0].head())

#%% Plot settings
SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

plt.rc('font',      size        = SMALL_SIZE)   # controls default text sizes
plt.rc('axes',      titlesize   = BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes',      labelsize   = MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick',     labelsize   = SMALL_SIZE)   # fontsize of the tick labels
plt.rc('ytick',     labelsize   = SMALL_SIZE)   # fontsize of the tick labels
plt.rc('legend',    fontsize    = SMALL_SIZE)   # legend fontsize
plt.rc('figure',    titlesize   = BIGGER_SIZE)  # fontsize of the figure title
# plt.rc('text',      usetex      = True)         # use latex for interpreter
plt.rc('font',      family      = 'Computer Modern Serif')      # use serif font (to look like latex)
plt.rc('font',      weight      = 'heavy')     # controls font weight

cm = 1/2.54                                     # centimeters in inches

colors = dc(len(labels))

#%%%%%%%%%%%%%%%%%%%%% Plot L1 Regularization %%%%%%%%%%%%%%%%%%%%%

# Create a new figure
gc = gridspec.GridSpec(1, 5)
fig = plt.figure(layout='constrained', figsize=(17.01*cm, 8*cm))

# Create layout
axs = []
axs.append(fig.add_subplot(gc[0, 0:4]))
axs.append(fig.add_subplot(gc[0, 4]))

# Plot 'accuracy' and 'val_accuracy' for each DataFrame
for df, label, color in zip(data_l1, labels, colors): 
    axs[0].plot(df['accuracy'], linestyle='-', color=color)
    axs[0].plot(df['val_accuracy'], linestyle=':', color=color)
    axs[0].set_ylim(0, 1)
    axs[0].set_xlim(0, 250)
    axs[0].grid(True)
    axs[0].set_xlabel('Epochs [.]')
    axs[0].set_ylabel('Accuracy [.]')
    
# Create legend for the linestyles
style_legend = [mlines.Line2D([], [], color='black', linestyle='-', label='accuracy'),
                mlines.Line2D([], [], color='black', linestyle=':', label='val_accuracy')]
axs[0].legend(handles=style_legend, loc='upper left')

# Create a legend for the colors
color_legend = [mlines.Line2D([], [], color=c, label=l) for c, l in zip(colors, labels)]
axs[1].legend(handles=color_legend, loc='upper left', ncol=2)
axs[1].axis('off')

# Save and show
plt.savefig('classes_accuracy.pdf', bbox_inches='tight')
plt.show()
plt.close()

#%% Calculate the average time spent on each epoch
# Mean time spent on each epoch
average_times = []
for df in data_l1:
    # Calculate the mean time for each DataFrame
    mean_time = df.loc[:, 'time'].mean()

    # Store the mean time in the dictionary 
    average_times.append(mean_time)

for label, time in zip(labels, average_times):
    print(f'{label}: {time:.2f} s')

#%%%%%%%%%%%%%%%%%%%%% Plot L2 Regularization %%%%%%%%%%%%%%%%%%%%%

# Create a new figure
gc = gridspec.GridSpec(1, 5)
fig = plt.figure(layout='constrained', figsize=(17.01*cm, 8*cm))

# Create layout
axs = []
axs.append(fig.add_subplot(gc[0, 0:4]))
axs.append(fig.add_subplot(gc[0, 4]))

# Plot 'accuracy' and 'val_accuracy' for each DataFrame
for df, label, color in zip(data_l2, labels, colors): 
    axs[0].plot(df['accuracy'], linestyle='-', color=color)
    axs[0].plot(df['val_accuracy'], linestyle=':', color=color)
    axs[0].set_ylim(0, 1)
    axs[0].set_xlim(0, 250)
    axs[0].grid(True)
    axs[0].set_xlabel('Epochs [.]')
    axs[0].set_ylabel('Accuracy [.]')
    
# Create legend for the linestyles
style_legend = [mlines.Line2D([], [], color='black', linestyle='-', label='accuracy'),
                mlines.Line2D([], [], color='black', linestyle=':', label='val_accuracy')]
axs[0].legend(handles=style_legend, loc='upper left')

# Create a legend for the colors
color_legend = [mlines.Line2D([], [], color=c, label=l) for c, l in zip(colors, labels)]
axs[1].legend(handles=color_legend, loc='upper left', ncol=2)
axs[1].axis('off')

# Save and show
plt.savefig('classes_accuracy.pdf', bbox_inches='tight')
plt.show()
plt.close()

#%% Calculate the average time spent on each epoch
# Mean time spent on each epoch
average_times = []
for df in data_l2:
    # Calculate the mean time for each DataFrame
    mean_time = df.loc[:, 'time'].mean()

    # Store the mean time in the dictionary 
    average_times.append(mean_time)

for label, time in zip(labels, average_times):
    print(f'{label}: {time:.2f} s')

#%%%%%%%%%%%%%%%%%%%%% Plot L1L2 Regularization %%%%%%%%%%%%%%%%%%%%%

# Create a new figure
gc = gridspec.GridSpec(1, 5)
fig = plt.figure(layout='constrained', figsize=(17.01*cm, 8*cm))

# Create layout
axs = []
axs.append(fig.add_subplot(gc[0, 0:4]))
axs.append(fig.add_subplot(gc[0, 4]))

# Plot 'accuracy' and 'val_accuracy' for each DataFrame
for df, label, color in zip(data_l1l2, labels, colors): 
    axs[0].plot(df['accuracy'], linestyle='-', color=color)
    axs[0].plot(df['val_accuracy'], linestyle=':', color=color)
    axs[0].set_ylim(0, 1)
    axs[0].set_xlim(0, 250)
    axs[0].grid(True)
    axs[0].set_xlabel('Epochs [.]')
    axs[0].set_ylabel('Accuracy [.]')
    
# Create legend for the linestyles
style_legend = [mlines.Line2D([], [], color='black', linestyle='-', label='accuracy'),
                mlines.Line2D([], [], color='black', linestyle=':', label='val_accuracy')]
axs[0].legend(handles=style_legend, loc='upper left')

# Create a legend for the colors
color_legend = [mlines.Line2D([], [], color=c, label=l) for c, l in zip(colors, labels)]
axs[1].legend(handles=color_legend, loc='upper left', ncol=2)
axs[1].axis('off')

# Save and show
plt.savefig('classes_accuracy.pdf', bbox_inches='tight')
plt.show()
plt.close()

#%% Calculate the average time spent on each epoch
# Mean time spent on each epoch
average_times = []
for df in data_l1l2:
    # Calculate the mean time for each DataFrame
    mean_time = df.loc[:, 'time'].mean()

    # Store the mean time in the dictionary 
    average_times.append(mean_time)

for label, time in zip(labels, average_times):
    print(f'{label}: {time:.2f} s')
# %%
