#%% Dependencies
import os
import matplotlib
import warnings

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
file_paths_with_bias = []
file_paths_without_bias = []
for number_of_classes in list(range(5,101,5)):
    file_paths_with_bias.append(f'../../96-data/01-small_network/classes_with_bias/{number_of_classes}_classes.csv')
    file_paths_without_bias.append(f'../../96-data/01-small_network/classes_without_bias/{number_of_classes}_classes.csv')

data_with_bias = []
data_without_bias = []
for file_path in file_paths_with_bias:
    data_with_bias.append(pd.read_csv(file_path))
for file_path in file_paths_without_bias:
    data_without_bias.append(pd.read_csv(file_path))

if 'SURPRESS_FIGURES' not in os.environ:
    print(data_with_bias[0].head())

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
plt.rc('text',      usetex      = True)         # use latex for interpreter
plt.rc('font',      family      = 'Computer Modern Serif')      # use serif font (to look like latex)
plt.rc('font',      weight      = 'heavy')     # controls font weight

cm = 1/2.54                                     # centimeters in inches



#%% Plot
labels = [
    '5 Classes with Bias', '5 Classes without Bias',
    '10 Classes with Bias', '10 Classes without Bias',
    '15 Classes with Bias', '15 Classes without Bias',
    '20 Classes with Bias', '20 Classes without Bias',
    '25 Classes with Bias', '25 Classes without Bias'
]
titles = [
    '5 Classes',
    '10 Classes',
    '15 Classes',
    '20 Classes',
    '25 Classes'
]

colors = dc(10)

# Create a new figure
gc = gridspec.GridSpec(1, 6)
fig = plt.figure(layout='constrained', figsize=(17.01*cm, 8*cm))

# Create layout
axs = []
for i in range(6):
    axs.append(fig.add_subplot(gc[0, i]))

for i in range(5):
    # Plot 'acc' and 'val_acc' for each DataFrame
    axs[i].plot(data_with_bias[i]['accuracy'], linestyle='-', color=colors[2*i])
    axs[i].plot(data_with_bias[i]['val_accuracy'], linestyle=':', color=colors[2*i])
    axs[i].plot(data_without_bias[i]['accuracy'], linestyle='-', color=colors[2*i+1])
    axs[i].plot(data_without_bias[i]['val_accuracy'], linestyle=':', color=colors[2*i+1])
    axs[i].set_ylim(0, 1)
    axs[i].set_xlim(0, 250)
    axs[i].grid(True)
    axs[i].set_xlabel('Epochs [.]')
    # axs[i].set_title(titles[i])
    if i == 0:
        axs[i].set_ylabel('Loss [.]')
    else: 
        axs[i].set_yticklabels([])


# Create legend for the linestyles
style_legend = [mlines.Line2D([], [], color='black', linestyle='-', label='accuracy'),
                mlines.Line2D([], [], color='black', linestyle=':', label='val_accuracy')]

# Create a legend for the colors
color_legend = [mlines.Line2D([], [], color=c, label=l) for c, l in zip(colors, labels)]

# Add color_legend to the last subplot
axs[-1].legend(handles=color_legend, loc='upper left', ncol=1)
axs[-1].axis('off')

# Create a new axes for the style_legend
legend_fig = plt.figure(figsize=(3, 2))
style_ax = legend_fig.add_subplot(111)
style_ax.legend(handles=style_legend, loc='upper left', ncol=1)
style_ax.axis('off')

#%%
plt.close()

#%% Calculate the average time spent on each epoch
# Mean time spent on each epoch
average_times = []
for df in data:
    # Calculate the mean time for each DataFrame
    mean_time = df.loc[:, 'time'].mean()

    # Store the mean time in the dictionary 
    average_times.append(mean_time)

for label, time in zip(labels, average_times):
    print(f'{label}: {time:.2f} s')
# %%
