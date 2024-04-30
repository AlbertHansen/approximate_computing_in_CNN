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
labels = []
for number_of_classes in list(range(5,101,5)):
    labels.append(f'{number_of_classes} classes')

file_paths = []
for number_of_classes in list(range(5,101,5)):
    file_paths.append(f'{number_of_classes}_classes.csv')

data = []
for file_path in file_paths:
    data.append(pd.read_csv(file_path))

if 'SURPRESS_FIGURES' in os.environ:
    print(data[0].head())

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

colors = dc(len(labels))

#%% Plot

# Create a new figure
gc = gridspec.GridSpec(1, 3)
fig = plt.figure(layout='constrained', figsize=(17.01*cm, 8*cm))

# Create layout
axs = []
axs.append(fig.add_subplot(gc[0, 0:2]))
axs.append(fig.add_subplot(gc[0, 2]))

# Plot 'accuracy' and 'val_accuracy' for each DataFrame
for df, label, color in zip(data, labels, colors): 
    axs[0].plot(df['accuracy'], linestyle='-', color=color)
    axs[0].plot(df['val_accuracy'], linestyle=':', color=color)
    axs[0].set_ylim(0, 0.7)
    axs[0].set_xlim(0, 250)
    axs[0].grid(True)
    axs[0].set_xlabel('Epochs \([\cdot]\)')
    axs[0].set_ylabel('Accuracy \([\cdot]\)')
    
# Create legend for the linestyles
style_legend = [mlines.Line2D([], [], color='black', linestyle='-', label='\\texttt{accuracy}'),
                mlines.Line2D([], [], color='black', linestyle=':', label='\\texttt{val_accuracy}')]
axs[0].legend(handles=style_legend, loc='upper left')

# Create a legend for the colors
color_legend = [mlines.Line2D([], [], color=c, label=l) for c, l in zip(colors, labels)]
axs[1].legend(handles=color_legend, loc='upper left', ncol=2)
axs[1].axis('off')

# Save and show
plt.show()
plt.savefig('classes_accuracy.pdf', bbox_inches='tight')
# plt.close()

# Mean time spent on each epoch
average_times = []
for df in data:
    # Calculate the mean time for each DataFrame
    mean_time = df.loc[:, 'time'].mean()

    # Store the mean time in the dictionary 
    average_times.append(mean_time)

for label, time in zip(labels, average_times):
    print(f'{label}: {time:.2f} s')