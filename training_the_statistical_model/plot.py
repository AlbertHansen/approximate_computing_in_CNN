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
# from distinguishable_colours import distinguishable_colors as dc

#%% Load data
labels = [
    'mul8s_1KV9'
]

file_paths_stat = [
    'mul8s_1kv9_stats_and_approx.csv'
]


file_paths_approx = [
    '1KV9_weights/mul8s_1KV9_ref_accuracy.csv'
]

data_stat = []
for file_path in file_paths_stat:
    df = pd.read_csv(file_path, header=None)
    df.columns = ['epoch', 'accuracy', 'accuracy_val']
    data_stat.append(df)

data_approx = []
for file_path in file_paths_approx:
    df = pd.read_csv(file_path)
    # df.columns = ['epoch', 'accuracy', 'accuracy_val', 'time']
    data_approx.append(df)

if 'SURPRESS_FIGURES' not in os.environ:
    print(data_approx[0].head())

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
# plt.rc('font',      family      = 'Computer Modern Serif')      # use serif font (to look like latex)
plt.rc('font',      weight      = 'heavy')     # controls font weight

cm = 1/2.54                                     # centimeters in inches

# colors = dc(len(labels))
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
colors = colors[:len(labels)]

#%%%%%%%%%%%%%%%%%%%%% Plot 15 Approximate Epochs %%%%%%%%%%%%%%%%%%%%%
# Create a new figure
# gc = gridspec.GridSpec(1, 5)
fig, ax = plt.subplots(figsize=(17.01*cm, 6*cm))

# Create layout
#axs = []
#axs.append(fig.add_subplot(gc[0, 0:4]))
#axs.append(fig.add_subplot(gc[0, 4]))

# Plot 'accuracy' and 'val_accuracy' for each DataFrame
epochs_stat = data_stat[0]['epoch']
epochs_approx = data_approx[0]['epoch'] + 1
for df_stat, df_approx, label, color in zip(data_stat, data_approx, labels, colors): 
    # ax.plot(list(epochs_approx), list(df_approx['accuracy']), label=f'{label}-approx: accuracy', linestyle='-', color=color)
    ax.plot(list(epochs_approx), list(df_approx['accuracy_val']), label=f'{label}-approx: accuracy_val', linestyle=':', color=color)
    # ax.scatter(epochs_stat, df_stat['accuracy'], label=f'{label}-stat: accuracy', color=color, marker="1", s=150)
    ax.scatter(epochs_stat, df_stat['accuracy_val'], label=f'{label}-stat: accuracy_val', color=color, marker="2", s=150)

    #axs[0].set_ylim(0, 0.75)
ax.set_xlim(44, 86)
ax.grid(True)
ax.set_xlabel('Epochs [.]')
ax.set_ylabel('Accuracy [.]')
    
ax.legend(bbox_to_anchor=(1.4, 1.1), loc='upper center', ncol=1)


# Save and show
# plt.savefig('classes_accuracy.pdf', bbox_inches='tight')
#plt.savefig('../../../98-diagrams/04-training_approx_network/small_network/adamax_and_sgd_does_it_train.pdf', bbox_inches='tight')
plt.show()
plt.close()
# %%
