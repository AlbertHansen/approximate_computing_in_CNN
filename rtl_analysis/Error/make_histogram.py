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

colors = dc(1)

# Load data from CSV files
actual = np.loadtxt("./Error/Error_files/Actual.csv", delimiter=",")
expected = np.loadtxt("./Error/Error_files/Expected.csv", delimiter=",")

# Calculate the difference
diff = expected - actual

# Count for bins
unique_values, value_counts = np.unique(diff, return_counts=True)
freq = value_counts / len(diff)


# Create the histogram
fig = plt.figure(layout='constrained', figsize=(17.01*cm, 8*cm))
plt.bar(unique_values,freq)
plt.xlabel("Error [.]")
plt.ylabel("Probability [.]")
plt.grid(True)
# plt.xlim(-2, 2)
plt.ylim(0, 1)

# Save the plot as PDF
plt.savefig("./figures/histogram.pdf", dpi=150)
plt.show()
