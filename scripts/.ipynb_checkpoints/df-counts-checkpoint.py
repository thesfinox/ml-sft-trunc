import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

# set common options
sns.set()
os.makedirs('./img', exist_ok=True)

# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', type=str, help='input dataset (CSV)')
parser.add_argument('-o', '--output', type=str, help='basename of the output plots')
parser.add_argument('-r', '--rows', type=int, default=5, help='number of rows in the plot')
parser.add_argument('-c', '--columns', type=int, default=5, help='number of columns in the plot')

args = parser.parse_args()

# import the dataset
df = pd.read_csv(args.input)

# plot all features
nrows, ncols = (args.rows, args.columns)
_, ax = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))

df = df.drop(columns=['solutions', 'init'])
cols = np.array(df.columns).reshape(nrows, ncols)

# define the bins by order of magnitude (oom)
oom_bins = [-np.inf, -1e6, -1e3, -1e1,
            -1, 1,
            1e1, 1e3, 1e6, np.inf
           ]
oom_bins_labels = ['$(-\infty, -10^6]$',
                   '$(-10^6, -10^3]$',
                   '$(-10^3, -10]$',
                   '$(-10, -1]$',
                   '$(-1, 1]$',
                   '$(1, 10]$',
                   '$(10, 10^3]$',
                   '$(10^3, 10^6]$',
                   '$(10^6, +\infty]$'
                  ]

# cut the data into the bins
for i in range(nrows):
    for j in range(ncols):
        # cut in defined places and return the counts
        cuts = pd.cut(df[cols[i,j]],
                      bins=oom_bins,
                      labels=range(len(oom_bins)-1)
                     ).values
        bins, counts = np.unique(cuts, return_counts=True)
        
        # insert counts in the list
        counts_plot = np.zeros(len(oom_bins) - 1)
        for n in range(len(bins)):
            counts_plot[bins[n]] = counts[n]
            
        # plot the result
        sns.barplot(x=np.arange(len(oom_bins)-1),
                    y=counts_plot,
                    color=sns.color_palette('muted', 1)[0],
                    alpha=0.5,
                    ax=ax[i,j]
                   )
        ax[i,j].set(ylabel='count',
                    xlabel=cols[i,j]
                   )
        ax[i,j].set_xticklabels(oom_bins_labels,
                                rotation=90,
                                va='top'
                               )

plt.tight_layout()
plt.savefig('./img/{}.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}.png'.format(args.output), dpi=150, format='png')