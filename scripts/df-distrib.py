import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

def discrete_bins(series: pd.Series,
                  n_bins: int,
                  ax: plt.axes
                 ) -> plt.axes:
    '''
    Digitise the series into a given number of discrete bins.
    
    Required arguments:
        series: the Pandas series to manipulate,
        n_bins: the number of bins,
        ax:     the subplot axis.
        
    Returns:
        the subplot axis
    '''
    # check common options
    sns.set()
    
    # first check the number of unique values
    n_unique = pd.unique(series).shape[0]
    if n_bins > n_unique:
        n_bins = n_unique
        
    # compute the discretisation
    cuts, bins = pd.cut(series,
                        right=False,
                        bins=n_bins,
                        labels=range(n_bins),
                        retbins=True,
                        precision=1
                       )
    
    # plot the discretization
    sns.distplot(cuts,
                 bins=range(n_bins + 1),
                 kde=False,
                 ax=ax
                )
    ax.set(ylabel='count',
           xlabel=series.name,
           xticks=range(n_bins + 1),
           yscale='log'
          )
    ax.set_xticklabels(np.round(bins, 1), rotation=90, ha='right')
    
    return ax

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

cols = np.array(df.columns).reshape(nrows, ncols)

for i in range(nrows):
    for j in range(ncols):
        discrete_bins(df[cols[i,j]], n_bins=10, ax=ax[i,j])
        
plt.tight_layout()
plt.savefig('./img/{}.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}.png'.format(args.output), dpi=150, format='png')