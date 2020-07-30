import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
import time
import argparse

def svd(df: pd.DataFrame) -> np.ndarray:
    '''
    Compute the retained variance of the singular values of a matrix given as DataFrame in Pandas.
    
    Required arguments:
        df: the dataframe.
        
    Returns:
        the singular values.
    '''
    
    _, S, _ = np.linalg.svd(df.values)
    var     = np.square(S) / np.sum(np.square(S))
    
    return var

# set common options
sns.set()
os.makedirs('./img', exist_ok=True)
os.makedirs('./data', exist_ok=True)

# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-r', '--re', type=str, help='input dataset (CSV) with real parts')
parser.add_argument('-i', '--im', type=str, help='input dataset (CSV) with imaginary parts')
parser.add_argument('-f', '--full', type=str, help='full input dataset (CSV)')
parser.add_argument('-o', '--output', type=str, help='basename of the output plots')

args = parser.parse_args()

# import the datasets
df_full = pd.read_csv(args.full)
df_re   = pd.read_csv(args.re)
df_im   = pd.read_csv(args.im)

# compute the SVD
t = time.time()
var_full, var_re, var_im = joblib.Parallel(n_jobs=-1)\
                          (joblib.delayed(svd)(m) for m in [df_full, df_re, df_im])
t = time.time() - t
print('SVD computed in {:.3f} seconds'.format(t))

# create a dataframe
df_tot = pd.DataFrame(var_full, columns=['full'])
df_sep = pd.DataFrame(zip(var_re, var_im), columns=['re', 'im'])
df_tot.to_csv('./data/{}_tot.csv'.format(args.output), index=False)
df_sep.to_csv('./data/{}_sep.csv'.format(args.output), index=False)

# plot the retained variance
_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.scatterplot(data=df_tot,
                ax=ax
               )
ax.set(title='',
       xlabel='principal components',
       ylabel='retained variance',
       yscale='log'
      )
ax.legend().remove()
        
plt.tight_layout()
plt.savefig('./img/{}_tot.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}_tot.png'.format(args.output), dpi=150, format='png')

_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.scatterplot(data=df_sep,
                ax=ax
               )
ax.set(title='',
       xlabel='principal components',
       ylabel='retained variance',
       yscale='log'
      )
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ['$\mathrm{Re}$', '$\mathrm{Im}$'])
        
plt.tight_layout()
plt.savefig('./img/{}_sep.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}_sep.png'.format(args.output), dpi=150, format='png')