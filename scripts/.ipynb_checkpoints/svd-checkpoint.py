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

parser.add_argument('-l', '--low', type=str, help='input dataset (CSV) with low weights')
parser.add_argument('-u', '--high', type=str, help='input dataset (CSV) with high weights')
parser.add_argument('-o', '--output', type=str, help='basename of the output plots')

args = parser.parse_args()

# import the datasets
df_low  = pd.read_csv(args.low)
df_high = pd.read_csv(args.high)

# compute the SVD
t = time.time()
var_low, var_high = joblib.Parallel(n_jobs=-1)(joblib.delayed(svd)(m) for m in [df_low, df_high])
t = time.time() - t
print('SVD computed in {:.3f} seconds'.format(t))

# create a dataframe
df = pd.DataFrame(zip(var_low, var_high), columns=['weight < 1.5', 'weight ≥ 1.5'])
df.to_csv('./data/{}.csv'.format(args.output), index=False)

# plot the retained variance
_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.scatterplot(data=df,
                ax=ax
               )
ax.set(title='',
       xlabel='principal components',
       ylabel='retained variance',
       yscale='log'
      )
        
plt.tight_layout()
plt.savefig('./img/{}.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}.png'.format(args.output), dpi=150, format='png')