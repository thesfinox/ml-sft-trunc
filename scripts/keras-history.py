import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import argparse

def run_avg(values: pd.Series, window: int) -> np.ndarray:
    '''
    Compute the (simple) running average of a series of data.
    
    Required arguments:
        values: list of values,
        window: temporal window for the computation.
    '''
    values = values.values.reshape(-1,)
    avg    = np.zeros((np.shape(values)[0] - window + 1,))
    
    for i in range(avg.shape[0]):
        avg[i] = np.sum(values[i:i + window]) / window
    
    return avg

# set options
sns.set()
os.makedirs('./img', exist_ok=True)

# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, help='history (JSON)')
parser.add_argument('-o', '--output', type=str, help='basename of the output')

args = parser.parse_args()

# load the datasets
with open(args.data, 'r') as f:
    df = pd.DataFrame(json.load(f))
    
# compute the running average
win    = int(df.shape[0] / 10)
df_avg = df.apply(lambda s: run_avg(s, win))

# plot the scatter plot of the predictions
_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.lineplot(data=df[['loss', 'val_loss']],
             ax=ax
            )
ax.set(title='',
       xlabel='epoch',
       ylabel='loss',
       yscale='log'
      )
ax.legend(['training', 'validation'])
        
plt.tight_layout()
plt.savefig('./img/{}.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}.png'.format(args.output), dpi=150, format='png')

# plot the running average
_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.lineplot(data=df_avg[['loss', 'val_loss']],
             ax=ax
            )
ax.set(title='',
       xlabel='epoch',
       ylabel='loss',
       yscale='log'
      )
ax.legend(['training', 'validation'])
        
plt.tight_layout()
plt.savefig('./img/{}_avg.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}_avg.png'.format(args.output), dpi=150, format='png')