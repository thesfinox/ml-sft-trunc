import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

# set options
sns.set()
os.makedirs('./img', exist_ok=True)

# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, help='dataset (CSV)')
parser.add_argument('-o', '--output', type=str, help='basename of the output')

args = parser.parse_args()

# load the datasets
df = pd.read_csv(args.data)

# plot the scatter plot of the predictions
_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.lineplot(data=df[['exp', 'pred']],
             ci=None,
             markers=True,
             markersize=8,
             color=sns.color_palette('muted', 2),
             alpha=0.5,
             dashes=True,
             ax=ax
            )
ax.set(title='',
       xlabel='ID of the prediction',
       ylabel='exp'
      )
        
plt.tight_layout()
plt.savefig('./img/{}_lineplot.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}_lineplot.png'.format(args.output), dpi=150, format='png')

# plot the histogram of the residuals
_, ax = plt.subplots(1, 1, figsize=(6,5))

residual = df['exp'] - df['pred']
sns.distplot(residual,
             bins=10,
             kde=False,
             ax=ax
            )
ax.set(title='',
       xlabel='residual',
       ylabel='count',
       xticks=np.round(np.linspace(np.min(residual), np.max(residual), 10), 2)
      )
        
plt.tight_layout()
plt.savefig('./img/{}_histogram.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}_histogram.png'.format(args.output), dpi=150, format='png')

# residual plot
_, ax = plt.subplots(1, 1, figsize=(6,5))

residual = df['exp'] - df['pred']
sns.scatterplot(x=df['pred'],
                y=residual,
                ax=ax
               )
ax.axhline(0, color='black', ls='--')
ax.set(title='',
       xlabel='prediction',
       ylabel='residual'
      )
        
plt.tight_layout()
plt.savefig('./img/{}_resplot.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}_resplot.png'.format(args.output), dpi=150, format='png')