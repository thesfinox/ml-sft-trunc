import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# set common options
sns.set()
os.makedirs('./img', exist_ok=True)

# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-e', '--errors', type=str, help='errors dataset (CSV)')
parser.add_argument('-o', '--output', type=str, help='basename of the output file')

args = parser.parse_args()

# import the dataset
metrics = pd.read_csv(args.errors)

# plot all features
_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.scatterplot(data=metrics[['training', 'validation']],
                ax=ax
               )
ax.set(title='',
       xlabel='validation set ratio',
       ylabel='MSE',
       yscale='log',
       xticks=np.arange(metrics.shape[0])
      )
ax.set_xticklabels(np.round(metrics['size'], 2),
                   rotation=45,
                   ha='right',
                   va='top'
                  )

plt.tight_layout()
plt.savefig('./img/{}.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}.png'.format(args.output), dpi=150, format='png')