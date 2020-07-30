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

args = parser.parse_args()

# import the dataset
df = pd.read_csv(args.input, index_col=0)

# plot the distribution
_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.heatmap(df,
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            cmap='RdBu_r',
            ax=ax
           )
ax.set(title='',
       yticks=np.arange(df.shape[0]) + 0.5,
       xticks=np.arange(df.shape[1]) + 0.5
      )
ax.set_xticklabels(df.columns.tolist(), rotation=90, ha='center', va='top')
ax.set_yticklabels(df.index.tolist(), rotation=0, ha='right', va='center')
        
plt.tight_layout()
plt.savefig('./img/{}.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}.png'.format(args.output), dpi=150, format='png')