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

parser.add_argument('-d', '--data', type=str, help='dataset with the predictions(CSV)')
parser.add_argument('-a', '--algorithms', type=str, nargs='+', help='algorithms to display')
parser.add_argument('-l', '--labels', type=str, nargs='+', help='algorithms to display')
parser.add_argument('-o', '--output', type=str, help='output base name')

args = parser.parse_args()

# import the dataset
df = pd.read_csv(args.data)

# select the columns
if args.algorithms is not None:
    columns = ['exp_' + s for s in args.algorithms]
    df      = df[columns]

# plot all features
_, ax = plt.subplots(1, 1, figsize=(6, 5))

df = df.filter(items=columns)

sns.lineplot(data=df,
             ci=None,
             markers=True,
             markersize=8,
             color=sns.color_palette('muted', len(columns)),
             alpha=0.5,
             dashes=True,
             ax=ax
            )
ax.set(title='',
       xlabel='ID of the prediction',
       ylabel='exp',
       xticks=np.arange(df.shape[0])
      )
if args.labels is not None:
    ax.legend(args.labels)
        
plt.tight_layout()
plt.savefig('./img/{}.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}.png'.format(args.output), dpi=150, format='png')