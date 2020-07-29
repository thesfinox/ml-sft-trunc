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

parser.add_argument('-i', '--input', type=str, help='input dataset (JSON)')
parser.add_argument('-o', '--output', type=str, help='basename of the output plots')

args = parser.parse_args()

# import the dataset
df = pd.read_json(args.input, orient='split')

# get the length of each list and find the unique values
shapes = df.applymap(len)
shapes = shapes.apply(lambda x: np.unique(x).squeeze(), axis=1)

# plot the counts for each shape (treat it as categorical)
_, ax = plt.subplots(1, 1, figsize=(6, 5))

sns.countplot(x=shapes.values,
              palette='muted',
              ax=ax
             )
ax.set(xlabel='length of the solutions',
       ylabel='counts'
      )

plt.tight_layout()
plt.savefig('./img/{}.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}.png'.format(args.output), dpi=150, format='png')