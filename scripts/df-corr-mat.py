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
df = pd.read_csv(args.input)

# plot the distribution
_, ax = plt.subplots(1, 1, figsize=(6,5))

# drop the column type since it's categorical
df = df.drop(columns=['solutions', 'init'])

sns.heatmap(df.corr(),
            vmin=-1.0,
            vmax=1.0,
            cmap='RdBu_r',
            ax=ax
           )
        
plt.tight_layout()
plt.savefig('./img/{}.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}.png'.format(args.output), dpi=150, format='png')