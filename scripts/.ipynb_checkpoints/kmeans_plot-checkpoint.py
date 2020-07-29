import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
import time
import argparse

from sklearn.cluster import KMeans

# set common options
sns.set()
os.makedirs('./img', exist_ok=True)
os.makedirs('./data', exist_ok=True)

# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-e', '--exp', type=str, help='Labels dataset (CSV)')
parser.add_argument('-k', '--kmeans', type=str, help='KMeans input dataset (CSV)')
parser.add_argument('-p', '--pca', type=str, help='PCA components dataset (CSV)')
parser.add_argument('-o', '--output', type=str, help='basename of the output plots')

args = parser.parse_args()

# import the datasets
exp    = pd.read_csv(args.exp)
kmeans = pd.read_csv(args.kmeans)
pca    = pd.read_csv(args.pca)

# merge the dataframe
columns = ['pca_1', 'pca_2', 'kmeans', 'exp']
df = pd.concat([pca, kmeans, exp], axis=1, ignore_index=True)
df = df.rename(columns=dict(zip(df.columns, columns)))
df.to_csv('./data/{}_plot.csv'.format(args.output), index=False)

# compute the number of colours
n_cls = len(df['exp'].unique())

# plot the results
_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.scatterplot(x='pca_1',
                y='pca_2',
                data=df,
                hue='kmeans',
                style='exp',
                palette=sns.color_palette('muted', n_cls),
                alpha=0.5,
                ax=ax
               )
ax.set(title='',
       xlabel='PCA #1',
       ylabel='PCA #2'
      )
        
plt.tight_layout()
plt.savefig('./img/{}.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}.png'.format(args.output), dpi=150, format='png')