import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import joblib
import argparse

# set common options
sns.set()
os.makedirs('./img', exist_ok=True)

# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-f', '--features', type=str, help='list of the features')
parser.add_argument('-s', '--shapley', type=str, help='Shapley values')
parser.add_argument('-i', '--inter', type=str, help='interacting Shapley values')
parser.add_argument('-r', '--rank', type=str, help='variable ranking')
parser.add_argument('-o', '--output', type=str, help='basename of the output')

args = parser.parse_args()

# import the data
features = joblib.load(args.features)
shapley  = joblib.load(args.shapley)
inter    = joblib.load(args.inter)
rank     = joblib.load(args.rank)

# plot the variable ranking
_, ax = plt.subplots(1, 1, figsize=(6,5))

# plot the importance of the features
sns.barplot(x=100 * rank / np.sum(rank),
            y=features,
            palette=sns.color_palette('muted', len(features)),
            ax=ax
           )
ax.set(title='',
       xlabel='importance (%)'
      )
        
plt.tight_layout()
plt.savefig('./img/{}_rank.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}_rank.png'.format(args.output), dpi=150, format='png')

# plot the mean Shapley values
_, ax = plt.subplots(1, 1, figsize=(6,5))

shap_values = pd.DataFrame({'features': features,
                            'shapley': shapley.mean(axis=0)
                           }
                          ).sort_values('shapley', ascending=False)
sns.barplot(x='shapley',
            y='features',
            data=shap_values,
            palette=sns.color_palette('muted', len(features)),
            ax=ax
           )   
ax.set(title='',
       xlabel='average Shapley value',
       ylabel=''
      )
        
plt.tight_layout()
plt.savefig('./img/{}_shap.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}_shap.png'.format(args.output), dpi=150, format='png')

# choose one random matrix and plot the heatmap
_, ax = plt.subplots(1, 1, figsize=(6,5))

random = np.random.randint(inter.shape[0])
matrix = inter[random]

sns.heatmap(matrix,
            vmin=matrix.min(),
            vmax=matrix.max(),
            center=0.0,
            xticklabels=features,
            yticklabels=features,
            cmap='RdBu_r',
            ax=ax
           )
ax.set(title='')
        
plt.tight_layout()
plt.savefig('./img/{}_shap_int.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}_shap_int.png'.format(args.output), dpi=150, format='png')