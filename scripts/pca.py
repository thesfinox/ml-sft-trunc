import pandas as pd
import numpy as np
import os
import time
import argparse

from sklearn.decomposition import PCA

# create data directory
os.makedirs('./data', exist_ok=True)

# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', type=str, help='input dataset (CSV)')
parser.add_argument('-c', '--comp', type=int, default=2, help='number of components')
parser.add_argument('-o', '--output', type=str, help='basename of the output plots')
parser.add_argument('-r', '--rand', type=str, help='random seed')

args = parser.parse_args()

# import the datasets
df = pd.read_csv(args.input)

# define the columns
columns = ['pca_' + str(n+1) for n in range(args.comp)]

# compute the PCA
t = time.time()
pca = PCA(n_components=args.comp, random_state=args.rand).fit_transform(df)
pca = pd.DataFrame(pca, columns=columns)
t = time.time() - t
print('PCA computed in {:.3f} seconds'.format(t))

# save the PCA components
pca.to_csv('./data/{}.csv'.format(args.output), index=False)