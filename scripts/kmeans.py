import pandas as pd
import os
import joblib
import time
import argparse

from sklearn.cluster import KMeans

# set common options
os.makedirs('./data', exist_ok=True)

# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', type=str, help='input dataset (CSV)')
parser.add_argument('-p', '--pca', type=str, help='PCA components dataset (CSV)')
parser.add_argument('-c', '--labels', type=int, default=3, help='number of cluster labels')
parser.add_argument('-o', '--output', type=str, help='basename of the output plots')
parser.add_argument('-r', '--rand', type=int, default=42, help='random seed')

args = parser.parse_args()

# import the datasets
df = pd.read_csv(args.input)

# import PCA data
pca = pd.read_csv(args.pca)

# compute the clustering
t = time.time()
kmeans = KMeans(n_clusters=args.labels, random_state=args.rand).fit_predict(df)
t = time.time() - t
print('KMeans clustering computed in {:.3f} seconds'.format(t))

# create a dataframe
kmeans = pd.DataFrame(kmeans, columns=['kmeans'])
kmeans.to_csv('./data/{}.csv'.format(args.output), index=False)