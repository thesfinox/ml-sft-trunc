import pandas as pd
import os
import joblib
import json
import time
import argparse

# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-t', '--train', type=str, help='training dataset (CSV)')
parser.add_argument('-l', '--labels', type=str, help='labels (CSV)')
parser.add_argument('-e', '--estimator', type=str, help='model file')
parser.add_argument('-p', '--params', type=str, help='hyperparameters dictionary (JSON)')

args = parser.parse_args()

# load the datasets
X = pd.read_csv(args.train)
y = pd.read_csv(args.labels)

# load the estimator
estimator = joblib.load(args.estimator)

# load the parameters
if args.params != 'None' and args.params is not None:
    with open(args.params, 'r') as p:
        params = json.load(p)
        estimator.set_params(**params)
        
# fit the estimator
t = time.time()
estimator.fit(X, y)
t = time.time() - t
print('{} trained in {:.3f} seconds.'.format(estimator.__class__.__name__, t))

# save the fitted estimator
joblib.dump(estimator, args.estimator)