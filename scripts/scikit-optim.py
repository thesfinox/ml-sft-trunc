import pandas as pd
from skopt import gp_minimize
from skopt.utils import use_named_args
import numpy as np
import os
import joblib
import json
import time
import argparse

from sklearn.metrics import mean_squared_error

# basic definitions
os.makedirs('./hypers', exist_ok=True)

# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-t', '--train', type=str, help='training dataset (CSV)')
parser.add_argument('-v', '--val', type=str, help='validation dataset (CSV)')
parser.add_argument('-lt', '--trainlabels', type=str, help='training labels (CSV)')
parser.add_argument('-lv', '--vallabels', type=str, help='validation labels (CSV)')
parser.add_argument('-e', '--estimator', type=str, help='model file')
parser.add_argument('-p', '--params', type=str, help='hyperparameters space')
parser.add_argument('-n', '--calls', type=int, default=100, help='no. of calls')
parser.add_argument('-r', '--rand', type=int, default=42, help='random seed')
parser.add_argument('-o', '--output', type=int, default=42, help='basename of the output')

args = parser.parse_args()

# load the datasets
X_train = pd.read_csv(args.train)
y_train = pd.read_csv(args.trainlabels).values.reshape(-1,2)

X_val = pd.read_csv(args.val)
y_val = pd.read_csv(args.vallabels).values.reshape(-1,2)

# load the estimator
estimator = joblib.load(args.estimator)

# load the parameters
params = joblib.load(args.params)
names  = [p.name for p in params]
        
# fit the estimator
t = time.time()

# minimise the objective
@use_named_args(params)
def objective(**args):
    '''
    Compute the objective function.
    
    Arguments:
        **args: arguments to pass to the estimator
    '''

    # fit the estimator
    estimator.set_params(**args)
    estimator.fit(X_train, y_train)

    # compute predictions on the validation set
    y_val_pred = estimator.predict(X_val).reshape(-1,2)
    
    return mean_squared_error(y_val, y_val_pred)

# compute the minimisation
results = gp_minimize(objective,
                      params,
                      n_calls=args.calls,
                      random_state=args.rand,
                      n_jobs=-1
                     )

t = time.time() - t
print('{} optimised in {:.3f} seconds.'.format(estimator.__class__.__name__, t))

# save the best hyperparameters
hypers = pd.DataFrame(zip(names, results.x), columns=['names', 'values'])
hypers.to_csv('./hypers/{}.csv'.format(estimator.__class__.__name__.lower()), index=False)

# train over the entire set
estimator.set_params(**dict(zip(names, results.x)))

t = time.time()
estimator.fit(X_train, y_train)
t = time.time() - t
print('{} trained in {:.3f} seconds.'.format(estimator.__class__.__name__, t))

# save the fitted estimator
joblib.dump(estimator, args.estimator)