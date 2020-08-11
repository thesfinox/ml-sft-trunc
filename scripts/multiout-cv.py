import pandas as pd
import os
import joblib
import json
import time
import argparse
import re

from skopt import BayesSearchCV

# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-t', '--train', type=str, help='training dataset (CSV)')
parser.add_argument('-l', '--labels', type=str, help='labels (CSV)')
parser.add_argument('-e', '--estimator', type=str, help='model file')
parser.add_argument('-p', '--params', type=str, help='hyperparameters space (Pickle - scikit-optimize)')
parser.add_argument('-cv', '--cv', type=int, default=5, help='cross validation splits')
parser.add_argument('-n', '--iter', type=int, default=10, help='no. of iterations')
parser.add_argument('-r', '--rand', type=int, default=42, help='random state')

args = parser.parse_args()

# load the datasets
X = pd.read_csv(args.train)
y = pd.read_csv(args.labels)

# load the estimator
estimator = joblib.load(args.estimator)
space     = joblib.load(args.params)

# create the estimator
estimator = BayesSearchCV(estimator,
                          space,
                          n_iter=args.iter,
                          scoring='neg_mean_squared_error',
                          n_jobs=-1,
                          cv=args.cv,
                          random_state=args.rand
                         )
        
# fit the estimator
t = time.time()
estimator.fit(X, y)
t = time.time() - t
print('{} trained in {:.3f} seconds.'.format(estimator.__class__.__name__, t))

# save the fitted estimator
joblib.dump(estimator.best_estimator_, args.estimator)

# save the parameters
filename = re.sub('pkl', 'json', args.params)
with open(filename, 'w') as f:
    json.dump(estimator.best_params_, f)