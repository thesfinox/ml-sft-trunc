import pandas as pd
import numpy as np

import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
import joblib
import json
import time
import argparse


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from typing import Tuple

# set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def mse_ci(y_true: np.ndarray,
           y_pred: np.ndarray,
           dof: float,
           confidence: float = 0.95) -> Tuple[float, float]:
    '''
    Compute the confidence interval of the variance.
    
    Required arguments:
        y_true: true values,
        y_pred: predictions,
        dof:    the no. of degrees of freedom.
        
    Returns:
        the array of lower and upper bounds of the confidence interval.
    '''
    
    # compute the deviation of the data and the squared errors
    deviation = y_pred - y_true
    sq_errors = deviation ** 2

    # compute the confidence intervals
    conf_interval = stats.t.interval(confidence,
                                     dof,
                                     loc   = sq_errors.mean(),
                                     scale = stats.sem(sq_errors)
                                    )
    
    return conf_interval

# set options
os.makedirs('./metrics', exist_ok=True)
os.makedirs('./predictions', exist_ok=True)

# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-t', '--test', type=str, help='test dataset (CSV)')
parser.add_argument('-l', '--labels', type=str, help='labels (CSV)')
parser.add_argument('-m', '--model', type=str, help='model file')
parser.add_argument('-o', '--output', type=str, help='basename of the output')

args = parser.parse_args()

# load the datasets
X = pd.read_csv(args.test)
y = pd.read_csv(args.labels)

y = {'re': y['exp_re'],
     'im': y['exp_im']
    }

# load the estimator
ann_mod = tf.keras.models.load_model(args.model)
        
# compute the predictions
t = time.time()
y_pred = ann_mod.predict(X)
t = time.time() - t
print('Model predicted in {:.3f} seconds.'.format(t))

# compute RMSE, MSE, MAE, R2
dof  = X.shape[0] - X.shape[1]
mse  = mean_squared_error(y, y_pred)
ci   = mse_ci(y, y_pred, dof)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y, y_pred)
r2   = r2_score(y, y_pred)

metrics = {'DOF': dof,
           'MSE': mse,
           'MSE 95% CI (lower)': ci[0],
           'MSE 95% CI (upper)': ci[1],
           'RMSE': rmse,
           'MAE': mae,
           'R2': r2
          }

with open('./metrics/{}.json'.format(args.output), 'w') as m:
    json.dump(metrics, m)
    
# save the predictions to file
pred = pd.DataFrame(zip(y, y_pred), columns=['exp', 'pred'])
pred = X.join(pred)
pred.to_csv('./predictions/{}.csv'.format(args.output), index=False)