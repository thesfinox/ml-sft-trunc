import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import logging
import os

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from typing import Tuple

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
        a tuple with lower and upper bounds of the confidence interval.
    '''
    
    # compute the deviation of the data and the squared errors
    deviation = y_pred - y_true
    sq_errors = np.square(deviation)

    # compute the confidence intervals
    conf_interval = stats.t.interval(confidence,
                                     dof,
                                     loc   = sq_errors.mean(),
                                     scale = stats.sem(sq_errors)
                                    )
    
    return conf_interval

# set less logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-tx', '--test-x', type=str, help='test features (CSV)')
parser.add_argument('-ty', '--test-y', type=str, help='test labels (CSV)')
parser.add_argument('-m', '--model', type=str, help='model file (HDF5)')
parser.add_argument('-o', '--output', type=str, help='basename of the output')

args = parser.parse_args()

# load files
model = tf.keras.models.load_model(args.model)

X_test = pd.read_csv(args.test_x)
y_test = pd.read_csv(args.test_y)

# transform labels
y_test = {f: y_test[f].values.reshape(-1,) for f in y_test}

# compute predictions
y_pred = model.predict(X_test)
y_pred = {k: v.reshape(-1,) for k, v in y_pred.items()}

pd.DataFrame(y_pred).to_csv('./data/y_{}_pred.csv'.format(args.output), index=False)

# compute metrics
dof = X_test.shape[0] - X_test.shape[1]
dof = {f: dof for f in ['exp_re', 'exp_im']}
mse = {f: mean_squared_error(y_test[f], y_pred[f]) for f in ['exp_re', 'exp_im']}
mae = {f: mean_absolute_error(y_test[f], y_pred[f]) for f in ['exp_re', 'exp_im']}
r2  = {f: r2_score(y_test[f], y_pred[f]) for f in ['exp_re', 'exp_im']}

ci_low  = {f: mse_ci(y_test[f], y_pred[f], dof[f])[0] for f in ['exp_re', 'exp_im']}
ci_high = {f: mse_ci(y_test[f], y_pred[f], dof[f])[1] for f in ['exp_re', 'exp_im']}

metrics = {f: [m[f] for m in [dof, mse, ci_low, ci_high, mae, r2]] \
                    for f in ['exp_re', 'exp_im']}
metrics = pd.DataFrame(metrics,
                       index=['dof', 'mse', 'mse_ci_low', 'mse_ci_high', 'mae', 'r2'],
                       columns=['exp_re', 'exp_im']
                      )
metrics.to_csv('./data/{}_metrics.csv'.format(args.output))