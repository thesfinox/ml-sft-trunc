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
    deviation = y_pred.reshape(-1,) - y_true.reshape(-1,)
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

y = {'re': y['exp_re'].values.reshape(-1,),
     'im': y['exp_im'].values.reshape(-1,)
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

mse = {}
mse['re'] = mean_squared_error(y['re'], y_pred['re'])
mse['im'] = mean_squared_error(y['im'], y_pred['im'])

ci = {}
ci['re'] = mse_ci(y['re'], y_pred['re'], dof)
ci['im'] = mse_ci(y['im'], y_pred['im'], dof)

rmse = {}
rmse['re'] = np.sqrt(mse['re'])
rmse['im'] = np.sqrt(mse['im'])

mae = {}
mae['re'] = mean_absolute_error(y['re'], y_pred['re'])
mae['im'] = mean_absolute_error(y['im'], y_pred['im'])

r2 = {}
r2['re'] = r2_score(y['re'], y_pred['re'])
r2['im'] = r2_score(y['im'], y_pred['im'])

metr_re = {'DOF': dof,
           'MSE': mse['re'].tolist(),
           'MSE 95% CI (lower)': ci['re'][0].tolist(),
           'MSE 95% CI (upper)': ci['re'][1].tolist(),
           'RMSE': rmse['re'].tolist(),
           'MAE': mae['re'].tolist(),
           'R2': r2['re'].tolist()
          }
metr_im = {'DOF': dof,
           'MSE': mse['im'].tolist(),
           'MSE 95% CI (lower)': ci['im'][0].tolist(),
           'MSE 95% CI (upper)': ci['im'][1].tolist(),
           'RMSE': rmse['im'].tolist(),
           'MAE': mae['im'].tolist(),
           'R2': r2['im'].tolist()
          }

with open('./metrics/{}_re.json'.format(args.output), 'w') as m:
    json.dump(metr_re, m)
with open('./metrics/{}_im.json'.format(args.output), 'w') as m:
    json.dump(metr_im, m)
    
# save the predictions to file
pred = pd.DataFrame(np.c_[y['re'], y['im'], y_pred['re'], y_pred['im']], columns=['exp_re', 'exp_im', 'pred_re', 'pred_im'])
pred = X.join(pred)
pred.to_csv('./predictions/{}.csv'.format(args.output), index=False)