import pandas as pd
import numpy as np
import os
import joblib
import json
import argparse
import sklearn

from scipy import stats

def summary(estimator: sklearn.base.BaseEstimator,
            X: pd.Series,
            y_true: np.ndarray,
            y_pred: np.ndarray
           ) -> pd.DataFrame:
    '''
    Build a summary of the linear regression.
    
    Required arguments:
        estimator: the linear fit model,
        X:         the predictor,
        y:         the estimand.
        
    Returns:
        the statistics on the coefficients.
    '''
    # compute the no. of dof and the coefficients
    dof  = X.shape[0] - X.shape[1]
    coef = estimator.coef_
    
    # compute residuals, residual variance and the square of deviations
    res = y_true - y_pred
    var = np.sum(res**2) / dof
    ssx = np.sum(np.square(X.values - np.mean(X.values, axis=0)), axis=0)
    
    # compute standard error, t coeffcient and p-value that t_obs > |t|
    se  = np.sqrt(var / (ssx + 1.0e-6)) #---------- avoid division by zero
    t   = coef / (se + 1.0e-6) #------------------- avoid division by zero
    p   = 2 * (1.0 - stats.t.cdf(abs(t), dof))
    
    # compute confidence intervals (two sided)
    intervals = stats.t.interval(0.975,
                                 dof,
                                 loc=coef,
                                 scale=se
                                )
    
    # create the dataframe
    summary = {'coefficients':          np.round(coef, 10),
               'standard error':        np.round(se, 10),
               't statistic':           np.round(t, 3),
               'p value (t_obs > |t|)': np.round(p, 3),
               '95% CI (lower)':        np.round(intervals[0], 10),
               '95% CI (upper)':        np.round(intervals[1], 10)
              }
    
    return pd.DataFrame(summary, index=X.columns)


# set options
os.makedirs('./metrics', exist_ok=True)
os.makedirs('./predictions', exist_ok=True)

# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, help='dataset (CSV)')
parser.add_argument('-e', '--estimator', type=str, help='model file')
parser.add_argument('-o', '--output', type=str, help='basename of the output')

args = parser.parse_args()

# load the datasets
df     = pd.read_csv(args.data)
X      = df.drop(columns=['exp', 'pred'])
y_true = df['exp'].values.reshape(-1,)
y_pred = df['pred'].values.reshape(-1,)

# load the estimator
estimator = joblib.load(args.estimator)
        
# compute the analysis of variance
anova = summary(estimator, X, y_true, y_pred)
anova.to_csv('./metrics/{}.csv'.format(args.output))