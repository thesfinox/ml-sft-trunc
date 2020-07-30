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
    # compute the no. of dof and the coefficients (shape: 2 x coef)
    dof  = X.shape[0] - X.shape[1]
    coef = estimator.coef_
    
    # compute residuals (shape: n x 2),
    # residual variance (shape: 2 x 1),
    # and the square of deviations (shape: coef x 1)
    res = y_true - y_pred
    var = np.sum(np.square(res), axis=0) / dof
    ssx = np.sum(np.square(X.values - np.mean(X.values, axis=0)), axis=0)
    
    # compute standard error, t coefficient and p-value that t_obs > |t|
    se_re = np.sqrt(var[0] / (ssx + 1.0e-6)) #------- avoid division by zero
    se_im = np.sqrt(var[1] / (ssx + 1.0e-6)) #------- avoid division by zero
    t_re  = coef[0,:] / (se_re + 1.0e-6) #------------- avoid division by zero
    t_im  = coef[1,:] / (se_im + 1.0e-6) #------------- avoid division by zero
    p_re  = 2 * (1.0 - stats.t.cdf(abs(t_re), dof))
    p_im  = 2 * (1.0 - stats.t.cdf(abs(t_im), dof))
    
    # compute confidence intervals (two sided)
    intervals_re = stats.t.interval(0.975,
                                    dof,
                                    loc=coef[0,:],
                                    scale=se_re
                                   )
    intervals_im = stats.t.interval(0.975,
                                    dof,
                                    loc=coef[1,:],
                                    scale=se_im
                                   )
    
    # create the dataframe
    summary = {'coefficients [Re(exp)]':          coef[0,:],
               'coefficients [Im(exp)]':          coef[1,:],
               'standard error [Re(exp)]':        se_re,
               'standard error [Im(exp)]':        se_im,
               't statistic [Re(exp)]':           np.round(t_re, 3),
               't statistic [Im(exp)]':           np.round(t_im, 3),
               'p value (t_obs > |t|) [Re(exp)]': np.round(p_re, 3),
               'p value (t_obs > |t|) [Im(exp)]': np.round(p_im, 3),
               '95% CI (lower) [Re(exp)]':        intervals_re[0],
               '95% CI (upper) [Re(exp)]':        intervals_re[1],
               '95% CI (lower) [Im(exp)]':        intervals_im[0],
               '95% CI (upper) [Im(exp)]':        intervals_im[1]
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
X      = df.drop(columns=['exp_re', 'exp_im', 'pred_re', 'pred_im'])
y_true = df[['exp_re', 'exp_im']].values.reshape(-1,2)
y_pred = df[['pred_re', 'pred_im']].values.reshape(-1,2)

# load the estimator
estimator = joblib.load(args.estimator)
        
# compute the analysis of variance
anova = summary(estimator, X, y_true, y_pred)
anova.to_csv('./metrics/{}.csv'.format(args.output))