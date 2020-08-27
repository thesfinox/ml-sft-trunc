import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set()

# shortcuts
proot = lambda s: os.path.join('.', s)
pdata = lambda s: os.path.join(proot('data'), s)
pimg  = lambda s: os.path.join(proot('img'), s)
pmet  = lambda s: os.path.join(proot('metrics'), s)
pmod  = lambda s: os.path.join(proot('models'), s)

def metric(y_true, y_finite, y_pred, tensor=False):
    '''
    Compute the evaluation metric.
    
    Arguments:
        y_true:   the true values,
        y_finite: the finite truncation level,
        y_pred:   the predicted labels.
        
    Optional:
        tensor: whether to return a tensor value (True) or a float (False).
        
    Returns:
        the logarithm of the average ratio between finite and predicted residuals.
    '''
    if type(y_true) is list:
        y_true = np.array(y_true)
    if type(y_finite) is list:
        y_finite = np.array(y_finite)
    if type(y_pred) is list:
        y_pred = np.array(y_pred)
    
    # compute the metric (use Tensorflow or Numpy)
    if tensor:
        
        # compute residuals (avoid division by zero)
        res_pred = tf.math.subtract(y_true, y_pred)
        res_fin  = tf.math.subtract(y_true, y_finite)
        res_fin  = tf.math.add(res_fin, 1.0e-6)

        # compute the ratio
        ratio = tf.math.divide(res_pred, res_fin)
        ratio = tf.math.abs(ratio)

        # compute the logarithm
        log_ratio = tf.math.log(ratio) / tf.math.log(tf.constant(10, dtype=ratio.dtype))
        log_ratio = tf.math.reduce_mean(log_ratio)
        
        return log_ratio
    
    else:
        
        # compute residuals (avoid division by zero)
        res_pred = np.subtract(y_true, y_pred)
        res_fin  = np.subtract(y_true, y_finite)
        res_fin  = np.add(res_fin, 1.0e-6)

        # compute the ratio
        ratio = np.divide(res_pred, res_fin)
        ratio = np.abs(ratio)

        # compute the logarithm
        log_ratio = np.log10(ratio)
        log_ratio = np.mean(log_ratio)
        
        return log_ratio

def statisticsCV(estimator, cv):
    '''
    Compute and display statistics.
    
    Arguments:
        estimator: the BayesSearchCV estimator,
        cv:        the cross-validation object.
        
    Returns:
        the best estimator, the CV score (tuple of mean and std), the best hyperparameters, the predictions.
    '''
    # get the best estimator
    best_estimator = estimator.best_estimator_
    
    if type(cv) is float:
        splits = cv
    else:
        splits = cv.n_splits
    
    # compute CV score
    results = estimator.cv_results_
    cv_mean = np.mean(results['mean_test_score'])
    cv_std  = np.mean(results['std_test_score']) / splits
    
    # display the hyperparameters
    hyperparameters = pd.DataFrame(estimator.best_params_,
                                   index=[best_estimator.__class__.__name__.lower()]
                                  )
    hyperparameters = hyperparameters.transpose()
    
    return best_estimator, (cv_mean, cv_std), hyperparameters

def make_predictions(estimator, X, y, y_finite, name=None, suffix='', tensor=False):
    '''
    Compute predictions and metrics.
    
    Arguments:
        estimator: the fitted estimator,
        X:         the features,
        y:         the labels,
        y_finite:  the finite level truncations.
        
    Optional:
        name:   the name of the metrics (e.g. 'train', 'test', etc.),
        suffix: the suffix of the file name,
        tensor: use Tensorflow (True) or NumPy (False).
        
    Returns:
        the metrics.
    '''
    if type(X) is pd.DataFrame:
        X = X.values
    if type(y) is pd.Series:
        y = y.values
    if type(y) is list:
        y = np.array(y)
    if type(y_finite) is pd.Series:
        y_finite = y_finite.values
    if type(y_finite) is list:
        y_finite = np.array(y_finite)
        
    # compute predictions
    y_true   = y.reshape(-1,)
    y_finite = y_finite.reshape(-1,)
    y_pred   = estimator.predict(X).reshape(-1,)
    
    # save predictions
    if name is not None and type(name) is str:
        y_pred_csv = pd.DataFrame({'exp_' + name: y_pred})
        y_pred_csv.to_csv(pdata('lumps_y_' + name + '_predictions' + suffix + '.csv'), index=False)
    else:
        y_pred_csv = pd.DataFrame({'exp': y_pred})
        y_pred_csv.to_csv(pdata('lumps_y_predictions' + suffix + '.csv'), index=False)
    
    # compute the metrics
    metrics = {'mean_squared_error':  mean_squared_error(y_true, y_pred),
               'mean_absolute_error': mean_absolute_error(y_true, y_pred),
               'r2_score':            r2_score(y_true, y_pred),
               'residual_ratio':      float(metric(y_true, y_finite, y_pred, tensor=tensor))
              }
    
    if name is not None and type(name) is str:
        return pd.DataFrame(metrics, index=[name])
    else:
        return pd.DataFrame(metrics, index=['predictions'])

def make_plots(estimator, X, y, y_finite, name=None, suffix='', figsize=(6,5)):
    '''
    Draw the histogram and residual plots.
    
    Arguments:
        estimator: the fitted estimator,
        X:         the features,
        y:         the labels,
        y_finite:  the finite truncation levels.
        
    Optional:
        name:    the name of the plots (e.g. 'train', 'test', etc.),
        suffix:  the suffix of the file name,
        figsize: the size of the subplots.
    '''
    if type(X) is pd.DataFrame:
        X = X.values
    if type(y) is pd.Series:
        y = y.values
    if type(y) is list:
        y = np.array(y)
    if type(y_finite) is pd.Series:
        y_finite = y_finite.values
    if type(y_finite) is list:
        y_finite = np.array(y_finite)
        
    # compute predictions and residuals
    y_true = y.reshape(-1,)
    y_pred = estimator.predict(X).reshape(-1,)
    resid_pred  = np.subtract(y_true, y_pred)
    resid_fin   = np.subtract(y_true, y_finite.reshape(-1,)) + 1.0e-6
    resid_ratio = np.log10(np.abs(np.divide(resid_pred, resid_fin)))
    
    # draw univariate distribution
    fig, hist_res = plt.subplots(1, 1, figsize=figsize)
    
    sns.distplot(resid_pred,
                 kde=False,
                 ax=hist_res
                )
    hist_res.set(title='',
                 xlabel='residual',
                 ylabel='count',
                 yscale='log'
                )
    
    plt.tight_layout()
    if name is not None and type(name) is str:
        plt.savefig(pimg(f'lumps_{name}_residual_histogram{suffix}.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg(f'lumps_{name}_residual_histogram{suffix}.png'), dpi=150, format='png')
    else:
        plt.savefig(pimg('lumps_residual_histogram' + suffix + '.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg('lumps_residual_histogram' + suffix + '.png'), dpi=150, format='png')
    
    # close the figure
    plt.close(fig)
    
    # draw ratio univariate distribution
    fig, hist_ratio = plt.subplots(1, 1, figsize=figsize)
    
    sns.distplot(resid_ratio,
                 kde=False,
                 ax=hist_ratio
                )
    hist_ratio.set(title='',
                   xlabel='$\log_{10}$ of the ratio of residuals',
                   ylabel='count',
                   yscale='log'
                  )
    
    plt.tight_layout()
    if name is not None and type(name) is str:
        plt.savefig(pimg(f'lumps_{name}_ratio_histogram{suffix}.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg(f'lumps_{name}_ratio_histogram{suffix}.png'), dpi=150, format='png')
    else:
        plt.savefig(pimg('lumps_ratio_histogram' + suffix + '.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg('lumps_ratio_histogram' + suffix + '.png'), dpi=150, format='png')
    
    # close the figure
    plt.close(fig)
    
    # draw residual scatter plot
    fig, scat_res = plt.subplots(1, 1, figsize=figsize)
    
    sns.scatterplot(x=y_pred,
                    y=resid_pred,
                    alpha=0.5,
                    ax=scat_res
                   )
    scat_res.set(title='',
                 xlabel='prediction',
                 ylabel='residual'
                )
    
    plt.tight_layout()
    if name is not None and type(name) is str:
        plt.savefig(pimg(f'lumps_{name}_residual_scatterplot{suffix}.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg(f'lumps_{name}_residual_scatterplot{suffix}.png'), dpi=150, format='png')
    else:
        plt.savefig(pimg('lumps_residual_scatterplot' + suffix + '.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg('lumps_residual_scatterplot' + suffix + '.png'), dpi=150, format='png')
    
    # close the figure
    plt.close(fig)
    
    # draw ratio scatter plot
    fig, scat_ratio = plt.subplots(1, 1, figsize=figsize)
    
    sns.scatterplot(x=y_pred,
                    y=resid_ratio,
                    alpha=0.5,
                    ax=scat_ratio
                   )
    scat_ratio.set(title='',
                   xlabel='prediction',
                   ylabel='$\log_{10}$ of the ratio of residuals'
                  )
    
    plt.tight_layout()
    if name is not None and type(name) is str:
        plt.savefig(pimg(f'lumps_{name}_ratio_scatterplot{suffix}.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg(f'lumps_{name}_ratio_scatterplot{suffix}.png'), dpi=150, format='png')
    else:
        plt.savefig(pimg('lumps_ratio_scatterplot' + suffix + '.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg('lumps_ratio_scatterplot' + suffix + '.png'), dpi=150, format='png')
    
    # close the figure
    plt.close(fig)

def compare_plots(estimator, X_train, y_train, y_train_finite, X_test, y_test, y_test_finite, legend=['traning', 'test'], name=None, suffix='', figsize=(6,5)):
    '''
    Draw the histogram and residual plots.
    
    Arguments:
        estimator:      the fitted estimator,
        X_train:        the training features,
        y_train:        the training labels,
        y_train_finite: the training finite level truncations,
        X_test:         the test features,
        y_test:         the test labels,
        y_test_finite:  the test finite level truncations.
        
    Optional:
        legend:  list of legend labels (e.g. ['training', 'validation'])
        name:    the name of the plots (e.g. 'lr', 'svr', etc.),
        suffix:  the suffix of the file name,
        figsize: the size of the subplots.
    '''
    if type(X_train) is pd.DataFrame:
        X_train = X_train.values
    if type(y_train) is pd.Series:
        y_train = y_train.values
    if type(y_train) is list:
        y_train = np.array(y_train)
    if type(y_train_finite) is pd.Series:
        y_train_finite = y_train_finite.values
    if type(y_train_finite) is list:
        y_train_finite = np.array(y_train_finite)
    if type(X_test) is pd.DataFrame:
        X_test = X_test.values
    if type(y_test) is pd.Series:
        y_test = y_test.values
    if type(y_test) is list:
        y_test = np.array(y_test)
    if type(y_test_finite) is pd.Series:
        y_test_finite = y_test_finite.values
    if type(y_test_finite) is list:
        y_test_finite = np.array(y_test_finite)
        
    # compute predictions and residuals
    y_train_true = y_train.reshape(-1,)
    y_train_pred = estimator.predict(X_train).reshape(-1,)
    y_test_true = y_test.reshape(-1,)
    y_test_pred = estimator.predict(X_test).reshape(-1,)
    
    resid_train_pred  = np.subtract(y_train_true, y_train_pred)
    resid_train_fin   = np.subtract(y_train_true, y_train_finite) + 1.0e-6
    resid_train_ratio = np.log10(np.abs(np.divide(resid_train_pred, resid_train_fin)))
    resid_test_pred  = np.subtract(y_test_true, y_test_pred)
    resid_test_fin   = np.subtract(y_test_true, y_test_finite) + 1.0e-6
    resid_test_ratio = np.log10(np.abs(np.divide(resid_test_pred, resid_test_fin)))
    
    # draw univariate distribution
    fig, hist_res = plt.subplots(1, 1, figsize=figsize)
    
    sns.distplot(resid_train_pred,
                 kde=False,
                 ax=hist_res
                )
    sns.distplot(resid_test_pred,
                 kde=False,
                 ax=hist_res
                )
    hist_res.set(title='',
                 xlabel='residual',
                 ylabel='count',
                 yscale='log'
                )
    hist_res.legend(legend, loc='best')
    
    plt.tight_layout()
    if name is not None and type(name) is str:
        plt.savefig(pimg(f'lumps_{name}_residual_histogram_compare{suffix}.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg(f'lumps_{name}_residual_histogram_compare{suffix}.png'), dpi=150, format='png')
    else:
        plt.savefig(pimg('lumps_residual_histogram_compare' + suffix + '.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg('lumps_residual_histogram_compare' + suffix + '.png'), dpi=150, format='png')
    
    # close the figure
    plt.close(fig)
    
    # draw ratio univariate distribution
    fig, hist_ratio = plt.subplots(1, 1, figsize=figsize)
    
    sns.distplot(resid_train_ratio,
                 kde=False,
                 ax=hist_ratio
                )
    sns.distplot(resid_test_ratio,
                 kde=False,
                 ax=hist_ratio
                )
    hist_ratio.set(title='',
                   xlabel='$\log_{10}$ of the ratio of residuals',
                   ylabel='count',
                   yscale='log'
                  )
    hist_ratio.legend(legend, loc='best')
    
    plt.tight_layout()
    if name is not None and type(name) is str:
        plt.savefig(pimg(f'lumps_{name}_ratio_histogram_compare{suffix}.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg(f'lumps_{name}_ratio_histogram_compare{suffix}.png'), dpi=150, format='png')
    else:
        plt.savefig(pimg('lumps_ratio_histogram_compare' + suffix + '.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg('lumps_ratio_histogram_compare' + suffix + '.png'), dpi=150, format='png')
    
    # close the figure
    plt.close(fig)
    
    # draw residual scatter plot
    fig, scat_res = plt.subplots(1, 1, figsize=figsize)
    
    sns.scatterplot(x=y_train_pred,
                    y=resid_train_pred,
                    alpha=0.35,
                    ax=scat_res
                   )
    sns.scatterplot(x=y_test_pred,
                    y=resid_test_pred,
                    alpha=0.35,
                    ax=scat_res
                   )
    scat_res.set(title='',
                 xlabel='prediction',
                 ylabel='residual'
                )
    scat_res.legend(legend, loc='best')
    
    plt.tight_layout()
    if name is not None and type(name) is str:
        plt.savefig(pimg(f'lumps_{name}_residual_scatterplot_compare{suffix}.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg(f'lumps_{name}_residual_scatterplot_compare{suffix}.png'), dpi=150, format='png')
    else:
        plt.savefig(pimg('lumps_residual_scatterplot_compare' + suffix + '.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg('lumps_residual_scatterplot_compare' + suffix + '.png'), dpi=150, format='png')
    
    # close the figure
    plt.close(fig)
    
    # draw ratio scatter plot
    fig, scat_ratio = plt.subplots(1, 1, figsize=figsize)
    
    sns.scatterplot(x=y_train_pred,
                    y=resid_train_ratio,
                    alpha=0.35,
                    ax=scat_ratio
                   )
    sns.scatterplot(x=y_test_pred,
                    y=resid_test_ratio,
                    alpha=0.35,
                    ax=scat_ratio
                   )
    scat_ratio.set(title='',
                   xlabel='prediction',
                   ylabel='$\log_{10}$ of the ratio of residuals'
                  )
    scat_ratio.legend(legend, loc='best')
    
    plt.tight_layout()
    if name is not None and type(name) is str:
        plt.savefig(pimg(f'lumps_{name}_ratio_scatterplot_compare{suffix}.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg(f'lumps_{name}_ratio_scatterplot_compare{suffix}.png'), dpi=150, format='png')
    else:
        plt.savefig(pimg('lumps_ratio_scatterplot_compare' + suffix + '.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg('lumps_ratio_scatterplot_compare' + suffix + '.png'), dpi=150, format='png')
    
    # close the figure
    plt.close(fig)

def compare_plots_val(estimator, X_train, y_train, y_train_finite, X_val, y_val, y_val_finite, X_test, y_test, y_test_finite, legend=['traning', 'validation', 'test'], name=None, suffix='', figsize=(6,5)):
    '''
    Draw the histogram and residual plots.
    
    Arguments:
        estimator:      the fitted estimator,
        X_train:        the training features,
        y_train:        the training labels,
        y_train_finite: the training finite level truncations,
        X_val:          the validation features,
        y_val:          the validation labels,
        y_val_finite:   the validation finite level truncations,
        X_test:         the test features,
        y_test:         the test labels
        y_test_finite:  the test finite level truncations,.
        
    Optional:
        legend:  list of legend labels (e.g. ['training', 'validation', 'test'])
        name:    the name of the plots (e.g. 'lr', 'svr', etc.),
        suffix:  the suffix of the file name,
        figsize: the size of the subplots.
    '''
    if type(X_train) is pd.DataFrame:
        X_train = X_train.values
    if type(y_train) is pd.Series:
        y_train = y_train.values
    if type(y_train) is list:
        y_train = np.array(y_train)
    if type(y_train_finite) is pd.Series:
        y_train_finite = y_train_finite.values
    if type(y_train_finite) is list:
        y_train_finite = np.array(y_train_finite)
    if type(X_val) is pd.DataFrame:
        X_val = X_val.values
    if type(y_val) is pd.Series:
        y_val = y_val.values
    if type(y_val) is list:
        y_val = np.array(y_val)
    if type(y_val_finite) is pd.Series:
        y_val_finite = y_val_finite.values
    if type(y_val_finite) is list:
        y_val_finite = np.array(y_val_finite)
    if type(X_test) is pd.DataFrame:
        X_test = X_test.values
    if type(y_test) is pd.Series:
        y_test = y_test.values
    if type(y_test) is list:
        y_test = np.array(y_test)
    if type(y_test_finite) is pd.Series:
        y_test_finite = y_test_finite.values
    if type(y_test_finite) is list:
        y_test_finite = np.array(y_test_finite)
        
    # compute predictions and residuals
    y_train_true = y_train.reshape(-1,)
    y_train_pred = estimator.predict(X_train).reshape(-1,)
    y_val_true = y_val.reshape(-1,)
    y_val_pred = estimator.predict(X_val).reshape(-1,)
    y_test_true = y_test.reshape(-1,)
    y_test_pred = estimator.predict(X_test).reshape(-1,)
    
    resid_train_pred  = np.subtract(y_train_true, y_train_pred)
    resid_train_fin   = np.subtract(y_train_true, y_train_finite) + 1.0e-6
    resid_train_ratio = np.log10(np.abs(np.divide(resid_train_pred, resid_train_fin)))    
    resid_val_pred  = np.subtract(y_val_true, y_val_pred)
    resid_val_fin   = np.subtract(y_val_true, y_val_finite) + 1.0e-6
    resid_val_ratio = np.log10(np.abs(np.divide(resid_val_pred, resid_val_fin)))
    resid_test_pred  = np.subtract(y_test_true, y_test_pred)
    resid_test_fin   = np.subtract(y_test_true, y_test_finite) + 1.0e-6
    resid_test_ratio = np.log10(np.abs(np.divide(resid_test_pred, resid_test_fin)))
    
    # draw univariate distribution
    fig, hist_res = plt.subplots(1, 1, figsize=figsize)
    
    sns.distplot(resid_train_pred,
                 kde=False,
                 ax=hist_res
                )
    sns.distplot(resid_val_pred,
                 kde=False,
                 ax=hist_res
                )
    sns.distplot(resid_test_pred,
                 kde=False,
                 ax=hist_res
                )
    hist_res.set(title='',
                 xlabel='residual',
                 ylabel='count',
                 yscale='log'
                )
    hist_res.legend(legend, loc='best')
    
    plt.tight_layout()
    if name is not None and type(name) is str:
        plt.savefig(pimg(f'lumps_{name}_residual_histogram_compare{suffix}.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg(f'lumps_{name}_residual_histogram_compare{suffix}.png'), dpi=150, format='png')
    else:
        plt.savefig(pimg('lumps_residual_histogram_compare' + suffix + '.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg('lumps_residual_histogram_compare' + suffix + '.png'), dpi=150, format='png')
    
    # close the figure
    plt.close(fig)
    
    # draw ratio univariate distribution
    fig, hist_ratio = plt.subplots(1, 1, figsize=figsize)
    
    sns.distplot(resid_train_ratio,
                 kde=False,
                 ax=hist_ratio
                )
    sns.distplot(resid_val_ratio,
                 kde=False,
                 ax=hist_ratio
                )
    sns.distplot(resid_test_ratio,
                 kde=False,
                 ax=hist_ratio
                )
    hist_ratio.set(title='',
                   xlabel='$\log_{10}$ of the ratio of residuals',
                   ylabel='count',
                   yscale='log'
                  )
    hist_ratio.legend(legend, loc='best')
    
    plt.tight_layout()
    if name is not None and type(name) is str:
        plt.savefig(pimg(f'lumps_{name}_ratio_histogram_compare{suffix}.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg(f'lumps_{name}_ratio_histogram_compare{suffix}.png'), dpi=150, format='png')
    else:
        plt.savefig(pimg('lumps_ratio_histogram_compare' + suffix + '.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg('lumps_ratio_histogram_compare' + suffix + '.png'), dpi=150, format='png')
    
    # close the figure
    plt.close(fig)
    
    # draw residual scatter plot
    fig, scat_res = plt.subplots(1, 1, figsize=figsize)
    
    sns.scatterplot(x=y_train_pred,
                    y=resid_train_pred,
                    alpha=0.35,
                    ax=scat_res
                   )
    sns.scatterplot(x=y_val_pred,
                    y=resid_val_pred,
                    alpha=0.35,
                    ax=scat_res
                   )
    sns.scatterplot(x=y_test_pred,
                    y=resid_test_pred,
                    alpha=0.35,
                    ax=scat_res
                   )
    scat_res.set(title='',
                 xlabel='prediction',
                 ylabel='residual'
                )
    scat_res.legend(legend, loc='best')
    
    plt.tight_layout()
    if name is not None and type(name) is str:
        plt.savefig(pimg(f'lumps_{name}_residual_scatterplot_compare{suffix}.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg(f'lumps_{name}_residual_scatterplot_compare{suffix}.png'), dpi=150, format='png')
    else:
        plt.savefig(pimg('lumps_residual_scatterplot_compare' + suffix + '.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg('lumps_residual_scatterplot_compare' + suffix + '.png'), dpi=150, format='png')
    
    # close the figure
    plt.close(fig)
    
    # draw ratio scatter plot
    fig, scat_ratio = plt.subplots(1, 1, figsize=figsize)
    
    sns.scatterplot(x=y_train_pred,
                    y=resid_train_ratio,
                    alpha=0.35,
                    ax=scat_ratio
                   )
    sns.scatterplot(x=y_val_pred,
                    y=resid_val_ratio,
                    alpha=0.35,
                    ax=scat_ratio
                   )
    sns.scatterplot(x=y_test_pred,
                    y=resid_test_ratio,
                    alpha=0.35,
                    ax=scat_ratio
                   )
    scat_ratio.set(title='',
                   xlabel='prediction',
                   ylabel='$\log_{10}$ of the ratio of residuals'
                  )
    scat_ratio.legend(legend, loc='best')
    
    plt.tight_layout()
    if name is not None and type(name) is str:
        plt.savefig(pimg(f'lumps_{name}_ratio_scatterplot_compare{suffix}.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg(f'lumps_{name}_ratio_scatterplot_compare{suffix}.png'), dpi=150, format='png')
    else:
        plt.savefig(pimg('lumps_ratio_scatterplot_compare' + suffix + '.pdf'), dpi=150, format='pdf')
        plt.savefig(pimg('lumps_ratio_scatterplot_compare' + suffix + '.png'), dpi=150, format='png')
    
    # close the figure
    plt.close(fig)

def history_plots(history, suffix='', figsize=(6,5)):
    '''learning
    Plot the history of the training.
    
    Arguments:
        history: the history dataframe.
        
    Optional:
        suffix:  the suffix of the file name,
        figsize: the size of the subplots.
    '''
    loss = history[['loss', 'val_loss']]
    lr   = history['lr']
    
    # draw loss function
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    sns.lineplot(data=loss,
                 ci=None,
                 ax=ax
                )
    ax.set(title='',
           xlabel='epoch',
           ylabel='loss',
           yscale='log'
          )
    ax.legend(['training', 'validation'], loc='best')
    
    plt.tight_layout()
    plt.savefig(pimg('lumps_ann_loss' + suffix + '.pdf'), dpi=150, format='pdf')
    plt.savefig(pimg('lumps_ann_loss' + suffix + '.png'), dpi=150, format='png')
    
    # close the figure
    plt.close(fig)
    
    # draw learning rate
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    sns.lineplot(data=lr,
                 ci=None,
                 ax=ax
                )
    ax.set(title='',
           xlabel='epoch',
           ylabel='learning rate',
           yscale='log'
          )
    
    plt.tight_layout()
    plt.savefig(pimg('lumps_ann_lr' + suffix + '.pdf'), dpi=150, format='pdf')
    plt.savefig(pimg('lumps_ann_lr' + suffix + '.png'), dpi=150, format='png')
    
    # close the figure
    plt.close(fig)