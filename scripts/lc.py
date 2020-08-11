import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import argparse
import re
import logging
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# create new directory
from datetime import datetime
now = datetime.now()
now = now.strftime("%Y%m%d_%H%M%S")

# set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

# set less logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, help='dataset (CSV)')
parser.add_argument('-m', '--model', type=str, help='model file (HDF5)')
parser.add_argument('-ep', '--epochs', type=int, default=10, help='epochs of training')
parser.add_argument('-b', '--batch-size', type=int, default=32, help='batch size')
parser.add_argument('-es', '--early-stop', type=int, default=10, help='early stopping patience')
parser.add_argument('-pf', '--plateau-factor', type=float, default=0.1, help='readuce LR factor')
parser.add_argument('-pp', '--plateau-patience', type=int, default=10, help='readuce LR patience')
parser.add_argument('-r', '--rand', type=int, default=42, help='random seed')

args = parser.parse_args()

# set random seed
tf.random.set_seed(args.rand)

# load files
model = tf.keras.models.load_model(args.model)

df = pd.read_csv(args.data)

# keep 10% of the set for validation
df_train, df_val = train_test_split(df, test_size=0.1, shuffle=True, random_state=args.rand)

X_val = df_val.drop(columns=['exp_re', 'exp_im'])
y_val = df_val[['exp_re', 'exp_im']]
y_val = {s: y_val[s].values.reshape(-1,) for s in ['exp_re', 'exp_im']}

# define the training ratios
ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
metric = {'re_training': [],
          'im_training': [],
          're_test':     [],
          'im_test':     []
         }

for r in ratios:
    # get training and test sets
    train, test = train_test_split(df_train, train_size=r, shuffle=True, random_state=args.rand)
    
    X_train = train.drop(columns=['exp_re', 'exp_im'])
    X_test  = test.drop(columns=['exp_re', 'exp_im'])
    
    y_train = train[['exp_re', 'exp_im']]
    y_test  = test[['exp_re', 'exp_im']]
    
    y_train = {s: y_train[s].values.reshape(-1,) for s in ['exp_re', 'exp_im']}
    y_test  = {s: y_test[s].values.reshape(-1,) for s in ['exp_re', 'exp_im']}

    # define the callbacks
    os.makedirs('./models', exist_ok=True)
    callbacks = [tf.keras.callbacks.ModelCheckpoint('./models/ann_lc_tmp.h5', save_best_only=True),
                 tf.keras.callbacks.EarlyStopping(patience=args.early_stop),
                 tf.keras.callbacks.ReduceLROnPlateau(factor=args.plateau_factor,
                                                      patience=args.plateau_patience,
                                                      min_lr=1.0e-6
                                                     )
                ]

    elapsed = time.time()
    _ = model.fit(x=X_train,
                  y=y_train,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  verbose=0,
                  callbacks=callbacks,
                  validation_data=(X_val, y_val)
                 )
    elapsed = time.time() - elapsed
    print('Training ratio {:.2f} finished training in {:.2f} seconds.'.format(r, elapsed))
    
    # compute predictions
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)
    
    # compute the metric
    metric['re_training'].append(mean_squared_error(y_train_pred['exp_re'].reshape(-1,),
                                                    y_train['exp_re'].reshape(-1,)
                                                   )
                                )
    metric['im_training'].append(mean_squared_error(y_train_pred['exp_im'].reshape(-1,),
                                                    y_train['exp_im'].reshape(-1,)
                                                   )
                                )
    metric['re_test'].append(mean_squared_error(y_test_pred['exp_re'].reshape(-1,),
                                                y_test['exp_re'].reshape(-1,)
                                               )
                            )
    metric['im_test'].append(mean_squared_error(y_test_pred['exp_im'].reshape(-1,),
                                                y_test['exp_im'].reshape(-1,)
                                               )
                            )
    
# create the dataframe and plot
metric = pd.DataFrame(metric, index=ratios)

os.makedirs('./img', exist_ok=True)
sns.set()
_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.lineplot(data=metric,
             dashes=['', '', (4,2), (4,2)],
             ax=ax
            )
ax.set(title='',
       xlabel='training ratio',
       ylabel='MSE',
       yscale='log'
      )
ax.legend(['training $\mathrm{Re}(\mathrm{exp})$',
           'training $\mathrm{Im}(\mathrm{exp})$',
           'test $\mathrm{Re}(\mathrm{exp})$',
           'test $\mathrm{Im}(\mathrm{exp})$',
          ]
         )

plt.tight_layout()
plt.savefig('./img/ann_lc.pdf', dpi=150, format='pdf')
plt.savefig('./img/ann_lc.png', dpi=150, format='png')