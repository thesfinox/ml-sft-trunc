import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
import json
import time
import argparse

# set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# common settings
sns.set()
os.makedirs('./img', exist_ok=True)

# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, help='dataset (CSV)')
parser.add_argument('-m', '--model', type=str, help='model file')
parser.add_argument('-r', '--rand', type=int, default=42, help='random seed')
parser.add_argument('-e', '--epochs', type=int, default=10, help='epochs of training')
parser.add_argument('-b', '--batch', type=int, default=32, help='batch size')
parser.add_argument('-s', '--stop', type=int, default=10, help='early stopping patience')
parser.add_argument('-p', '--plateau', type=int, default=10, help='readuce LR patience')
parser.add_argument('-o', '--output', type=str, help='output basename')

args = parser.parse_args()

# set random seed
tf.random.set_seed(args.rand)

# load the datasets
df = pd.read_csv(args.data)

# load the estimator
ann_mod = tf.keras.models.load_model(args.model)

# define callbacks
callbacks = [tf.keras.callbacks.ModelCheckpoint('./models/ann_model_lc.h5',
                                                monitor='val_loss',
                                                verbose=0,
                                                save_best_only=True
                                               ),
             tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=args.stop,
                                              verbose=0
                                             ),
             tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  factor=0.1,
                                                  patience=args.plateau,
                                                  verbose=0,
                                                  min_lr=1.0e-6
                                                 )
            ]
        
# fit the estimator
t = time.time()

size      = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
mse_train = []
mse_test  = []
sol       = df['solutions'].unique()

for s in size:
    sol_train, sol_test = train_test_split(sol, train_size=s, shuffle=True, random_state=args.rand)
    
    # divide training and test set
    train = df.loc[df['solutions'].isin(sol_train)]
    test  = df.loc[df['solutions'].isin(sol_test)]
    
    # divide features and labels
    lab_train = train['exp'].values.reshape(-1,)
    lab_test  = test['exp'].values.reshape(-1,)
    
    train = train.drop(columns=['solutions', 'init', 'exp'])
    test = test.drop(columns=['solutions', 'init', 'exp'])
    
    # train and predict
    tt = time.time()
    _ = ann_mod.fit(train,
                    lab_train,
                    batch_size=args.batch,
                    epochs=args.epochs,
                    verbose=0,
                    callbacks=callbacks,
                    validation_data=(test, lab_test)
                   )
    tt = time.time() - tt
    
    print('Trained ratio {:.2f} in {:.3f} seconds.'.format(s, tt))
    
    # load best model
    ann_mod = tf.keras.models.load_model('./models/ann_model_lc.h5')

    # compute the predictions
    lab_train_pred = ann_mod.predict(train).reshape(-1,)
    lab_test_pred  = ann_mod.predict(test).reshape(-1,)
    
    # compute the mean squared error
    mse_train.append(mean_squared_error(lab_train, lab_train_pred))
    mse_test.append(mean_squared_error(lab_test, lab_test_pred))

t = time.time() - t
print('Learning curve trained in {:.3f} seconds.'.format(t))

# produce the plot
mse_data = pd.DataFrame({'training': mse_train, 'validation': mse_test}, index=size)

_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.lineplot(data=mse_data,
             ax=ax
            )
ax.set(title='',
       xlabel='training ratio',
       ylabel='MSE',
       yscale='log'
      )
        
plt.tight_layout()
plt.savefig('./img/{}.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}.png'.format(args.output), dpi=150, format='png')