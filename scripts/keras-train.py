import pandas as pd

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
        
# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-t', '--train', type=str, help='training dataset (CSV)')
parser.add_argument('-lt', '--trainlabels', type=str, help='training labels (CSV)')
parser.add_argument('-v', '--val', type=str, help='validation dataset (CSV)')
parser.add_argument('-lv', '--vallabels', type=str, help='validation labels (CSV)')
parser.add_argument('-m', '--model', type=str, help='model file')
parser.add_argument('-e', '--epochs', type=int, default=10, help='epochs of training')
parser.add_argument('-b', '--batch', type=int, default=32, help='batch size')
parser.add_argument('-s', '--stop', type=int, default=10, help='early stopping patience')
parser.add_argument('-p', '--plateau', type=int, default=10, help='readuce LR patience')
parser.add_argument('-r', '--rand', type=int, default=42, help='random seed')

args = parser.parse_args()

# set random seed
tf.random.set_seed(args.rand)

# load the datasets
X_train = pd.read_csv(args.train)
y_train = pd.read_csv(args.trainlabels)
X_val = pd.read_csv(args.val)
y_val = pd.read_csv(args.vallabels)

y_train = {'re': y_train['exp_re'].values,
           'im': y_train['exp_im'].values
          }
y_val = {'re': y_val['exp_re'].values,
         'im': y_val['exp_im'].values
        }

# load the estimator
ann_mod = tf.keras.models.load_model(args.model)

# define callbacks
callbacks = [tf.keras.callbacks.ModelCheckpoint(args.model,
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

# fit the model
t = time.time()

ann_mod_hst = ann_mod.fit(X_train,
                          y_train,
                          batch_size=args.batch,
                          epochs=args.epochs,
                          verbose=0,
                          callbacks=callbacks,
                          validation_data=(X_val, y_val)
                         )

t = time.time() - t
print('Model trained in {:.3f} seconds.'.format(t))

# save the history file
if 'lr' in ann_mod_hst.history:
    del ann_mod_hst.history['lr']
    
with open('./models/ann_mod_hst.json', 'w') as f:
    json.dump(ann_mod_hst.history, f)