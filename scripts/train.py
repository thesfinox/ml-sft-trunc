import tensorflow as tf
import pandas as pd
import time
import argparse
import re
import logging
import os

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

parser.add_argument('-tx', '--train-x', type=str, help='training features (CSV)')
parser.add_argument('-ty', '--train-y', type=str, help='training labels (CSV)')
parser.add_argument('-vx', '--val-x', type=str, help='validation features (CSV)')
parser.add_argument('-vy', '--val-y', type=str, help='validation labels (CSV)')
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

X_train = pd.read_csv(args.train_x)
X_val   = pd.read_csv(args.val_x)

y_train = pd.read_csv(args.train_y)
y_val   = pd.read_csv(args.val_y)

# transform labels
y_train = {s: y_train[s].values.reshape(-1,) for s in ['exp_re', 'exp_im']}
y_val   = {s: y_val[s].values.reshape(-1,) for s in ['exp_re', 'exp_im']}

# define the callbacks
os.makedirs('./logs', exist_ok=True)
os.makedirs('./models', exist_ok=True)
callbacks = [tf.keras.callbacks.ModelCheckpoint(args.model, save_best_only=True),
             tf.keras.callbacks.EarlyStopping(patience=args.early_stop),
             tf.keras.callbacks.ReduceLROnPlateau(factor=args.plateau_factor,
                                                  patience=args.plateau_patience,
                                                  min_lr=1.0e-6
                                                 ),
             tf.keras.callbacks.TensorBoard(log_dir=os.path.join('./logs', now),
                                            profile_batch=0,
                                            write_graph=True
                                           )
            ]

elapsed = time.time()
history = model.fit(x=X_train,
                    y=y_train,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    verbose=0,
                    callbacks=callbacks,
                    validation_data=(X_val, y_val)
                   )
elapsed = time.time() - elapsed
print('Training finished in {:.2f} seconds.'.format(elapsed))

# save output files
hst = pd.DataFrame(history.history)

rename = hst.columns
rename = [re.sub('(.*)mean_squared_error', r'\1mse', c) for c in rename]
rename = [re.sub('(.*)mean_absolute_error', r'\1mae', c) for c in rename]

hst = hst.rename(columns=dict(zip(hst.columns, rename)))
hst.to_csv('./data/history.csv', index=False)