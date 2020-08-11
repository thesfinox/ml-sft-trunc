import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
import json
import time
import argparse

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# common settings
sns.set()
os.makedirs('./img', exist_ok=True)

# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, help='dataset (CSV)')
parser.add_argument('-e', '--estimator', type=str, help='model file')
parser.add_argument('-p', '--params', type=str, help='hyperparameters dictionary (JSON)')
parser.add_argument('-r', '--rand', type=int, default=42, help='random seed')
parser.add_argument('-o', '--output', type=str, help='output basename')

args = parser.parse_args()

# load the datasets
df = pd.read_csv(args.data)

# load the estimator
estimator = joblib.load(args.estimator)

# load the parameters
if args.params != 'None' and args.params is not None:
    with open(args.params, 'r') as p:
        params = json.load(p)
        estimator.set_params(**params)
        
# fit the estimator
size      = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
mse_train = {'exp_re': [],
             'exp_im': []
            }
mse_test  = {'exp_re': [],
             'exp_im': []
            }

t = time.time()

for s in size:
    df_train, df_test = train_test_split(df, train_size=s, shuffle=True, random_state=args.rand)
    
    # divide features and labels
    lab_train = df_train[['exp_re', 'exp_im']].values
    lab_test  = df_test[['exp_re', 'exp_im']].values
    
    train = df_train.drop(columns=['exp_re', 'exp_im')
    test = df_test.drop(columns=['exp_re', 'exp_im'])
    
    # train and predict
    tt = time.time()
    estimator.fit(train, lab_train)
    tt = time.time() - tt
    
    print('Trained ratio {:.2f} in {:.3f} seconds.'.format(s, tt))
    
    lab_train_pred = estimator.predict(train)
    lab_test_pred  = estimator.predict(test)
    
    # compute the mean squared error
    mse_train['exp_re'].append(mean_squared_error(lab_train[:,0], lab_train_pred[:,0]))
    mse_test['exp_re'].append(mean_squared_error(lab_test[:,0], lab_test_pred[:,0]))
                                   
    mse_train['exp_im'].append(mean_squared_error(lab_train[:,1], lab_train_pred[:,1]))
    mse_test['exp_im'].append(mean_squared_error(lab_test[:,1], lab_test_pred[:,1]))

t = time.time() - t
print('Learning curve for {} trained in {:.3f} seconds.'.format(estimator.__class__.__name__, t))

# produce the plot
mse_data = pd.DataFrame({'training -- Re': mse_train['exp_re'],
                         'validation -- Re': mse_test['exp_re'],
                         'training -- Im': mse_train['exp_im'],
                         'validation -- Im': mse_test['exp_im'],
                        },
                        index=size
                       )

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