import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def run_avg(values: pd.Series, window: int) -> np.ndarray:
    '''
    Compute the (simple) running average of a series of data.
    
    Required arguments:
        values: list of values,
        window: temporal window for the computation.
    '''
    values = values.values.reshape(-1,)
    avg    = np.zeros((np.shape(values)[0] - window + 1,))
    
    for i in range(avg.shape[0]):
        avg[i] = np.sum(values[i:i + window]) / window
    
    return avg

# configuration
os.makedirs('./img', exist_ok=True)
sns.set()
        
# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, help='history data (CSV)')
parser.add_argument('-s', '--smooth', type=float, default=0.1, help='smoothening factor')

args = parser.parse_args()

# read data
data = pd.read_csv(args.data)

# smooth the data
smooth = lambda s: run_avg(s, int(args.smooth * data.shape[0]))
data_smooth = data.apply(smooth)

# plot the loss function
_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.lineplot(data=data[['loss', 'val_loss']],
             dashes=['', (4,2)],
             ax=ax
            )
ax.set(title='',
       xlabel='epochs',
       ylabel='loss',
       xscale='log',
       yscale='log'
      )
ax.legend(['training', 'validation'])

plt.tight_layout()
plt.savefig('./img/loss.pdf', dpi=150, format='pdf')
plt.savefig('./img/loss.png', dpi=150, format='png')

# plot the mse metric
_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.lineplot(data=data[['exp_re_mse',
                        'exp_im_mse',
                        'val_exp_re_mse',
                        'val_exp_im_mse'
                       ]
                      ],
             dashes=['', '', (4,2), (4,2)],
             ax=ax
            )
ax.set(title='',
       xlabel='epochs',
       ylabel='MSE',
       xscale='log',
       yscale='log'
      )
ax.legend(['training - $\mathrm{Re}(\mathrm{exp})$',
           'training - $\mathrm{Im}(\mathrm{exp})$',
           'validation - $\mathrm{Re}(\mathrm{exp})$',
           'validation - $\mathrm{Im}(\mathrm{exp})$'
          ]
         )

plt.tight_layout()
plt.savefig('./img/mse.pdf', dpi=150, format='pdf')
plt.savefig('./img/mse.png', dpi=150, format='png')

# plot the mae metric
_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.lineplot(data=data[['exp_re_mae',
                        'exp_im_mae',
                        'val_exp_re_mae',
                        'val_exp_im_mae'
                       ]
                      ],
             dashes=['', '', (4,2), (4,2)],
             ax=ax
            )
ax.set(title='',
       xlabel='epochs',
       ylabel='MAE',
       xscale='log',
       yscale='log'
      )
ax.legend(['training - $\mathrm{Re}(\mathrm{exp})$',
           'training - $\mathrm{Im}(\mathrm{exp})$',
           'validation - $\mathrm{Re}(\mathrm{exp})$',
           'validation - $\mathrm{Im}(\mathrm{exp})$'
          ]
         )

plt.tight_layout()
plt.savefig('./img/mae.pdf', dpi=150, format='pdf')
plt.savefig('./img/mae.png', dpi=150, format='png')

# plot the smooth loss function
_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.lineplot(data=data_smooth[['loss', 'val_loss']],
             dashes=['', (4,2)],
             ax=ax
            )
ax.set(title='',
       xlabel='epochs ({:.2f} x no. of samples window)'.format(args.smooth),
       ylabel='loss',
       xscale='log',
       yscale='log'
      )
ax.legend(['training', 'validation'])

plt.tight_layout()
plt.savefig('./img/smooth_loss.pdf', dpi=150, format='pdf')
plt.savefig('./img/smooth_loss.png', dpi=150, format='png')

# plot the smooth mse metric
_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.lineplot(data=data_smooth[['exp_re_mse',
                               'exp_im_mse',
                               'val_exp_re_mse',
                               'val_exp_im_mse'
                              ]
                             ],
             dashes=['', '', (4,2), (4,2)],
             ax=ax
            )
ax.set(title='',
       xlabel='epochs ({:.2f} x no. of samples window)'.format(args.smooth),
       ylabel='MSE',
       xscale='log',
       yscale='log'
      )
ax.legend(['training - $\mathrm{Re}(\mathrm{exp})$',
           'training - $\mathrm{Im}(\mathrm{exp})$',
           'validation - $\mathrm{Re}(\mathrm{exp})$',
           'validation - $\mathrm{Im}(\mathrm{exp})$'
          ]
         )

plt.tight_layout()
plt.savefig('./img/smooth_mse.pdf', dpi=150, format='pdf')
plt.savefig('./img/smooth_mse.png', dpi=150, format='png')

# plot the smooth mae metric
_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.lineplot(data=data_smooth[['exp_re_mae',
                               'exp_im_mae',
                               'val_exp_re_mae',
                               'val_exp_im_mae'
                              ]
                             ],
             dashes=['', '', (4,2), (4,2)],
             ax=ax
            )
ax.set(title='',
       xlabel='epochs ({:.2f} x no. of samples window)'.format(args.smooth),
       ylabel='MAE',
       xscale='log',
       yscale='log'
      )
ax.legend(['training - $\mathrm{Re}(\mathrm{exp})$',
           'training - $\mathrm{Im}(\mathrm{exp})$',
           'validation - $\mathrm{Re}(\mathrm{exp})$',
           'validation - $\mathrm{Im}(\mathrm{exp})$'
          ]
         )

plt.tight_layout()
plt.savefig('./img/smooth_mae.pdf', dpi=150, format='pdf')
plt.savefig('./img/smooth_mae.png', dpi=150, format='png')