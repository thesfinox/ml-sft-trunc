import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# configuration
os.makedirs('./img', exist_ok=True)
sns.set()
        
# set parser
parser = argparse.ArgumentParser()

parser.add_argument('-gt', '--ground-truth', type=str, help='ground truth labels (CSV)')
parser.add_argument('-pl', '--predicted-labels', type=str, help='predicted labels (CSV)')
parser.add_argument('-o', '--output', type=str, help='basename of the output')

args = parser.parse_args()

# read data
y_pred = pd.read_csv(args.predicted_labels)
y_true = pd.read_csv(args.ground_truth)

# compute residuals
res = y_true - y_pred

# plot the histogram of the residuals
_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.distplot(res['exp_re'],
             kde=False,
             label='$\mathrm{Re}(\mathrm{exp})$',
             ax=ax
            )
sns.distplot(res['exp_im'],
             kde=False,
             label='$\mathrm{Im}(\mathrm{exp})$',
             ax=ax
            )
ax.set(title='',
       xlabel='residual',
       ylabel='count'
      )
ax.legend(loc='best')

plt.tight_layout()
plt.savefig('./img/{}_res_dist.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}_res_dist.png'.format(args.output), dpi=150, format='png')

# plot predictions vs true values (Re(exp))
_, ax = plt.subplots(1, 1, figsize=(6,5))

data = pd.DataFrame({'true values': y_true['exp_re'],
                     'predictions': y_pred['exp_re']
                    }
                   )

sns.lineplot(data=data,
             alpha=0.5,
             markers=True,
             ax=ax
            )
ax.set(title='',
       xlabel='ID of the prediction',
       ylabel='$\mathrm{Re}(\mathrm{exp})$'
      )
ax.legend(loc='best')

plt.tight_layout()
plt.savefig('./img/{}_exp_re_plot.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}_exp_re_plot.png'.format(args.output), dpi=150, format='png')

# plot predictions vs true values (Im(exp))
_, ax = plt.subplots(1, 1, figsize=(6,5))

data = pd.DataFrame({'true values': y_true['exp_im'],
                     'predictions': y_pred['exp_im']
                    }
                   )

sns.lineplot(data=data,
             alpha=0.5,
             markers=True,
             ax=ax
            )
ax.set(title='',
       xlabel='ID of the prediction',
       ylabel='$\mathrm{Im}(\mathrm{exp})$'
      )
ax.legend(loc='best')

plt.tight_layout()
plt.savefig('./img/{}_exp_im_plot.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}_exp_im_plot.png'.format(args.output), dpi=150, format='png')

# residual plot
_, ax = plt.subplots(1, 1, figsize=(6,5))

sns.scatterplot(x=y_pred['exp_re'],
                y=res['exp_re'],
                alpha=0.5,
                ax=ax
               )
sns.scatterplot(x=y_pred['exp_im'],
                y=res['exp_im'],
                alpha=0.5,
                ax=ax
               )
ax.set(title='',
       xlabel='predictions',
       ylabel='residual'
      )
ax.legend(['$\mathrm{Re}(\mathrm{exp})$', '$\mathrm{Im}(\mathrm{exp})$'])

plt.tight_layout()
plt.savefig('./img/{}_res_plot.pdf'.format(args.output), dpi=150, format='pdf')
plt.savefig('./img/{}_res_plot.png'.format(args.output), dpi=150, format='png')