# Machine learning for level truncation in String Field Theory

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/thesfinox/ml-sft-trunc/master) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/thesfinox/ml-sft-trunc/blob/master/sft-level-trunc.ipynb)

## Description

In the framework of bosonic **Open String Field Theory** we consider several observables at different finite mass level truncations. We then try to extract the prediction for the truncation at infinity.

## Methodology

The analysis is written using Python and Jupyter notebooks. The main file is [sft-level-trunc.ipynb](./sft-level-trunc.ipynb): it is self-consistent and can be run from top to bottom without input by the user.

The first part of the analysis is dedicated to the study and the **exploratory data analysis** of the dataset. We first divide each observation in a separate entry and tidy the dataset for the analysis. We then study properties and correlations between the variables as well as a first exploratory linear regression to figure out the impact of each variable on the final result. We also include featurisation of the dataset by studying the principal values and the KMeans clustering of the truncation levels.

In the second part we focus on the **regression analysis** of the prepared data. We start by considering linear models, move to support vector machines and decision trees, and finally consider an approach using a shallow neural network. The optimisation of the algorithms is performed using Bayes statistics in order to minimise the mean squared error of the predictions.

After the analysis is concluded, data is saved into CSV files for comparison with the original data. The Python script [parser.py](./parser.py) can be used to parse the CSV output to the previous format used by Mathematica notebooks (it automatically takes care of discrepancy in scientific notation and curly braces).

## Libraries

We mainly use [Pandas](https://pandas.pydata.org/) and [Numpy](https://numpy.org/) for manipulation and [Scipy](https://www.scipy.org/) for statistical analysis on the dataset. Pictures of the regression results and the plots were created using [Seaborn](https://seaborn.pydata.org/). We used [Scikit-learn](https://scikit-learn.org/stable/) for linear models and support vectors, [LightGBM](https://www.microsoft.com/en-us/research/project/lightgbm/) for trees (both random forests and gradient boosted trees) and [Tensorflow](https://www.tensorflow.org/) for the neural network. Hyperparameter optimisation was performed using [Scikit-optimize](https://scikit-optimize.github.io/stable/).

During development the analysis was built in a Conda environment with modules installed from the base and conda-forge repositories. In general you should be able to install all dependencies using:

```shell
conda install numpy pytables scipy pandas scikit-learn scikit-optimize tensorflow-gpu matplotlib seaborn pydot shap
```

if you are using Conda, or

```shell
pip install --user numpy tables scipy pandas scikit-learn scikit-optimize tensorflow-gpu matplotlib seaborn pydot shap
```

if you are using `pip`. Python should be at least at version 3.6 for the dependencies to work.

In the repository you can also find a [summary](./environment.yml) of the dependencies needed to create the virtual environment (e.g. using `conda env create -f environment.yml`).
