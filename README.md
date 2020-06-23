# Machine learning for level truncation in String Field Theory.

## Description

In the framework of bosonic **Open String Field Theory** we consider several observables and the positions of the vacua of the potential at different finite mass level truncations. We then try to extract the prediction for the truncation at infinity.

## Methodology

The first part of the analysis is dedicated to the study and the **exploratory data analysis** of the dataset. We first divide each observation in a separate entry and tidy the dataset for the analysis. We then study properties and correlations between the variables as well as a first exploratory linear regression to figure out the impact of each variable on the final result. We also include featurisation of the dataset by studying the principal values and the KMeans clustering of the truncation levels.

In the second part we focus on the **regression analysis** of the prepared data. We start by considering linear models, move to support vector machines and decision trees, and finally consider an approach using a shallow neural network. The optimisation of the algorithms is performed using Bayes statistics in order to minimise the mean squared error of the predictions.

## Libraries

We mainly use [Pandas](https://pandas.pydata.org/) and [Numpy](https://numpy.org/) for manipulation and [Scipy](https://www.scipy.org/) for statistical analysis on the dataset. Pictures of the regression results and the plots were created using [Seaborn](https://seaborn.pydata.org/). We used [Scikit-learn](https://scikit-learn.org/stable/) for inear models and support vectors, [LightGBM](https://www.microsoft.com/en-us/research/project/lightgbm/) for trees (both random forests and gradient boosted trees) and [Tensorflow](https://www.tensorflow.org/) for the neural network. Hyperparameter optimisation was performed using [Scikit-optimize](https://scikit-optimize.github.io/stable/).
