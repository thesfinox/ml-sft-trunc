# Machine Learning for Level Truncation in Open Bosonic String Field Theory

## Description

In the framework of bosonic Open String Field Theory we consider data for different types of observables of different weights. We then consider their computation for various values of mass level truncations and predict the extrapolation for the level-infinity using machine learning techniques.

## Analysis

Given the variability of the data we approach the problem in different ways:

- we consider the full database and perform the analysis using decision tree based algorithms, which have proven to be good estimators (we provide a simpler linear algorithm for comparison),
- we consider values of `weight` < 1.5 (such that all entries of the dataset are of order 1) and perform the analysis,
- we take values of `weight` > 1.5 (entries have a wide range of values), transform the data and then perform the analysis,
- if needed, we vary the objective function to minimize 1 - exp(-(y_true - y_pred)^2) when `weight` > 1.5.
