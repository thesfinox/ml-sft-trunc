# Machine learning for level truncation in String Field Theory

## Description

In the framework of bosonic **Open String Field Theory** we consider several observables at different finite mass level truncations. We then try to extract the prediction for the truncation at infinity.

## Methodology

The analysis is written using Python and Jupyter notebooks. The main file is [sft-level-trunc.ipynb](./sft-level-trunc.ipynb): it is self-consistent and can be run from top to bottom without input by the user.

The first part of the analysis is dedicated to the study and the **exploratory data analysis** of the dataset. We first divide each observation in a separate entry and tidy the dataset for the analysis. We then study properties and correlations between the variables as well as a first exploratory linear regression to figure out the impact of each variable on the final result. We also include featurisation of the dataset by studying the principal values and the KMeans clustering of the truncation levels.

In the second part we focus on the **regression analysis** of the prepared data. We start by considering linear models, move to support vector machines and decision trees, and finally consider an approach using a shallow neural network. The optimisation of the algorithms is performed using Bayes statistics in order to minimise the mean squared error of the predictions.

After the analysis is concluded, data is saved into CSV files for comparison with the original data. The Python script [parser.py](./parser.py) can be used to parse the CSV output to the previous format used by Mathematica notebooks (it automatically takes care of discrepancy in scientific notation and curly braces).

## Notes

This branch contains a brief report of the analysis summarising the main results and methodologies.
