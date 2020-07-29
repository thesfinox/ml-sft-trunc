# Machine Learning for Level Truncation in String Field Theory

The analysis is split across several notebook to be more readable.
Ideally the order to run the notebooks is the following:

1. [tidy](./sft-trunc_tidy.ipynb): tidy the dataset and prepare for the analysis,
2. [eda](./sft-trunc_eda.ipynb): perform the exploratory data analysis to probe the dataset,
3. [unsupervised](./sft-trunc_unsup.ipynb): perform basic unsupervised tasks such as PCA and clustering,
4. [inference](./sft-trunc_infer.ipynb): study statistical inference using linear models,
5. [machine learning](./sft-trunc_ml.ipynb): perform the machine learning analysis,
6. [shap](./sft-trunc_shap.ipynb): study the variable ranking and Shapley values of the decision trees,
7. [double lumps](./sft-trunc_dlump.ipynb): perform basic predictions on the double lumps using previously studied properties.
