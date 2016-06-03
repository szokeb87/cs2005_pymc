# Cogley and Sargent (2005) in `pymc`

This repository is a collection of programs and jupyter notebooks that implement the Bayesian estimation of the time-varying parameter VAR with stochastic volatility as it was discussed by [Cogley and Sargent (2005)](http://www.sciencedirect.com/science/article/pii/S1094202505000049) using the Python package `pymc`.

There are three main notebooks:
 - [CogleySargent2005_replication.ipynb](https://github.com/szokeb87/cs2005_pymc/blob/master/notebooks/CogleySargent2005_replication.ipynb): this notebook replicates the paper using the exact same dataset and prior parameters. By means of a few illustrative figures it compares the `pymc` posterior sample with the sample drawn by Tim Cogley's Matlab files
 - [CogleySargent2005_reestimation.ipynb](https://github.com/szokeb87/cs2005_pymc/blob/master/notebooks/CogleySargent2005_reestimation.ipynb): this notebook reestimates the model using an updated dataset. The updating is accomplished by [this notebook](https://github.com/szokeb87/cs2005_pymc/blob/master/notebooks/Updating_the_sample.ipynb) (using pandas built-in datareader for FRED)
 - [Laboratory.ipynb](https://github.com/szokeb87/cs2005_pymc/blob/master/notebooks/Laboratory.ipynb): this notebook does not use any real data, it is a laboratory to investigate the "accuracy" of the sampler given that we know the true data generating process. It simulates an artificial dataset (for arbitrary fixed parameter values), which is then used to fit the model (just like we do with real data).

The [data](https://github.com/szokeb87/cs2005_pymc/tree/master/data) folder consists of .csv, .mat and .pickle files storing the posterior samples and the datasets that have been used for the estimation  

The generated figures can be found in [this folder](https://github.com/szokeb87/cs2005_pymc/tree/master/figures)

> **Note:** the Matlab files in [matlab_files_from_cogley](https://github.com/szokeb87/cs2005_pymc/tree/master/matlab_files_from_cogley) are written entirely by Tim Cogley and Thomas Sargent
