# LightGBM Hyperparameter Genetic Tuner

This is a code to tune the hyperparameter of LightGBM by a genetic algorithm.

First, "chromosomes" are generated. These are actually class instances containing hyperparameters to tune. At each generation the chromosomes are mutated or crossed over at certain probabilities.

Mutation changes a hyperparameter slightly according to a certain probability distribution, including a gamma distribution and a normal one.

Crossover swaps the hyperparameters of two instances.

For a hyperparameter whose range is 0 from 1, we have defined an original probability distribution. Its mode is the original value of the hyperparameter. Thu further it is from the mode, the smaller the density is.

The fitness is the negative logloss of the cross validated out-of-folds. To make sure each cross validation generates the same result, all random seeds are fixed.

# Usage

Change parameters in `if __name__ == '__main__'` in `HyperParamTuner.py` and execute it.

# License

MIT

# Acknowledgement

A genetic algorithm code was taken from here:  
https://qiita.com/simonritchie/items/d7f1596e7d034b9422ce
