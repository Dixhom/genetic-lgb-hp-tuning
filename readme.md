# LightGBM Hyperparameter Genetic Tuner

This is a code to tune the hyperparameter of LightGBM by a genetic algorithm.

First, "chromosomes" are generated. These are actually class instances containing hyperparameters to tune. At each generation the chromosomes are mutated or crossed over at certain probabilities.

Mutation changes a hyperparameter slightly according to a certain probability distribution, including a gamma distribution and a normal one.

Crossover swaps the hyperparameters of two instances.

# License

MIT

# Acknowledgement

A genetic algorithm code was taken from here:  
https://qiita.com/simonritchie/items/d7f1596e7d034b9422ce
