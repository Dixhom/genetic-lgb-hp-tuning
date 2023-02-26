from __future__ import annotations

from typing import TypeVar, List, Dict
from random import choices, random, randrange, shuffle, uniform
import random
from heapq import nlargest
from copy import deepcopy
from datetime import datetime
from contextlib import contextmanager
import sys, os
import warnings
from math import erf
from io import StringIO

from scipy.stats import norm
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.datasets import load_breast_cancer

from GeneticAlgorithm import GeneticAlgorithm
from Chromosome import Chromosome

# suppress all warnings
warnings.simplefilter('ignore')

# genetic algorithm code was taken from here
# https://qiita.com/simonritchie/items/d7f1596e7d034b9422ce

class HyperParamTuner(Chromosome):
    def __init__(self, params: dict, X: np.array, y: np.array, model_const, n_cvfolds: int) -> None:
        """
        to tune the hyperparameters of a machine learning model using genetic algorithm 

        Parameters
        ----------
        params : dict
            hyperparameters
        X : np.array
            initial value of X
        y : np.array
            initial value of y
        model: 
            model object
        """

        self.params = params
        self.X = X
        self.y = y
        self.model_const = model_const
        self.n_cvfolds = n_cvfolds

    def get_fitness(self) -> float:
        """
        Returns
        -------
        fitness : int
            式の計算結果の値。
        """

        # make sure cv always generates the same result for the same params
        # random
        random.seed(42)
        # Numpy
        np.random.seed(42)

        # folds of cv
        folds = StratifiedKFold(n_splits=self.n_cvfolds, shuffle=True, random_state=42)
        # entire predictions of cv
        oof_preds_proba = np.zeros((len(self.y), 2), dtype=np.float32)

        # cv loop
        print('starting cv...')
        for i, (trn_index, tst_index) in enumerate(folds.split(self.X, self.y)):
            print(f' cv round {i + 1}/{self.n_cvfolds}')
            # train-valid
            X_train, y_train = X[trn_index], y[trn_index]
            # test
            X_test, y_test = X[tst_index], y[tst_index]
            # train and valid
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

            # model fit
            model = self.model_const(**self.params)
            # with suppress_stdout():
            mystdout = StringIO()
            try:
                sys.stdout = mystdout # redirect. suppress info logs
                model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)])
                sys.stdout = sys.__stdout__ # lift redirect
            except Exception as ex:
                from datetime import datetime
                now = datetime.now().strftime('%Y%m%d_%H%M%S')
                # record error
                with open(f'errorlog_{now}.txt', 'w') as f:
                    f.write(mystdout.getvalue())
                print(self.params)
                raise ex

            # predict
            pred = model.predict_proba(X_test)
            oof_preds_proba[tst_index] = pred
        
        # logloss
        logloss = log_loss(self.y, oof_preds_proba)

        return -logloss # for logloss, less is better

    @classmethod
    def make_random_instance(cls, params: dict, X: np.array, y: np.array, model_const, n_cvfolds: int) -> HyperParamTuner:
        """
        generate an instance of HyperParamTuner class with random parameters

        Parameters
        ----------
        params : dict
            hyperparameters
        X : np.array
            initial value of X
        y : np.array
            initial value of y
        model_const: 
            model constructor
        mutate_prob:
            probability to cause mutation
        crossover_prob:
            probability to cause crossover

        Returns
        -------
        problem : HyperParamTuner
            A generated instance. Each entry in the dictionary has a initial value.
        """

        search_params = {'learning_rate': 10 ** np.random.randint(-3, 0),
                        'max_depth': np.random.randint(2,20),
                        'num_leaves': np.random.randint(4,40),
                        'feature_fraction': np.random.rand(),
                        'subsample': np.random.rand()}

        params.update(search_params)

        problem = HyperParamTuner(params=params, X=X, y=y, model_const=model_const, n_cvfolds=n_cvfolds)
        return problem

    def _uniform(self, a: float, b: float) -> float:
        return np.random.rand() * (b - a) + a

    def _ratio_distribution(self, x: float) -> float:
        """make a skewed probability distribution between 0 and 1"""


    def _ratio_distribution(self, peak: float) -> float:
        """make a normal distribution between 0 and 1"""
        unif = uniform(0, 1)
        dist_dict = dict(loc=peak, scale=max(peak, 1 - peak) / 3)
        cdf = lambda x: norm.cdf(x, **dist_dict)
        tmp = (cdf(1) - cdf(0)) * unif + cdf(0)
        x = norm.ppf(tmp, **dist_dict)
        return x

    def mutate(self) -> None:
        """
        mutate an instance randomly
        """

        self.params['learning_rate'] *= 10 ** self._uniform(-1, 1)
        
        k = 9
        lambda_ = k / self.params['max_depth']
        self.params['max_depth'] = max(1, int(np.random.gamma(k, lambda_)))
        
        lambda_ = k / self.params['num_leaves']
        self.params['num_leaves'] = max(2, int(np.random.gamma(k, lambda_)))
        max_leaves = 2 ** self.params['max_depth']
        self.params['num_leaves'] = min(max_leaves, self.params['num_leaves'])
        
        self.params['feature_fraction'] = self._ratio_distribution(self.params['feature_fraction'])

        self.params['subsample'] = self._ratio_distribution(self.params['subsample'])

    def _random_swap(self, a, b):
        if np.random.rand() > 0.5:
            return a, b
        else:
            return b, a

    def exec_crossover(
            self, other: HyperParamTuner
            ) -> List[HyperParamTuner]:
        """
        Refer to the other instance and execute crossover

        Parameters
        ----------
        other : HyperParamTuner
            the other instance used for crossover

        Returns
        -------
        result_chromosomes : list of HyperParamTuner
            A list of two instances generated after crossover
        """

        child_1: HyperParamTuner = deepcopy(self)
        child_2: HyperParamTuner = deepcopy(other)

        for param in ['learning_rate', 'max_depth', 'num_leaves', 'feature_fraction', 'subsample']:
            child_1.params[param], child_2.params[param] = self._random_swap(child_1.params[param], child_2.params[param])

        result_chromosomes: List[HyperParamTuner] = [
            child_1, child_2,
        ]
        return result_chromosomes

    def __str__(self) -> str:
        """
        returns the parameter information

        Returns
        -------
        info : str
            string of parameter information
        """

        learning_rate: float = self.params['learning_rate']
        max_depth: float = self.params['max_depth']
        num_leaves: int = self.params['num_leaves']
        feature_fraction: float = self.params['feature_fraction']
        subsample: float = self.params['subsample']
        fitness: float = self.get_fitness()

        info: str = f'learning_rate = {learning_rate}, num_leaves = {num_leaves},\
                    max_depth = {max_depth}, feature_fraction = {feature_fraction}, \
                    subsample = {subsample}, fitness = {fitness}'
        return info


if __name__ == '__main__':

    # change these parameters to your purpose
    params = {'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting':'gbdt',
            'num_boost_round':10000,
            'early_stopping_rounds':30,
            'random_state':0}

    # change these variables to your purpose
    data = load_breast_cancer()
    X = data.data
    y = data.target
    n_cvfolds = 3 # cross validation folds in 
    n_hyperparameter_tuner_population = 5

    # Don't change this
    model_const = lgb.LGBMClassifier

    hyperparameter_tuner_population: List[HyperParamTuner] = \
        [HyperParamTuner.make_random_instance(params=params, X=X, y=y, model_const=model_const, n_cvfolds=n_cvfolds) for _ in range(n_hyperparameter_tuner_population)]
    ga: GeneticAlgorithm = GeneticAlgorithm(
        initial_population=hyperparameter_tuner_population,
        threshold=-0.01,
        max_generations=10,
        mutation_probability=0.2,
        crossover_probability=0.5,
        selection_type=GeneticAlgorithm.SELECTION_TYPE_TOURNAMENT)
    best_chromosome, history = ga.run_algorithm()

    print('### best params ###')
    print(best_chromosome)
