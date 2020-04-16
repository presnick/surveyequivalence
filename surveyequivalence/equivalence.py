from abc import ABC, abstractmethod
from typing import Sequence, Dict
import numpy as np
import random
from .predictors import Prediction
from .scoring_functions import Score

N = 1000

def power_curve(W: np.matrix, comb: Prediction, score: Score, K: int ):
    assert(K>0)
    for k in range(1,K+1): #TODO check 1, and K
        # Sample N rows from the rating matrix W with replacement
        I = W[np.random.choice(W.shape[0], N, replace=True)]

        #for each item/row in sample
        for item in I:
            #sample ratings from nonzero ratings of the item
            sample_ratings = np.random.choice(item[np.nonzero(item)], k+1)
            predictor_ratings = sample_ratings[0:-1]
            reference_rater = sample_ratings[-1:]
            comb.combiner()  #TODO stopped here.
