import os
import random
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Sequence, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from surveyequivalence import Prediction, DiscretePrediction, DiscreteDistributionPrediction

class BernoulliNoise:
    def __init__(self,p=0.5) -> None:
        self.p=p
    def draw(self):
        r=random.random()
        return r<self.p

def noisy_wiki_attack(dirname='',inputfile='wiki_attack_labels_and_predictor.csv',outputfile='wiki_attack_labels_and_predictor.csv',new_file=True,noise=BernoulliNoise(0.1)):
    """
        Load the wiki_attack_labels_and_predictor dataset and add some noise to the labels
        
        Parameters
        ----------
        dirname: directory string
        inputfile: filename, a string
        outputfile: filename, a string
        new_file: if True, append timestamp to output filename
        noise: an instance of BernoulliNoise, which returns True with some probability, indicating that we should add noise to that instance

        Returns
        -------
        None
        
    """

    path = f'data/{dirname}/'

    dataset = pd.read_csv(f"{path}/{inputfile}", index_col=0)
    V = dataset.values
 
    for row in V:
        n = int(row[2]) # number of raters
        m = int(row[1]) # number of raters giving positive label
        pos_num = m
        for i in range(m):
            # for each positive label, "noise" probability of turning it to negative 
            if noise.draw():
                pos_num -= 1
        for i in range(n-m):
            # for each negative label, "noise" probability of turning it to positive
            if noise.draw():
                pos_num += 1
        # save the new counts in "row"
        row[0]= pos_num*1.0/n
        row[1]= pos_num

    dataset=pd.DataFrame(data=V,index=dataset.index,columns=dataset.columns)

    if new_file:
        path = f'data/{dirname}/{datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")}'
    else:
        path = f'data/{dirname}/'
    
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

    dataset.to_csv(f'{path}/{outputfile}')


if __name__ == '__main__':
    noisy_wiki_attack()
