import os
import random
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Sequence, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import sys
sys.path.append("..") 
from surveyequivalence import Prediction, DiscretePrediction, DiscreteDistributionPrediction

class BernoulliNoise:
    def __init__(self,p=0.5) -> None:
        self.p=p
    def draw(self):
        r=random.random()
        if(r<self.p):
            return 1
        else:
            return 0

def noisy_wiki_attack(dirname='',inputfile='wiki_attack_labels_and_predictor.csv',outputfile='wiki_attack_labels_and_predictor.csv',new_file=1):

    path = f'data/{dirname}/'

    dataset = pd.read_csv(f"{path}/{inputfile}", index_col=0)

    V = dataset.values
    print(V)

    noise = BernoulliNoise(0.1)

    for row in V:
        n = int(row[2])
        m = int(row[1])
        pos_num = m
        for i in range(m):
            if noise.draw():
                pos_num -= 1
        for i in range(n-m):
            if noise.draw():
                pos_num += 1
        row[0]= pos_num*1.0/n
        row[1]= pos_num

    print(V)

    dataset=pd.DataFrame(data=V,index=dataset.index,columns=dataset.columns)
    print(dataset)

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

