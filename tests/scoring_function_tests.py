import numpy as np

from surveyequivalence import AgreementScore,CrossEntropyScore,DiscreteDistributionPrediction, NumericPrediction, DiscretePrediction
from surveyequivalence import DMIScore_for_Hard_Classifier,DMIScore_for_Soft_Classifier

import unittest

import pandas as pd
import numpy as np

class TestScoringFunctions(unittest.TestCase):
    Wrows = [
            ['pos', 'pos', 'neg'], 
            ['pos', 'pos', 'pos'],
            ['pos', 'pos', 'pos'],
            ['pos', 'neg', 'neg'],
            ['pos', 'neg', 'neg'],
            ['pos', 'neg', 'pos'],
            ['neg', 'pos', 'pos'],
            ['neg', 'neg', 'neg'],
            ['neg', 'neg', 'pos'],
            ['neg', 'neg', 'pos']
        ]
    def test_majority_vote_agreement_score(self):

        # 2-majority vote is 2/3 pos, 1 pos, 1 pos, 1/3 pos, 1/3, 2/3, 2/3,0, 1/3, 1/3
        # 3-majority vote is pos,pos,pos,neg,neg,pos,pos,neg,neg, neg
        W = pd.DataFrame(self.Wrows, columns=['r1', 'r2', 'r3'])

        classifier_predictions = [DiscretePrediction('pos') for i in range(10)]

        # 16 total positive; 14 total negative
        scorer = AgreementScore(num_ref_raters_per_virtual_rater=1)
        score=scorer.expected_score_anonymous_raters(classifier_predictions,W)
        self.assertAlmostEqual(score, 16/30, delta=0.001)

        # mean of the 2-majority votes is still 16/30
        scorer = AgreementScore(num_ref_raters_per_virtual_rater=2)
        score=scorer.expected_score_anonymous_raters(classifier_predictions,W)
        self.assertAlmostEqual(score, 16/30, delta=0.001)

        # five items with majority positive; five negative
        scorer = AgreementScore(num_ref_raters_per_virtual_rater=3)
        score=scorer.expected_score_anonymous_raters(classifier_predictions,W)
        self.assertAlmostEqual(score, 5/10, delta=0.001)
    
    def test_majority_vote_cross_entropy_score(self):

        # 2-majority vote is 2/3 pos, 1 pos, 1 pos, 1/3 pos, ...
        # 3-majority vote is pos,pos,pos,neg,neg,pos,pos,neg,neg
        W = pd.DataFrame(self.Wrows, columns=['r1', 'r2', 'r3'])

        classifier_predictions = [DiscreteDistributionPrediction(label_names=['pos','neg'],probabilities=[0.3,0.7]) for i in range(10)]

        # 16/30 * log(.3) + 14/30*log(.7) = -1.166515798
        scorer = CrossEntropyScore(num_ref_raters_per_virtual_rater=1)
        score=scorer.expected_score_anonymous_raters(classifier_predictions,W)
        self.assertAlmostEqual(score, -1.166515798, delta=0.001)

        scorer = CrossEntropyScore(num_ref_raters_per_virtual_rater=2)
        score=scorer.expected_score_anonymous_raters(classifier_predictions,W)
        self.assertAlmostEqual(score, -1.166515798, delta=0.001)

        # 5/10 * log(.3) + 5/10*log(.7) = -1.125769383
        scorer = CrossEntropyScore(num_ref_raters_per_virtual_rater=3)
        score=scorer.expected_score_anonymous_raters(classifier_predictions,W)
        self.assertAlmostEqual(score, -1.125769383, delta=0.001)
    
    def test_Hard_DMI_score(self):
        pos = DiscretePrediction('pos')
        neg = DiscretePrediction('neg')
        classifier_predictions = [pos,pos,neg,pos,neg,pos]
        rater_labels = ['pos','neg','pos','neg','pos','neg']
        #     pos neg
        # pos [[1 3],
        # neg [2 0]]
        scorer = DMIScore_for_Hard_Classifier()
        score = scorer.score(classifier_predictions=classifier_predictions,rater_labels=rater_labels)
        self.assertAlmostEqual(score, 0.166666, delta=0.001)
    
    def test_Hard_DMI_expected_score(self):
        
        W = pd.DataFrame(self.Wrows, columns=['r1', 'r2', 'r3'])

        pos = DiscretePrediction('pos')
        neg = DiscretePrediction('neg')
        classifier_predictions = [pos,pos,pos,pos,pos,pos,pos,pos,neg,neg]
        
        '''
        W_matrix = np.array(
        [[2,1],
        [3,0],
        [3,0],
        [1,2],
        [1,2],
        [2,1],
        [2,1],
        [0,3],
        [1,2],
        [1,2]])

        Classifier_matrix = np.array(
        [[1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [0,1],
        [0,1]])
        
        freq_matrix = np.dot(W_matrix.T,Classifier_matrix)
        freq_matrix = freq_matrix/np.sum(freq_matrix)

        print(freq_matrix)
        > [[0.46666667 0.06666667]
        >  [0.33333333 0.13333333]]

        print(np.linalg.det(freq_matrix))
        > 0.04
        '''
        scorer = DMIScore_for_Hard_Classifier(num_ref_raters_per_virtual_rater=1)
        score=scorer.expected_score_anonymous_raters(classifier_predictions,W)
        self.assertAlmostEqual(score, 0.04, delta=0.001)
        scorer = DMIScore_for_Hard_Classifier(num_ref_raters_per_virtual_rater=2)
        score=scorer.expected_score_anonymous_raters(classifier_predictions,W)
        self.assertAlmostEqual(score, 0.04, delta=0.001)
        '''
        W_matrix = np.array(
        [[1,0],
        [1,0],
        [1,0],
        [0,1],
        [0,1],
        [1,0],
        [1,0],
        [0,1],
        [0,1],
        [0,1]])

        Classifier_matrix = np.array(
        [[1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [0,1],
        [0,1]])

        freq_matrix = np.dot(W_matrix.T,Classifier_matrix)
        freq_matrix = freq_matrix/np.sum(freq_matrix)

        print(freq_matrix)
        > [[0.5 0. ]
        > [0.3 0.2]]
        print(np.linalg.det(freq_matrix))
        > 0.1
        '''

        scorer = DMIScore_for_Hard_Classifier(num_ref_raters_per_virtual_rater=3)
        score=scorer.expected_score_anonymous_raters(classifier_predictions,W)
        self.assertAlmostEqual(score, 0.1, delta=0.001)
    
    def test_Soft_DMI_score(self):
        pos = DiscreteDistributionPrediction(label_names=['pos','neg'],probabilities=[0.7,0.3])
        neg = DiscreteDistributionPrediction(label_names=['pos','neg'],probabilities=[0.3,0.7])
        classifier_predictions = [pos,pos,neg,pos,neg,pos]
        rater_labels = ['pos','neg','pos','neg','pos','neg']
        #     pos neg
        # pos [[1.3 1.7],
        # neg [2.1 0.9]]
        scorer = DMIScore_for_Soft_Classifier()
        score = scorer.score(classifier_predictions=classifier_predictions,rater_labels=rater_labels)
        self.assertAlmostEqual(score, 0.066666, delta=0.001)
    
    def test_Soft_DMI_expected_score(self):
        
        W = pd.DataFrame(self.Wrows, columns=['r1', 'r2', 'r3'])

        pos = DiscreteDistributionPrediction(label_names=['pos','neg'],probabilities=[0.7,0.3])
        neg = DiscreteDistributionPrediction(label_names=['pos','neg'],probabilities=[0.3,0.7])
        classifier_predictions = [pos,pos,pos,pos,pos,pos,pos,pos,neg,neg]
        '''
        W_matrix = np.array(
        [[2,1],
        [3,0],
        [3,0],
        [1,2],
        [1,2],
        [2,1],
        [2,1],
        [0,3],
        [1,2],
        [1,2]])

        Classifier_matrix = np.array(
        [[0.7,0.3],
        [0.7,0.3],
        [0.7,0.3],
        [0.7,0.3],
        [0.7,0.3],
        [0.7,0.3],
        [0.7,0.3],
        [0.7,0.3],
        [0.3,0.7],
        [0.3,0.7]])
        
        freq_matrix = np.dot(W_matrix.T,Classifier_matrix)
        freq_matrix = freq_matrix/np.sum(freq_matrix)

        print(freq_matrix)
        > [[0.34666667 0.18666667]
        > [0.27333333 0.19333333]]
        print(np.linalg.det(freq_matrix))
        > 0.016
        '''
        scorer = DMIScore_for_Soft_Classifier(num_ref_raters_per_virtual_rater=1)
        score=scorer.expected_score_anonymous_raters(classifier_predictions,W)
        self.assertAlmostEqual(score, 0.016, delta=0.001)
        scorer = DMIScore_for_Soft_Classifier(num_ref_raters_per_virtual_rater=2)
        score=scorer.expected_score_anonymous_raters(classifier_predictions,W)
        self.assertAlmostEqual(score, 0.016, delta=0.001)
        '''
        W_matrix = np.array(
        [[1,0],
        [1,0],
        [1,0],
        [0,1],
        [0,1],
        [1,0],
        [1,0],
        [0,1],
        [0,1],
        [0,1]])
        Classifier_matrix = np.array(
        [[0.7,0.3],
        [0.7,0.3],
        [0.7,0.3],
        [0.7,0.3],
        [0.7,0.3],
        [0.7,0.3],
        [0.7,0.3],
        [0.7,0.3],
        [0.3,0.7],
        [0.3,0.7]])
        freq_matrix = np.dot(W_matrix.T,Classifier_matrix)
        freq_matrix = freq_matrix/np.sum(freq_matrix)

        print(freq_matrix)
        > [[0.35 0.15]
        > [0.27 0.23]]
        print(np.linalg.det(freq_matrix))
        > 0.04
        '''
        scorer = DMIScore_for_Soft_Classifier(num_ref_raters_per_virtual_rater=3)
        score=scorer.expected_score_anonymous_raters(classifier_predictions,W)
        self.assertAlmostEqual(score, 0.04, delta=0.001)
    

if __name__ == '__main__':
    unittest.main()
