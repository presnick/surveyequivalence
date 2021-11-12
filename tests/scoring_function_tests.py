import numpy as np

from surveyequivalence import AgreementScore,CrossEntropyScore,DiscreteDistributionPrediction, NumericPrediction, DiscretePrediction


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

        classifier_predictions = [DiscreteDistributionPrediction(label_names=['pos','neg'],probabilities=[0.3,0.7]) for i in range(9)]

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
    
    def test_DMI_score(self):

        pass
    

if __name__ == '__main__':
    unittest.main()
