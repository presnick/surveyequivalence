import numpy as np

from surveyequivalence import AgreementScore,CrossEntropyScore,DiscreteDistributionPrediction, NumericPrediction, DiscretePrediction


import unittest

import pandas as pd
import numpy as np

class TestScoringFunctions(unittest.TestCase):
    def test_majority_vote_agreement_score(self):

        Wrows = [
            ['pos', 'pos', 'neg'], 
            ['pos', 'pos', 'pos'],
            ['pos', 'pos', 'pos'],
            ['pos', 'neg', 'neg'],
            ['pos', 'neg', 'neg'],
            ['pos', 'neg', 'pos'],
            ['neg', 'pos', 'pos'],
            ['neg', 'neg', 'neg'],
            ['neg', 'neg', 'pos']
        ]
        # 2-majority vote is 2/3 pos, 1 pos, 1 pos, 1/3 pos, ...
        # 3-majority vote is pos,pos,pos,neg,neg,pos,pos,neg,neg
        W = pd.DataFrame(Wrows, columns=['r1', 'r2', 'r3'])

        classifier_predictions = [DiscretePrediction('pos') for i in range(9)]

        scorer = AgreementScore(num_ref_raters_per_virtual_rater=1)
        score=scorer.expected_score_anonymous_raters(classifier_predictions,W)
        self.assertAlmostEqual(score, 0.55555, delta=0.001)

        scorer = AgreementScore(num_ref_raters_per_virtual_rater=2)
        score=scorer.expected_score_anonymous_raters(classifier_predictions,W)
        self.assertAlmostEqual(score, 0.55555, delta=0.001)

        scorer = AgreementScore(num_ref_raters_per_virtual_rater=3)
        score=scorer.expected_score_anonymous_raters(classifier_predictions,W)
        self.assertAlmostEqual(score, 0.55555, delta=0.001)
    
    def test_majority_vote_cross_entropy_score(self):

        Wrows = [
            ['pos', 'pos', 'neg'], 
            ['pos', 'pos', 'pos'],
            ['pos', 'pos', 'pos'],
            ['pos', 'neg', 'neg'],
            ['pos', 'neg', 'neg'],
            ['pos', 'neg', 'pos'],
            ['neg', 'pos', 'pos'],
            ['neg', 'neg', 'neg'],
            ['neg', 'neg', 'pos']
        ]
        # 2-majority vote is 2/3 pos, 1 pos, 1 pos, 1/3 pos, ...
        # 3-majority vote is pos,pos,pos,neg,neg,pos,pos,neg,neg
        W = pd.DataFrame(Wrows, columns=['r1', 'r2', 'r3'])

        classifier_predictions = [DiscreteDistributionPrediction(label_names=['pos','neg'],probabilities=[0.3,0.7]) for i in range(9)]

        scorer = CrossEntropyScore(num_ref_raters_per_virtual_rater=1)
        score=scorer.expected_score_anonymous_raters(classifier_predictions,W)
        self.assertAlmostEqual(score, -1.1936, delta=0.001)

        scorer = CrossEntropyScore(num_ref_raters_per_virtual_rater=2)
        score=scorer.expected_score_anonymous_raters(classifier_predictions,W)
        self.assertAlmostEqual(score, -1.1936, delta=0.001)

        scorer = CrossEntropyScore(num_ref_raters_per_virtual_rater=3)
        score=scorer.expected_score_anonymous_raters(classifier_predictions,W)
        self.assertAlmostEqual(score, -1.1936, delta=0.001)
    
    def test_DMI_score(self):

        pass
    

if __name__ == '__main__':
    unittest.main()