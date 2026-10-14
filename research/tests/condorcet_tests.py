"""Independent mathematical checks for the Condorcet research workflow."""
import itertools
import math
import unittest

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import beta, betabinom

from research.condorcet_data import city_votes
from research.condorcet_equivalence import binomial_curve, finite_curves, whole_crossing
from research.sports_survey_equivalence import exact_curve
from research.panel_power import first_equivalent_size


class CondorcetTests(unittest.TestCase):
    def test_finite_pools_against_enumeration(self):
        for n in range(1,9):
            for c in range(n+1):
                votes=[1]*c+[0]*(n-c)
                expected=[]
                for k in range(1,n+1):
                    scores=[float(sum(s)>k/2)+.5*(sum(s)==k/2)
                            for s in itertools.combinations(votes,k)]
                    expected.append(np.mean(scores))
                np.testing.assert_allclose(finite_curves([n],[c],n)[0],expected,atol=1e-13)

    def test_binomial_against_joint_event_enumeration(self):
        for p in [0,.2,.5,.65,1]:
            for k in range(1,9):
                expected=0
                for votes in itertools.product([0,1],repeat=k):
                    c=sum(votes)
                    expected+=(float(c>k/2)+.5*(c==k/2))*p**c*(1-p)**(k-c)
                self.assertAlmostEqual(float(binomial_curve(k,p)),expected,places=13)

    def test_weighted_subsets_with_abstaining_weights(self):
        votes=[0,-2,1,3,-1,0,5]
        expected=[]
        for k in range(1,len(votes)+1):
            expected.append(np.mean([float(sum(s)>0)+.5*(sum(s)==0)
                              for s in itertools.combinations(votes,k)]))
        np.testing.assert_allclose(exact_curve(votes,len(votes)),expected,atol=1e-14)

    def test_beta_mixture_matches_tail(self):
        p,rho=.65,.2
        a,b=p*(1/rho-1),(1-p)*(1/rho-1)
        for k in [3,8,15]:
            integral=quad(lambda t:float(binomial_curve(k,t))*beta.pdf(t,a,b),0,1)[0]
            tail=betabinom.sf(k//2,k,a,b)+(.5*betabinom.pmf(k//2,k,a,b) if k%2==0 else 0)
            self.assertAlmostEqual(integral,tail,places=8)

    def test_first_crossing_censoring_and_plateaus(self):
        self.assertEqual(int(whole_crossing([.6,.6,.7,.65],.68)),3)
        self.assertEqual(int(whole_crossing([.6,.6,.7,.65],.8)),5)
        self.assertAlmostEqual(first_equivalent_size([1,2,3,4],[.6,.6,.7,.65],.65),2.5)

    def test_missing_city_decision_is_not_a_wrong_vote(self):
        # Full-session cohort with alternating left/right positions.
        d=pd.DataFrame({'Subject':[1]*1000,'Session':[1]*500+[2]*500,
            'CityLeft':['A','B']*500,'CityRight':['B','A']*500,
            'CityLeftPop':[100,200]*500,'CityRightPop':[200,100]*500,
            'CityResponse':['CityRightBox','CityLeftBox']*500,
            'DecisionResponse':['CityRightBox','CityLeftBox']*500,
            'CorrectResponse':['CityRightBox','CityLeftBox']*500,
            'DecisionAcc':[1]*1000,'Conf':[75]*1000})
        d.loc[0,['CityResponse','DecisionResponse']]=np.nan
        d.loc[0,'DecisionAcc']=0
        v,audit=city_votes(d)
        self.assertEqual(len(v),999)
        self.assertEqual(audit['missing_decisions_excluded'],1)
        self.assertTrue(v.vote.eq(1).all())
        self.assertTrue(v.truth.eq(1).all())
        self.assertTrue(v.item_id.eq('A | B').all())


if __name__=='__main__':unittest.main()
