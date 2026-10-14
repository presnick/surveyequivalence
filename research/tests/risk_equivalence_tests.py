"""Independent enumerations and regression checks for risk survey calculations."""
import itertools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from research import risk_survey_equivalence as risk
from research.sports_survey_equivalence import exact_curve


class RiskEquivalenceTests(unittest.TestCase):
    def test_squared_loss_matches_all_distinct_panels(self):
        for a,truth in [(np.array([[1.,4.],[3.,-2.],[7.,9.],[0.,3.]]),[2.,5.]),
                        (np.array([[0.,.2],[1.,.4],[0.,.8],[1.,.1]]),[1.,0.])]:
            expected=[np.mean([(a[list(s)].mean(0)-truth)**2
                               for s in itertools.combinations(range(len(a)),k)],axis=0)
                      for k in range(1,len(a)+1)]
            np.testing.assert_allclose(risk.finite_mse(a,truth,np.arange(1,len(a)+1)),expected)

    def test_crossings_keep_nonmonotonicity_equality_and_censoring(self):
        curves=np.array([[.9,.7,.8],[.9,.85,.81],[.8,.8,.8]])
        np.testing.assert_array_equal(risk.crossing(curves,.8),[2,4,1])

    def test_unsupported_panel_rejected(self):
        with self.assertRaises(ValueError):
            risk.finite_mse(np.array([[1],[2]]),[0],[3])

    def test_block_draws_preserve_local_order_and_wrap(self):
        x=risk.block_indices(17,5,np.random.default_rng(2),reps=30)
        self.assertEqual(x.shape,(30,17))
        for start in [0,5,10]:
            np.testing.assert_array_equal(np.diff(x[:,start:start+5],axis=1)%17,np.ones((30,4)))
        self.assertTrue(((x>=0)&(x<17)).all())

    def test_fred_extra_rows_and_missing_observation(self):
        dates=pd.date_range('2000-01-01',periods=202,freq='MS')
        body='<table id="data-table-observations">'+''.join(
            f'<tr><th>{d:%Y-%m-%d}</th><td>{i}</td></tr>' for i,d in enumerate(dates[:201]))+'</table>'
        body+=f'<div id="extra-rows">#{dates[201]:%Y-%m-%d}|999#2016-11-01|.</div>'
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'series.html').write_text(body)
            with patch.object(risk,'CACHE',Path(tmp)):
                result=risk.fred_table('series.html')
        self.assertEqual(len(result),202)
        self.assertEqual(result.iloc[-1],999)
        self.assertNotIn(pd.Timestamp('2016-11-01'),result.index)

    def test_decision_propensity_threshold_matches_enumerated_actions(self):
        proportions=np.array([0,.2,.6,.8,1.])
        for sign in [1,-1]:
            curve=exact_curve(sign*(2*np.rint(5*proportions)-5),len(proportions))
            expected=[]
            for k in range(1,len(proportions)+1):
                means=np.array([np.mean(s) for s in itertools.combinations(proportions,k)])
                signed=sign*(means-.5)
                expected.append(np.mean((signed>1e-12)+.5*np.isclose(signed,0)))
            np.testing.assert_allclose(curve,expected,atol=1e-14)


if __name__=='__main__':
    unittest.main()
