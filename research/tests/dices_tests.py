"""Offline adapter and independent mathematical checks for the DICES experiment."""
from fractions import Fraction
import math
import unittest
import numpy as np
import pandas as pd
from research.dices import (LABELS, TARGET, human_panel,
    single_generation, predictions_from_percentages, run_analysis)
from research.dices_calibration import fit_calibrator, apply_calibrator, group_split
from research.alpaca import prepare_humans as prepare


class DicesValidationTests(unittest.TestCase):
    def records(self):
        return [dict(dataset='dices', experiment_group='main',model_name='judge',
                     target_group=TARGET,generation_index=0,comment_id=i,parsed_percentage=p)
                for i,p in enumerate([0,50,100])]

    def human_inputs(self):
        raw=pd.DataFrame([dict(item_id=42,rater_id=i,context='USER: text\nLAMDA: text',
            response='answer',rater_education='College degree or higher',rater_gender='Woman',
            rater_race='White',rater_age='gen z',Q_overall=y)
            for i,y in enumerate(['Yes','No','Unsure'])])
        return raw,pd.DataFrame([dict(comment_id=0,comment='USER: text\nAI: text\nAI: answer')])

    def test_flags_preserve_uncertainty_as_nonflag_not_safe(self):
        W,meta,audit=human_panel(*self.human_inputs())
        self.assertEqual(W.iloc[0].tolist(),['flag','nonflag','nonflag'])
        self.assertEqual(meta.item_id.tolist(),[42])
        self.assertEqual(audit['original_labels'],dict(Yes=1,No=1,Unsure=1))

    def test_exact_text_and_vote_identity_are_required(self):
        raw,comments=self.human_inputs()
        comments.loc[0,'comment']+='!'
        with self.assertRaisesRegex(ValueError,'join'):human_panel(raw,comments)
        raw,comments=self.human_inputs()
        with self.assertRaisesRegex(ValueError,'Duplicate'):human_panel(pd.concat([raw,raw.iloc[:1]]),comments)

    def test_single_draw_does_not_average_or_substitute(self):
        records=self.records();records[0]['parsed_percentage']=None
        records += [{**r,'generation_index':1,'parsed_percentage':75} for r in self.records()]
        _,p,audit=single_generation(records,3)
        self.assertTrue(np.isnan(p[0]));self.assertEqual(p[1:].tolist(),[.5,1.])
        self.assertEqual(audit['valid'],2)

    def test_duplicate_and_invalid_predictions(self):
        records=self.records()
        with self.assertRaisesRegex(ValueError,'Duplicate'):single_generation(records+records[:1],3)
        records[0]['parsed_percentage']=-1;records[1]['parsed_percentage']=float('nan')
        records[2]['parsed_percentage']=True
        self.assertEqual(single_generation(records,3)[2]['valid'],0)

    def test_probability_direction_and_clipping_are_explicit(self):
        p=predictions_from_percentages(pd.DataFrame({'j':[0,.5,1.]}))
        self.assertEqual(p['j|clip=0'].iloc[0].label_probability('flag'),0.)
        self.assertEqual(p['j|clip=0'].iloc[2].label_probability('flag'),1.)
        self.assertEqual(p['j|clip=0'].iloc[1].value,'nonflag')
        self.assertEqual(p['j|clip=0.01'].iloc[0].label_probability('flag'),.01)

    def test_log_curve_matches_rational_without_replacement_oracle(self):
        W=pd.DataFrame([['flag']*n+['nonflag']*(4-n) for n in [0,1,2,2,3,4]])
        predictions=predictions_from_percentages(pd.DataFrame({'j':[.3]*len(W)}),False)
        samples=[np.arange(len(W)),np.array([2,2,0,3,4,5])]
        run=run_analysis(W,predictions,samples,1729,2,multiplier=10)
        array=W.to_numpy()
        def sequence_probability(row, sequence):
            counts={y:list(row).count(y) for y in LABELS};p=Fraction(1)
            for j,y in enumerate(sequence):
                p *= Fraction(counts[y],len(row)-j);counts[y]-=1
                if not p:return p
            return p
        for k,partitions in run.paper_partitions.items():
            terms=[]
            for benchmark,reference in partitions:
                values=[]
                for i,row in enumerate(array):
                    prefix=[row[j] for j in benchmark]
                    joint=[sum((sequence_probability(train,prefix+[y]) for j,train in enumerate(array) if j!=i),Fraction(0)) for y in LABELS]
                    probability=joint[LABELS.index(row[reference])]/sum(joint)
                    values.append(math.log2(float(probability)))
                terms.append(values)
            expected=[sum(sum(v[i] for i in s)/len(s) for v in terms)/len(terms) for s in samples]
            np.testing.assert_allclose(run.expert_power_curve.df[k],expected,rtol=0,atol=2e-14)

    def test_alpaca_blinding_keeps_labels_and_rater_identity_aligned(self):
        records=[dict(instruction='prompt',output_1='one',output_2='two',
                      preference=1 if i<3 else 2,annotator_index=i) for i in range(5)]
        seen_swaps=set()
        for seed in range(10):
            W,batch,meta=prepare(records,seed)
            seen_swaps.add(bool(meta.swapped.iloc[0]))
            for slot in range(4):
                rater=meta[f'rater_slot_{slot}'].iloc[0]
                chosen_response=batch[0]['response_'+W.iloc[0,slot]]
                self.assertEqual(chosen_response,'one' if rater<3 else 'two')
            pd.testing.assert_frame_equal(W,prepare(records,seed)[0],check_exact=True)
        self.assertEqual(seen_swaps,{False,True})
        with self.assertRaisesRegex(ValueError,'Duplicate'):prepare(records+records[:1])

    def test_calibration_recovers_constant_fraction_and_monotonicity(self):
        x=np.array([0,.2,.5,.8,1.])
        fit=fit_calibrator(x,np.full(5,.3))
        np.testing.assert_allclose(apply_calibrator(x,fit),.3,rtol=0,atol=1e-8)
        fit=fit_calibrator(x,np.array([.1,.2,.5,.8,.9]))
        self.assertTrue(np.all(np.diff(apply_calibrator(x,fit))>=0))
        self.assertLessEqual(fit['training_loss'],fit['constant_training_loss'])
        for bad in [np.full(5,np.nan),np.full(5,1.1)]:
            with self.assertRaises(ValueError):fit_calibrator(x,bad)

    def test_calibration_split_holds_entire_contexts_out(self):
        groups=np.array(['a','a','b','c','d','e','e','f','g','h','i','j'])
        train,test=group_split(groups)
        self.assertFalse(set(groups[train])&set(groups[test]))
        self.assertTrue(np.all(train^test))
        np.testing.assert_array_equal(group_split(groups)[0],train)


if __name__=='__main__':unittest.main()
