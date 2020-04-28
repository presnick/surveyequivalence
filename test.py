# import pandas as pd
import numpy as np
import pandas as pd
# from numpy.random import binomial, beta
# from scipy.special import comb


from surveyequivalence import generate_labels, State, DiscreteState, \
    DistributionOverStates, DiscreteLabelsWithNoise, MixtureOfBetas, \
    frequency_combiner, DiscreteDistributionPrediction, \
    agreement_score, \
    AnalysisPipeline


def make_test_datasets():
    num_items_per_dataset = 100
    num_labels_per_item = 10
    state_generator_1 = \
        DiscreteLabelsWithNoise(states=[DiscreteState(state_name='pos',
                                                      labels=['pos', 'neg'],
                                                      probabilities=[.9, .1]),
                                        DiscreteState(state_name='neg',
                                                      labels=['pos', 'neg'],
                                                      probabilities=[.25, .75])
                                        ],
                                probabilities=[.8, .2]
                                )

    item_states_1 = state_generator_1.draw_states(num_items_per_dataset)
    dataset_1 = generate_labels(item_states_1, num_labels_per_item)
    # Add a column with the "true" noiseless label
    # dataset_1['true_state'] = [s.state_name for s in item_states_1]

    return dataset_1

def main():
    d1 = make_test_datasets()
    print(d1)
    print("*****testing combiners********")
    pred1 = frequency_combiner(['pos', 'neg'], np.array([(1, 'pos'), (2, 'neg'), (4, 'neg')]))
    pred2 = frequency_combiner(['pos', 'neg'], np.array([(1, 'neg'), (2, 'neg'), (4, 'neg')]))
    assert pred1.probabilities == [0.3333333333333333, 0.6666666666666666]
    assert pred2.probabilities == [0.0, 1.0]
    print("*****testing scoring functions*******")
    assert agreement_score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['b', 'b', 'b']) == 0.6666666666666666
    assert agreement_score([DiscreteDistributionPrediction(['a', 'b'], prs) for prs in [[.3, .7], [.4, .6], [.6, .4]]],  ['a', 'b', 'b']) == 0.3333333333333333

    print("******running analysis pipeline on the generated sample dataset***********")
    p = AnalysisPipeline(d1,
                         frequency_combiner,
                         agreement_score,
                         allowable_labels=['pos', 'neg'],
                         null_prediction=DiscreteDistributionPrediction(['pos', 'neg'], [1, 0]),
                         num_runs=5
                         )
    # print("Power curve means")
    # print(p.power_curve.means)
    # print("Power curve widths of confidence intervals")
    # print(p.power_curve.cis)
    results = pd.concat([p.power_curve.means,p.power_curve.cis], axis=1)
    results.columns = ['mean', 'ci_width']
    print(results)
    for i in range (15):
        thresh = .75 + .01*i
        print(f"\tsurvey equivalence for {thresh} is ", p.power_curve.compute_equivalence(thresh))

    p.plot()


if __name__ == '__main__':
    main()