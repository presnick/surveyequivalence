from surveyequivalence import DiscreteLabelsWithNoise, DiscreteState, MockClassifier, generate_labels
import numpy as np

num_items_per_dataset = 1000
num_labels_per_item = 10

def make_noisier_binary_states(states, multiplier):

    def make_noisier_state(state, multiplier):
        pr_pos, pr_neg = state.probabilities
        if pr_pos >= .5:
            new_pr_pos = pr_pos / multiplier

        else:
            new_pr_pos = pr_pos * multiplier
        new_pr_neg = 1 - new_pr_pos
        print([new_pr_pos, new_pr_neg])

        return DiscreteState(state_name=state.state_name,
                             labels=state.labels,
                             probabilities=[new_pr_pos, new_pr_neg])

    unique_states = list(set(states))
    d = {s: make_noisier_state(s, multiplier) for s in unique_states}

    return np.array([d[s] for s in states])

def make_discrete_dataset_1():
    state_generator = \
        DiscreteLabelsWithNoise(states=[DiscreteState(state_name='pos',
                                                      labels=['pos', 'neg'],
                                                      probabilities=[.9, .1]),
                                        DiscreteState(state_name='neg',
                                                      labels=['pos', 'neg'],
                                                      probabilities=[.25, .75])
                                        ],
                                probabilities=[.8, .2]
                                )

    item_states = state_generator.draw_states(num_items_per_dataset)
    dataset = generate_labels(item_states, num_labels_per_item)
    mock_classifiers=[
        MockClassifier(name='.95 .2',
                       pos_state_predictor=[.95, .05],
                       neg_state_predictor=[.2, .8]),
        MockClassifier(name='.92 .24',
                       pos_state_predictor=[.92, .08],
                       neg_state_predictor=[.24, .76]),
        MockClassifier(name='one expert',
                       pos_state_predictor=[.9, .1],
                       neg_state_predictor=[.25, .75])
    ]
    return (item_states, dataset, mock_classifiers)

def make_discrete_dataset_2():
    state_generator = \
        DiscreteLabelsWithNoise(states=[DiscreteState(state_name='pos',
                                                      labels=['pos', 'neg'],
                                                      probabilities=[.5, .5]),
                                        DiscreteState(state_name='neg',
                                                      labels=['pos', 'neg'],
                                                      probabilities=[.3, .7])
                                        ],
                                probabilities=[.5, .5]
                                )

    item_states = state_generator.draw_states(num_items_per_dataset)
    dataset = generate_labels(item_states, num_labels_per_item)



def make_discrete_dataset_3():
    state_generator = \
        DiscreteLabelsWithNoise(states=[DiscreteState(state_name='pos',
                                                      labels=['pos', 'neg'],
                                                      probabilities=[.4, .6]),
                                        DiscreteState(state_name='neg',
                                                      labels=['pos', 'neg'],
                                                      probabilities=[.7, .3])
                                        ],
                                probabilities=[.4, .6]
                                )

    item_states = state_generator.draw_states(num_items_per_dataset)
    dataset = generate_labels(item_states, num_labels_per_item)
    return (item_states, dataset, [])