import pandas as pd
import numpy as np
from numpy.random import binomial, beta
from scipy.special import comb


def draw_labels(label_generator, n=1):
    """
    :param label_generator:
    :param n: number of labels to generate
    :return: array of labels

    Some example inputs:

    - A discrete distribution with 90% positive labels, 10% negative
    ```
    {
        'type': 'discrete',
        'labels': ['pos', 'neg'],
        'probabilities': [.9, .1]
    }
    ```

    - A beta scaled to have mean 3.5 on a 7-point likert scale
    ```
    {
        'type': 'beta',
        'params': {'alpha': 2, 'beta': 2},
        'scaling': 6,
        'translation': +1
    }```
    """
    if label_generator['type'] == 'discrete':
        return np.random.choice(
            label_generator['labels'],
            n,
            p=label_generator['probabilities']
        )
    elif label_generator['type'] == 'beta':
        return "not yet implemented"



def discrete_labels_with_noise(states, n=1):
    """

    :param states: a sequence of dictionaries giving probability of each label being the true state and conditional probabilities of observing labels given that state
    :param n: number of
    :return:

    Sample states input value
    ```
    [
        {
            'prob': .8,
            'observations_conditional_on_state': {
                'type': 'discrete',
                'state_name': 'pos'
                'labels': ['pos', 'neg'],
                'probabilities': [.9, .1]
            }
        },
        {
            'prob': .2,
            'observations_conditional_on_state': {
                'type': 'discrete',
                'state_name': 'neg',
                'labels': ['pos', 'neg'],
                'probabilities': [.25, .75]
            }
        }
    ]
    ```
    """
    return np.random.choice(
        [d['observations_conditional_on_state'] for d in states],
        size=n,
        p=[d['prob'] for d in states]
    )


def mix_of_two_betas():
    return "not yet implemented"


def generate_labels(item_states, num_labels_per_item=10):
    return pd.DataFrame(
        [draw_labels(state, num_labels_per_item) for state in item_states],
        columns = ["r{}".format(i) for i in range(1, num_labels_per_item+1)]
    )

def make_test_datasets():
    num_items_per_dataset = 1000
    num_labels_per_item = 10
    state_generator_1 = [
        {
            'prob': .8,
            'observations_conditional_on_state': {
                'type': 'discrete',
                'state_name': 'pos',
                'labels': ['pos', 'neg'],
                'probabilities': [.9, .1]
            }
        },
        {
            'prob': .2,
            'observations_conditional_on_state': {
                'type': 'discrete',
                'state_name': 'neg',
                'labels': ['pos', 'neg'],
                'probabilities': [.25, .75]
            }
        }
    ]

    item_states_1 = discrete_labels_with_noise(state_generator_1, num_items_per_dataset)
    dataset_1 = generate_labels(item_states_1, num_labels_per_item)
    dataset_1['true_prob'] = [d['state_name'] for d in item_states_1]

    return dataset_1

def main():
    print(make_test_datasets())

main()
