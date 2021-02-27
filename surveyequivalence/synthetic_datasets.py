from typing import Sequence, Dict

import os
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
from surveyequivalence import Prediction

########### States #############################

class State:

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def draw_labels(self, n):
        pass


class DiscreteState(State):
    """
    A discrete distribution over possible labels


    Parameters
    ----------
    state_name
    labels
        A sequence of strings; the allowable labels
    probabilities
        A sequence of the same length, with values adding to one, giving probabilities for each of the label strings
    """

    def __init__(self,
                 state_name: str,
                 labels: Sequence[str],
                 probabilities: Sequence[float],
                 ):
        super().__init__()
        self.state_name = state_name
        self.labels = labels
        self.probabilities = probabilities

    def __repr__(self):
        return f"DiscreteState {self.state_name}: {list(zip(self.labels, self.probabilities))}"

    def pr_dict(self):
        return {l: p for (l, p) in zip(self.labels, self.probabilities)}

    def draw_labels(self, n: int):
        """
        Make n iid draws of discrete labels from the distribution

        Parameters
        ----------
        n
            How many labels to draw from the distribution

        Returns
        -------
            a single item or a numpy array
        """
        return np.random.choice(
            self.labels,
            n,
            p=self.probabilities
        )

############ Distributions over states ###############

class DistributionOverStates(ABC):
    """
    Abstract base class
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def draw_states(self, n: int):
        pass


class DiscreteDistributionOverStates(DistributionOverStates):
    """

    Parameters
    ----------
    states
        a sequence of State objects
    probabilities
        a same length sequence of floats representing probabilities of the item states
    """

    def __init__(self, states: Sequence[State], probabilities: Sequence[float]):
        super().__init__()
        self.probabilities = probabilities
        self.states = states

    def draw_states(self, n: int) -> Sequence[DiscreteState]:
        """

        Parameters
        ----------
        n

        Returns
        -------
            a single item or numpy array of State instances, drawn iid from the probability distribution
        """
        return np.random.choice(
            self.states,
            size=n,
            p=self.probabilities
        )

class FixedStateGenerator(DiscreteDistributionOverStates):
    def draw_states(self, n: int):
        """
        Draw exactly in proportion to probabilities, rather than each draw random according to the probabilities
        Parameters
        ----------
        n
            How many items to draw

        Returns
        -------
        list of State instances
        """
        counts = [int(round(pr*n)) for pr in self.probabilities]
        if sum(counts) < n:
            counts[0] += 1

        states_list = []
        for count, state in zip(counts, self.states):
            for _ in range(count):
                states_list.append(state)
        return states_list

class MixtureOfBetas(DistributionOverStates):
    def __init__(self):
        super().__init__()
        pass

    def draw_states(self, n) -> Sequence[DiscreteState]:
        pass


############ Mock Classifier ###############

class MockClassifier:
    """A mock classifier has access to each item's state when generating a prediction,
    something that a real classifier would not have access to

    Parameters
    ----------
    name
    label_predictions
        a dictionary mapping from item state names to Predictions
    """
    def __init__(self,
                 name: str,
                 label_predictors: Dict[str, Prediction],
                 ):
        self.name = name
        self.label_predictors = label_predictors

    def make_predictions(self, item_states: Sequence[State])->Sequence[Prediction]:
        """
        Parameters
        ----------
        item_states
            a sequence of State objects, representing the states of some items
        Returns
        -------
        a sequence of Prediction objects, one for each item
        """

        return [self.label_predictors[s.state_name] for s in item_states]

class MappedDiscreteMockClassifier(MockClassifier):
    """
    A mock classifier that maps an item state to a Prediction, \
    draws a discrete label from that, \
    and then maps that discrete label to another Prediction.

    Parameters
    ----------
    name
    label_predictions
        a dictionary mapping from item state names to Predictions
    """
    def __init__(self,
                 name,
                 label_predictors: Dict[str, Prediction],
                 prediction_map: Dict[str, Prediction] # a dictionary mapping from labels to continuous Predictions
                 ):
        super().__init__(name, label_predictors)
        self.prediction_map = prediction_map

    def make_predictions(self, item_states):
        return [self.prediction_map[self.label_predictors[s.state_name].draw_discrete_label()] for s in item_states]


############ Synthetic Dataset Generator ###############

class SyntheticDatasetGenerator:
    """
    Generator for a set of items with some raters per item.
    Items are defined by States, which are drawn from a DistributionOverStates.
    Each State is a distribution over labels.
    Each label is an i.i.d. draw from the State

    Parameters
    ----------
    item_state_generator
    num_items_per_dataset
    num_labels_per_item
        How many raters to generate labels for, for each item
    mock_classifiers
        A list of MockClassifier instances, which generate label predictions based on the item state
    name
        A text string naming this dataset generator
    """
    def __init__(self,
                 item_state_generator: DistributionOverStates,
                 num_items_per_dataset=1000,
                 num_labels_per_item=10,
                 mock_classifiers=None,
                 name=''):
        self.item_state_generator = item_state_generator
        self.num_items_per_dataset = num_items_per_dataset
        self.num_labels_per_item = num_labels_per_item
        self.name = name
        # make a private empty list, not a shared default empty list if mock_classifiers not specified
        self.mock_classifiers = mock_classifiers if mock_classifiers else []

        self.reference_rater_item_states = item_state_generator.draw_states(num_items_per_dataset)

    def generate_labels(self, item_states, num_labels_per_item=None, rater_prefix="e"):
        """
        Normally called with item_states=self.reference_rater_item_states

        Parameters
        ----------
        self
        item_states
            a list of States, one for each item
        num_labels_per_item=None
            if None, use self.num_labels_per_item
        rater_prefix="e"
            Rater columns are named as `f"{rater_prefix}_{i}"` where i is an integer
        Returns
        -------
        A pandas DataFrame with one row for each item and one column for each rater. Cells are labels.
        """
        if not num_labels_per_item:
            num_labels_per_item = self.num_labels_per_item
        return pd.DataFrame(
            [state.draw_labels(self.num_labels_per_item) for state in item_states],
            columns=[f"{rater_prefix}_{i}" for i in range(1, self.num_labels_per_item + 1)]
        )

class SyntheticBinaryDatasetGenerator(SyntheticDatasetGenerator):
    """Dataset generator for binary labels

    Only additional parameters for this subclass are documented here.

    Parameters
    ----------
    pct_noise=0
        In addition to the reference rater labels, this generator can generator labels from "other" raters. \
        With probability pct_noise the binary labels will be drawn from a 50-50 coin flip, and otherwise from\
        the item's State.
        If pct_noise==0, the other raters' labels will always be i.i.d draws from the same distribution as the
        reference rater labels.
    k_other_raters_per_label=1
        The number of other raters to generate labels for.
    """
    def __init__(self, item_state_generator, num_items_per_dataset=50, num_labels_per_item=3,
                 mock_classifiers=None, name=None,
                 pct_noise=0, k_other_raters_per_label=1):
        super().__init__(item_state_generator, num_items_per_dataset, num_labels_per_item, mock_classifiers, name)

        self.k_other_raters_per_label = k_other_raters_per_label
        if pct_noise > 0:
            self.other_rater_item_states = self.make_noisier_binary_states(pct_noise)
        else:
            self.other_rater_item_states = None

    def make_noisier_binary_states(self, noise_multiplier):

        def make_noisier_state(state, pct_noise):
            pr_pos, pr_neg = state.probabilities
            new_pr_pos = (1 - pct_noise) * pr_pos + pct_noise * .5
            new_pr_neg = (1 - pct_noise) * pr_neg + pct_noise * .5

            return DiscreteState(state_name=state.state_name,
                                 labels=state.labels,
                                 probabilities=[new_pr_pos, new_pr_neg])

        unique_states = list(set(self.reference_rater_item_states))
        d = {s: make_noisier_state(s, noise_multiplier) for s in unique_states}

        return np.array([d[s] for s in self.reference_rater_item_states])

    def plot_item_state_distribution(self):
        """called if you are making a standalone graph; for insets, .make_histogram is called directly"""

        # make Figure and axes objects
        fig, ax = plt.subplots()

        fig.set_size_inches(18.5, 10.5)

        # add histogram
        self.make_histogram(ax)

        # save figure
        if not os.path.isdir('plots'):
            os.mkdir('plots')
        fig.savefig(f'plots/{self.name} {datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")}.png')

        pass

    def make_histogram(self, ax):
        """
        Parameters
        ----------
        ax
            A matplotlib Axes instance
        """
        ax.set_xlabel('State')
        ax.set_ylabel('Frequency')
        ax.set(xlim=(0, 1))
        ax.hist([s.probabilities[0] for s in self.reference_rater_item_states],
                25,
                align='right')
        # ax.set_yticks([])
        # ax.set_xticks([])


class Dataset():
    """
    A Dataset will have attributes set: dataset, other_rater_dataset, classifiers, classifier_predictions
    """
    def __init__(self, dataset, other_rater_dataset, classifiers):
        self.dataset = dataset
        self.other_rater_dataset = other_rater_dataset
        self.classifiers = classifiers


class SyntheticDataset(Dataset):
    """
    Parameters
    ----------
    ds_generator

    Sets all the attributes, by running the SyntheticBinaryDatasetGenerator
    """
    def __init__(self, ds_generator:SyntheticBinaryDatasetGenerator):
        self.ds_generator = ds_generator

        self.set_datasets()

    def set_datasets(self):
        ds_generator = self.ds_generator

        # create the reference_rater dataset
        self.dataset = ds_generator.generate_labels(ds_generator.reference_rater_item_states)

        # create the other_rater dataset if applicable
        if ds_generator.other_rater_item_states is not None:
            other_rater_dataset = ds_generator.generate_labels(ds_generator.other_rater_item_states,
                                                           num_labels_per_item=ds_generator.num_labels_per_item * ds_generator.k_other_raters_per_label,
                                                           rater_prefix='a')
            if ds_generator.k_other_raters_per_label > 1:
                # get a group of k other_rater labelers and take their majority label as the actual label
                majority_winners = stats.mode(other_rater_dataset.reshape(-1, k_other_raters_per_label))
                print(majority_winners)
                foobar  # this code hasn't been tested yet, so break if someone tries using it
                self.other_rater_dataset = stats.mode(other_rater_dataset.reshape(-1, k_other_raters_per_label)).mode
            else:
                self.other_rater_dataset = other_rater_dataset

        # create the classifier predictions for any mock classifiers
        self.classifier_predictions = pd.DataFrame({mc.name: mc.make_predictions(ds_generator.reference_rater_item_states)
                                                    for mc in ds_generator.mock_classifiers})


def make_perceive_with_noise_datasets():
    def make_perceive_with_noise_datasets(epsilon):
        pos_state_probabilities = [1 - epsilon, epsilon]
        neg_state_probabilities = [.05 + epsilon, .95 - epsilon]
        item_state_generator = \
            DiscreteDistributionOverStates(states=[DiscreteState(state_name='pos',
                                                          labels=['pos', 'neg'],
                                                          probabilities=pos_state_probabilities),
                                            DiscreteState(state_name='neg',
                                                          labels=['pos', 'neg'],
                                                          probabilities=neg_state_probabilities)
                                            ],
                                    probabilities=[.8, .2]
                                    )

        dsg = SyntheticBinaryDatasetGenerator(item_state_generator=item_state_generator,
                                              name=f'80% {pos_state_probabilities}; 20% {neg_state_probabilities}'
                                              )

        dsg.mock_classifiers.append(MockClassifier(
            name='h_infinity: ideal classifier',
            label_predictors={
                'pos': DiscreteDistributionPrediction(['pos', 'neg'], pos_state_probabilities),
                'neg': DiscreteDistributionPrediction(['pos', 'neg'], neg_state_probabilities)
            }))

        return SyntheticDataset(dsg)

    return [make_perceive_with_noise_datasets(pct / 100) for pct in range(2, 42, 4)]


def make_discrete_dataset_1(num_items_per_dataset=50):
    item_state_generator = \
        DiscreteDistributionOverStates(states=[DiscreteState(state_name='pos',
                                                      labels=['pos', 'neg'],
                                                      probabilities=[.9, .1]),
                                        DiscreteState(state_name='neg',
                                                      labels=['pos', 'neg'],
                                                      probabilities=[.25, .75])
                                        ],
                                probabilities=[.8, .2]
                                )

    dsg = SyntheticBinaryDatasetGenerator(item_state_generator=item_state_generator,
                                          pct_noise=.1,
                                          name='dataset1_80exprts_90-10onhigh_25-75onlow_10noise',
                                          num_items_per_dataset=num_items_per_dataset
                                          )

    # dsg.mock_classifiers.append(MockClassifier(
    #     name='.95 .2',
    #     label_predictors={
    #         'pos': DiscreteDistributionPrediction(['pos', 'neg'], [.95, .05]),
    #         'neg': DiscreteDistributionPrediction(['pos', 'neg'], [.2, .8])
    #     }))
    #
    # dsg.mock_classifiers.append(MockClassifier(
    #     name='.92 .24',
    #     label_predictors={
    #         'pos': DiscreteDistributionPrediction(['pos', 'neg'], [.92, .08]),
    #         'neg': DiscreteDistributionPrediction(['pos', 'neg'], [.24, .76])
    #     }))

    dsg.mock_classifiers.append(MockClassifier(
        name='h_infinity: ideal classifier',
        label_predictors={
            'pos': DiscreteDistributionPrediction(['pos', 'neg'], [.9, .1]),
            'neg': DiscreteDistributionPrediction(['pos', 'neg'], [.25, .75])
        }))

    return SyntheticDataset(dsg)


def make_discrete_dataset_2():
    item_state_generator = \
        DiscreteDistributionOverStates(states=[DiscreteState(state_name='pos',
                                                      labels=['pos', 'neg'],
                                                      probabilities=[.5, .5]),
                                        DiscreteState(state_name='neg',
                                                      labels=['pos', 'neg'],
                                                      probabilities=[.3, .7])
                                        ],
                                probabilities=[.5, .5]
                                )

    dsg = SyntheticBinaryDatasetGenerator(item_state_generator=item_state_generator)
    return SyntheticDataset(dsg)


def make_discrete_dataset_3():
    item_state_generator = \
        DiscreteDistributionOverStates(states=[DiscreteState(state_name='pos',
                                                      labels=['pos', 'neg'],
                                                      probabilities=[.4, .6]),
                                        DiscreteState(state_name='neg',
                                                      labels=['pos', 'neg'],
                                                      probabilities=[.7, .3])
                                        ],
                                probabilities=[.4, .6]
                                )

    dsg = SyntheticBinaryDatasetGenerator(item_state_generator=item_state_generator)
    return SyntheticDataset(dsg)

