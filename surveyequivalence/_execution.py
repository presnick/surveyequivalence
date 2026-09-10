"""Compact prediction blocks and worker-local state for bootstrap execution."""

import copy
import os
import pickle

import numpy as np
import pandas as pd

from .combiners import (
    AnonymousBayesianCombiner, FrequencyCombiner, PluralityVote, MeanCombiner,
    DiscreteDistributionPrediction, NumericPrediction,
)
from ._random import random_stream
from ._scoring import score_prepared


BUILTIN_COMBINERS = (AnonymousBayesianCombiner, FrequencyCombiner, PluralityVote, MeanCombiner)


class PredictionBlock:
    """Arrays for one ordered block of subsets; objects for custom combiners."""

    def __init__(self, directory, subsets, n_items, labels, combiner, n_refs,
                 disk=False, prepared_scores=False, anonymous=False, force_objects=False):
        self.subsets = subsets
        self.labels = list(labels) if labels is not None else []
        self.kind = ('soft' if type(combiner) in (AnonymousBayesianCombiner, FrequencyCombiner)
                     else 'label' if type(combiner) is PluralityVote
                     else 'numeric' if type(combiner) is MeanCombiner else 'object')
        if force_objects:
            self.kind = 'object'
        self.n_refs = n_refs
        self.directory = directory
        self.paths = {}
        self.score_modes = [None] * len(subsets)
        self.ref_columns = []
        self.objects = ([[None] * n_items for _ in subsets] if self.kind == 'object' else None)
        width = len(self.labels) if self.kind == 'soft' else 1
        pred_shape = (len(subsets), n_items, width) if self.kind != 'object' else (0, 0, 0)
        self.values = self._array('predictions', pred_shape, np.float64, disk)
        valid_shape = (len(subsets), n_items) if self.kind != 'object' else (0, 0)
        self.valid = self._array('prediction_valid', valid_shape, bool, disk)
        score_width = 1 if anonymous else n_refs
        self.scores = (self._array('scores', (len(subsets), n_items, score_width), np.float64, disk)
                       if prepared_scores else None)
        self.score_valid = (self._array('score_valid', (len(subsets), n_items, score_width), bool, disk)
                            if prepared_scores else None)
        self.lookup = {label: i for i, label in enumerate(self.labels)}

    def _array(self, name, shape, dtype, disk):
        if disk:
            path = os.path.join(self.directory, name + '.npy')
            self.paths[name] = path
            array = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
            array[:] = 0
            return array
        return np.zeros(shape, dtype=dtype)

    def put(self, subset, item, prediction):
        if self.kind == 'object':
            self.objects[subset][item] = prediction
            return
        if prediction is None:
            return
        self.valid[subset, item] = True
        if self.kind == 'soft':
            self.values[subset, item] = prediction.probabilities
        elif self.kind == 'label':
            self.values[subset, item, 0] = self.lookup[prediction.value]
        elif self.kind == 'numeric':
            self.values[subset, item, 0] = prediction.value

    def predictions(self, subset, items):
        if self.kind == 'object':
            return [self.objects[subset][item] for item in items]
        result = []
        for item in items:
            if not self.valid[subset, item]:
                result.append(None)
            elif self.kind == 'soft':
                # Values are already clipped and normalized. Re-running the
                # constructor would introduce a second normalization/rounding.
                prediction = DiscreteDistributionPrediction.__new__(DiscreteDistributionPrediction)
                prediction.label_names = self.labels
                prediction.probabilities = self.values[subset, item].tolist()
                result.append(prediction)
            elif self.kind == 'label':
                result.append(NumericPrediction(self.labels[int(self.values[subset, item, 0])]))
            else:
                result.append(NumericPrediction(float(self.values[subset, item, 0])))
        return result

    def seal(self):
        for array in (self.values, self.valid, self.scores, self.score_valid):
            if array is not None:
                if isinstance(array, np.memmap):
                    array.flush()
                array.flags.writeable = False

    def descriptor(self):
        if self.objects is not None:
            with open(os.path.join(self.directory, 'objects.pickle'), 'wb') as handle:
                pickle.dump(self.objects, handle)
        return {name: getattr(self, name) for name in
                ('subsets', 'labels', 'kind', 'n_refs', 'directory', 'paths', 'score_modes', 'ref_columns')}

    @classmethod
    def load(cls, descriptor):
        block = cls.__new__(cls)
        block.__dict__.update(descriptor)
        for attr, name in (('values', 'predictions'), ('valid', 'prediction_valid'),
                           ('scores', 'scores'), ('score_valid', 'score_valid')):
            path = block.paths.get(name)
            setattr(block, attr, np.load(path, mmap_mode='r') if path is not None else None)
        block.objects = None
        if block.kind == 'object':
            with open(os.path.join(block.directory, 'objects.pickle'), 'rb') as handle:
                block.objects = pickle.load(handle)
        return block

    def close(self):
        for attr in ('values', 'valid', 'scores', 'score_valid'):
            array = getattr(self, attr, None)
            if isinstance(array, np.memmap):
                array._mmap.close()


def score_block(state, block, run_id):
    indices = state['samples'][run_id]
    scorer = state['scorer']
    if state.get('custom_scorer', False) or type(scorer).__module__ != 'surveyequivalence.scoring_functions':
        # Stateful extensions historically received a fresh task copy.
        scorer = copy.deepcopy(scorer)
    reference = None
    scores = []
    for subset_id, (k, ordinal, raterset) in enumerate(block.subsets):
        mode = block.score_modes[subset_id]
        if mode is not None:
            width = len(block.ref_columns[subset_id])
            values = block.scores[subset_id, :, 0] if mode == 'anonymous' else block.scores[subset_id, :, :width]
            valid = block.score_valid[subset_id, :, 0] if mode == 'anonymous' else block.score_valid[subset_id, :, :width]
            score = score_prepared(values, valid, indices, mode)
        else:
            if reference is None:
                reference = state['W'].iloc[indices].reset_index(drop=True)
            predictions = block.predictions(subset_id, indices)
            with random_stream(state['seed'], 4, *state['curve_key'], run_id, k, ordinal):
                score = scorer.expected_score(pd.Series(predictions), block.ref_columns[subset_id], reference,
                                               anonymous=state['anonymous'], verbosity=state['verbosity'])
        scores.append(score)
    return scores


_worker_state = None
_worker_block = None


def initialize_worker(state):
    global _worker_state, _worker_block
    _worker_state = state
    _worker_block = None


def worker_score(job):
    global _worker_block
    descriptor, run_id = job
    if _worker_block is None or _worker_block.directory != descriptor['directory']:
        if _worker_block is not None:
            _worker_block.close()
        _worker_block = PredictionBlock.load(descriptor)
    return score_block(_worker_state, _worker_block, run_id)
