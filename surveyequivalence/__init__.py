from .generate_labels import State, DiscreteState, DistributionOverStates, DiscreteDistributionOverStates, DiscreteLabelsWithNoise, FixedStateGenerator, MixtureOfBetas
from .combiners import Combiner, Prediction, DiscreteDistributionPrediction, PluralityVote, FrequencyCombiner, \
    AnonymousBayesianCombiner, MeanCombiner, NumericPrediction
from .scoring_functions import AgreementScore, PrecisionScore, RecallScore, F1Score, AUCScore, CrossEntropyScore, Correlation
from .equivalence import AnalysisPipeline, Plot, ClassifierResults
from .synthetic_datasets import make_discrete_dataset_1, make_discrete_dataset_2, make_discrete_dataset_3, make_running_example_dataset, MockClassifier, make_perceive_with_noise_datasets

