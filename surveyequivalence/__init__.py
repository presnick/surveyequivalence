from .generate_labels import State, DiscreteState, DistributionOverStates, DiscreteLabelsWithNoise, MixtureOfBetas
from .combiners import Combiner, Prediction, DiscreteDistributionPrediction, FrequencyCombiner, AnonymousBayesianCombiner
from .scoring_functions import AgreementScore, PrecisionScore, RecallScore, F1Score, AUCScore, CrossEntropyScore
from .equivalence import AnalysisPipeline, Plot
from .synthetic_datasets import make_discrete_dataset_1, make_discrete_dataset_2, make_discrete_dataset_3, MockClassifier

