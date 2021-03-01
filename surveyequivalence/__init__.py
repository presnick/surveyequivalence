from .combiners import Combiner, Prediction, DiscretePrediction, DiscreteDistributionPrediction, PluralityVote, FrequencyCombiner, \
    AnonymousBayesianCombiner, MeanCombiner, NumericPrediction
from .scoring_functions import AgreementScore, PrecisionScore, RecallScore, F1Score, AUCScore, CrossEntropyScore, Correlation, Scorer
from .equivalence import AnalysisPipeline, Plot, ClassifierResults, load_saved_pipeline
from .synthetic_datasets import State, DiscreteState, DistributionOverStates, DiscreteDistributionOverStates, \
    FixedStateGenerator, MixtureOfBetas, make_discrete_dataset_1, make_discrete_dataset_2, make_discrete_dataset_3, \
    MockClassifier, make_perceive_with_noise_datasets, SyntheticDataset, SyntheticBinaryDatasetGenerator, \
    MappedDiscreteMockClassifier
from .examples import toxicity, credbank, paper_running_example, guessthekarma