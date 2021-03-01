API
===

Equivalence Module
------------------

AnalysisPipeline
^^^^^^^^^^^^^^^^
.. autoclass:: surveyequivalence.equivalence.AnalysisPipeline
   :members: output_csv, run, save, path_for_saving
   :show-inheritance:

.. autofunction:: surveyequivalence.equivalence.load_saved_pipeline

Plot
^^^^
.. autoclass:: surveyequivalence.equivalence.Plot
   :members:
   :show-inheritance:

Equivalences
^^^^^^^^^^^^
.. autoclass:: surveyequivalence.equivalence.Equivalences
   :members:
   :show-inheritance:

ClassifierResults
^^^^^^^^^^^^^^^^^
.. autoclass:: surveyequivalence.equivalence.ClassifierResults
   :members:
   :show-inheritance:

PowerCurve
^^^^^^^^^^
.. autoclass:: surveyequivalence.equivalence.PowerCurve
   :members: compute_equivalences, compute_equivalence_at_mean, compute_equivalence_at_actuals, reliability_of_difference, reliability_of_beating_classifier
   :show-inheritance:

Combiners
---------

.. automodule:: surveyequivalence.combiners
   :members:
   :undoc-members:
   :show-inheritance:

Scoring Functions
-----------------

.. automodule:: surveyequivalence.scoring_functions
   :members:
   :undoc-members:
   :show-inheritance:

Synthetic Dataset Generation
----------------------------

States
^^^^^^
.. autoclass:: surveyequivalence.synthetic_datasets.DiscreteState
   :members: draw_labels
   :show-inheritance:


Distributions Over States
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: surveyequivalence.synthetic_datasets.DistributionOverStates
   :members:
   :show-inheritance:

.. autoclass:: surveyequivalence.synthetic_datasets.DiscreteDistributionOverStates
   :members:
   :show-inheritance:

.. autoclass:: surveyequivalence.synthetic_datasets.FixedStateGenerator
   :members:
   :show-inheritance:

Mock Classifiers
^^^^^^^^^^^^^^^^

.. autoclass:: surveyequivalence.synthetic_datasets.MockClassifier
   :members:
   :show-inheritance:

.. autoclass:: surveyequivalence.synthetic_datasets.MappedDiscreteMockClassifier
   :members:
   :show-inheritance:

Dataset Generators
^^^^^^^^^^^^^^^^^^

.. autoclass:: surveyequivalence.synthetic_datasets.SyntheticDatasetGenerator
   :members:
   :show-inheritance:

.. autoclass:: surveyequivalence.synthetic_datasets.SyntheticBinaryDatasetGenerator
   :members:
   :show-inheritance:

Dataset
^^^^^^^
.. autoclass:: surveyequivalence.synthetic_datasets.Dataset
   :members:
   :show-inheritance:

.. autoclass:: surveyequivalence.synthetic_datasets.SyntheticDataset
   :members:
   :show-inheritance:

.. autofunction:: surveyequivalence.synthetic_datasets.make_running_example_dataset

