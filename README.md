# SurveyEquivalence
Author: Paul Resnick, Yuqing Kong, Grant Schoenebeck, Tim Weninger

**arxiv paper available**: https://arxiv.org/abs/TKTK

**Documentation available**: https://surveyequivalence.readthedocs.io 

## Overview
Given a dataset W of ratings for 

Installation
------------

.. code-block:: console

    pip install surveyequivalence

At root level, you should find config.py and directories surveyequivalence, docs, data, etc.

Executing the Running Example
-----------------------------

The running example dataset has 1000 items. It takes a while to run it with 500 bootstrap item samples.
If you're just trying to verify that your installation is good, you may want to run it on a smaller set of items
with fewer bootstrap item samples, as shown below.

.. code-block:: console

    (survey_equiv) surveyequivalence[master !?]$ python
    Python 3.7.4 (default, Aug 13 2019, 20:35:49)
    [GCC 7.3.0] :: Anaconda, Inc. on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> from surveyequivalence.examples.paper_running_example import main
    >>> main(path='data/running_example_50_items', num_bootstrap_item_samples=10)
    starting classifiers: computing scores

As described in the tutorial, the running example for the paper computes three survey power curves, for three different
combiner/scorer pairings.

If you have multiple processors, the AnalysisPipeline will try to take advantage of them to speed up execution.
That may cause the progress indicator output to show some things out of order, like what is shown here.
That's nothing to worry about.

.. code-block:: console

    starting power curve

    starting to precompute predictions for various rater subsets.
    Items processed:
    .....    10....  0.      20.................     30...   40...............

            computing power curve results for each bootstrap item sample.
    Samples processed:
    .        10.......       0.
    starting classifiers: computing scores

    starting power curve

    starting to precompute predictions for various rater subsets.
    Items processed:
    ..       10....  0..     20................      30..........    40...........

            computing power curve results for each bootstrap item sample.
    Samples processed:
    ....     0....   10.
    starting classifiers: computing scores

    starting power curve

    starting to precompute predictions for various rater subsets.
    Items processed:
    ..       0..     10..    20.................     30....  40..................

            computing power curve results for each bootstrap item sample.
    Samples processed:
    ....     0...    10..
    >>>

Locating the Results
--------------------

After executing the running example, look in the directory :code:`save_analyses`. There should be a subfolder with a
timestamp. Within that there are three subfolders, one for each pairing of a combiner with a scoring function.

Within each results folder, you should find:

    - README says what was analyzed.
    - results_summary.txt gives numeric summaries of the results
    - several .csv files provide detailed data about classifier scores and equivalences
    - plot.png, a survey power curve plot with equivalence points
    - plot.tex; pgf formatted text that will generate the same plot within latex.


## License
This Project is MIT-licensed.

Please send us e-mail or leave comments on github if have any questions.

Copyright (c) 2021, Paul Resnick, Yuqing Kong, Grant Schoenebeck, Tim Weninger 






# surveyequivalence

## make_power_curve_graph:
 This function receives the following:

#### Expert_scores
One dictionary defining the power curve:
- Name: a string
- Color: a color to use (if None, assign automatically)
- Show_lines: True or False
- Power_curve: a sequence, with each item in the sequence a dictionary
    - k, the number of amateurs in the survey (this will be x-value on the graph)
    - Score, a float
        - the expected score of k amateurs against a random expert (this will be y-value on the graph)
    - Confidence_radius: a float or None
        - If value is not None, it defines the confidence interval around the score
          - Score +/- confidence_radius

#### Amateur_scores, a sequence of power curves
Each item in the sequence is a dictionary:
- Name: a string
- Color: a color to use (if None, assign automatically)
- Show_lines: True or False
- Power_curve: a sequence, with each item in the sequence a dictionary
    - k, the number of amateurs in the survey (this will be x-value on the graph)
    - Score, a float
        - the expected score of k amateurs against a random expert (this will be y-value on the graph)
    - Confidence_radius: a float or None
        - If value is not None, it defines the confidence interval around the score
        - Score +/- confidence_radius

#### Classifier_scores:
A dataframe, with columns
- Name: the name for the classifier
- Color: a color to use (if None, assign automatically)
- Score: a float
    - the expected score of the classifier against a random expert (this will be y-value on the graph)
- Confidence_radius: a float or None
    - If value is not None, it defines the confidence interval around the score
    - Score +/- confidence_radius

#### Points_to_show_SurveyEquiv
Each item in the sequence is a dictionary
- Which_type: “amateur” or “classifier”
- Which_one: an integer
- Color: a color to use (if None, assign automatically)
