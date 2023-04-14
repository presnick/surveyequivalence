Getting Started
===============

Installation
------------

Requires Python version 3.8 or higher. We have tested this with VirtualEnvironment and Conda using Python 3.8 and 3.10

.. code-block:: console

    pip install surveyequivalence

At root level, you should find config.py and directories surveyequivalence, docs, data, etc.

Executing the Running Example
-----------------------------

The running example dataset has 1000 items. It takes a while to run it with 500 bootstrap item samples.
If you're just trying to verify that your installation is good, you may want to run it on a smaller set of items
with fewer bootstrap item samples.

.. code-block:: console

    % python
    Python 3.8.16 (default, Mar  1 2023, 21:18:45)
    [Clang 14.0.6 ] :: Anaconda, Inc. on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>> from surveyequivalence.examples.paper_running_example import main
    >>> import pkg_resources
    >>> DATA_PATH = pkg_resources.resource_filename('surveyequivalence', 'data')
    >>> main(path=f"{DATA_PATH}/running_example_50_items", num_bootstrap_item_samples=10)
    starting classifiers: computing scores

As described in the :ref:`tutorial <Tutorials>`, the running example for the paper computes three survey power curves, for three different
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
