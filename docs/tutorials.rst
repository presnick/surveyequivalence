Tutorials
=========

Synthetic Running Example
-------------------------

The synthetic running example, as described in the paper, is implemented in the file `synthetic_running_example.py`.

We first use the function :func:`surveyequivalence.synthetic_running_example.make_running_example_dataset`.

.. code-block:: python3

    num_items_per_dataset=1000
    num_labels_per_item=10
    num_bootstrap_item_samples = 500

    ds2 = make_running_example_dataset(minimal=False, num_items_per_dataset=num_items_per_dataset,
                                       num_labels_per_item=num_labels_per_item,
                                       include_soft_classifier=True, include_hard_classifier=True)

    hard_classifiers = ['mock hard classifier']
    soft_classifiers = ['calibrated hard classifier', 'h_infinity: ideal classifier']

The resulting SyntheticDataset object has an attribute `.classifier_predictions`, which is a dataframe with one column
each for several classifiers.

    -   'mock hard classifier': a mock classifier that outputs 90/10 pos labels for high state, 50/50 for med,
        and 05/95 for low. This classifier is more informative than a single reference rater,
        whose labels are generated 80/20, 50/50, and 10/90.
    -   'calibrated hard classifier': a mock classifier that converts the hard classifier outputs to their correct
        calibrated soft predictions (probability that the next reference rater will have a positive label).
    -   'h_infinity: ideal classifier': a mock classifier that correctly predicts 80/20, 50/50 or 10/90 for every item,
        magically knowing the item's true state. No classifier can achieve higher (expected)
        cross-entropy score than this classifier.


Run the analysis pipeline for PluralityVote plus AgreementScore
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Next, we run the analysis pipeline and generate plots. We will do this three times in all, with three different
pairings of combiner function with scoring function. We start with a scoring function that expects discrete label
predictions (i.e., hard classifiers). That requires a combiner function that produces a single predicted label from
a set of other labels.

We make an instance of the class :class:`surveyequivalence.surveyequivalence.equivalence.AnalysisPipeline`. We pass:

    -   the `.dataset` as the rating matrix W.
    -   the list of all column names as the reference rater's column names
    -   the plurality_combiner; with binary labels, it just selects the label that was more popular for this item
    -   for the scorer, agreement_score returns the percentage of items for which the predicted label matches the actual.

.. code-block:: python3

    plurality_combiner = PluralityVote(allowable_labels=['pos', 'neg'])
    agreement_score = AgreementScore()
    pipeline = AnalysisPipeline(ds2.dataset,
                                expert_cols=list(ds2.dataset.columns),
                                classifier_predictions=ds2.classifier_predictions[hard_classifiers],
                                combiner=plurality_combiner,
                                scorer=agreement_score,
                                allowable_labels=['pos', 'neg'],
                                num_bootstrap_item_samples=num_bootstrap_item_samples,
                                verbosity = 1)
    pipeline.save(dirname_base = "plurality_plus_agreement",
        msg = f"""
    Running example with {num_items_per_dataset} items and {num_labels_per_item} raters per item
    {num_bootstrap_item_samples} bootstrap itemsets
    Plurality combiner with agreement score
    """)



Plot the results
^^^^^^^^^^^^^^^^

Next we create an instance of the class :class:`surveyequivalence.surveyequivalence.equivalence.Plot`, passing:

    -   the power curve that we calculated in the AnalysisPipeline
    -   classifier scoeres calculated in the AnalysisPipeline
    -   The color_map says what colors to use for the different components of the plot. In this case, we don't have
        an amateur_power_curve, but we have included it to illustrate how to supply a color for it if we did have
        that additional power curve for other raters.


.. code-block:: python3

    fig, ax = plt.subplots()
    fig.set_size_inches(8.5, 10.5)

    color_map = {
        'expert_power_curve': 'black',
        'amateur_power_curve': 'green',
        'hard classifier': 'red',
        'mock classifier': 'blue',
        'calibrated hard classifier': 'red'
    }

    pl = Plot(ax,
              pipeline.expert_power_curve,
              classifier_scores=pipeline.classifier_scores,
              color_map=color_map,
              y_axis_label='percent agreement with reference rater',
              y_range=(0, 1),
              name='running example: majority vote + agreement score',
              legend_label='k raters',
              generate_pgf=True
              )


Then, we call the method :meth:`surveyequivalence.surveyequivalence.equivalence.Plot.plot` to actually create the plot.

.. code-block:: python3

    pl.plot(include_classifiers=True,
            include_classifier_equivalences=True,
            include_droplines=True,
            include_expert_points='all',
            connect_expert_points=True,
            include_classifier_cis=True ##change back to false
            )

Finally, we save the plot. This saves both a PDF version and, since we specified that we wanted it,
a pgf file suitable for importing into latex.

.. code-block:: python3

    pgf = None
    if pl.generate_pgf:
        pgf = pl.template.substitute(**pl.template_dict)
    save_plot(fig, 'runningexample_majority_vote_plus_agreement', pgf)

AnonymousBayesianCombiner plus CrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next we consider a scorer for soft classifiers, which predict a probability for each possible label, rather than
outputting a single label. The Anonymous Bayesian Combiner, as described in the paper, is one such combiner.
Essentially, it estimates the probability of a pos or neg next label conditional on having observed the labels
that have been seen so far.

The following code computes this baseline score.

.. code-block:: python3

    print(f"""mean label counts to use as prior for ABC: {ds2.dataset.apply(
            pd.Series.value_counts, normalize=True, axis=1).fillna(0).mean(axis=0)}""")
    base_pred = abc.combine(['pos', 'neg'], [], W=ds2.dataset.to_numpy(), item_id=1)
    predictions = [base_pred for _ in range(len(ds2.dataset))]
    c_0 = cross_entropy.score_classifier(predictions, ds2.dataset.columns, ds2.dataset)
    print(f"Cross Entropy on base_preds (i.e., c_0) = {c_0}")

The analysis code is similar to that for the previous combiner and scorer.

.. code-block:: python3

    abc = AnonymousBayesianCombiner(allowable_labels=['pos', 'neg'])
    cross_entropy = CrossEntropyScore()
    pipeline2 = AnalysisPipeline(ds2.dataset,
                                expert_cols=list(ds2.dataset.columns),
                                classifier_predictions=ds2.classifier_predictions[soft_classifiers],
                                combiner=abc,
                                scorer=cross_entropy,
                                allowable_labels=['pos', 'neg'],
                                num_bootstrap_item_samples=num_bootstrap_item_samples,
                                verbosity = 1)

    pipeline2.save(dirname_base = "abc_plus_cross_entropy",
                   msg = f"""
    Running example with {num_items_per_dataset} items and {num_labels_per_item} raters per item
    {num_bootstrap_item_samples} bootstrap itemsets
    Anonymous Bayesian combiner with cross entropy score
    """)

The plotting is similar with a couple twists.

Here we specify centering of y-axis values on the score for a survey of k=0 people. With the cross entropy scoring
function these centered values have a natural interpretation, as explained in the paper. The cross entropy of a
baseline classifier that predicts the overall empirical frequency of the labels (i.e., Anonymous Bayesian Combiner
with k=0) against
a reference rater's labels will approach the
entropy of the distribution as the number of items grows. Thus,
the cross-entropy of any other classifier minus this score estimates the
information gain of the classifier (mutual information of the classifier with a random reference rater's predictions).

Note that we are choosing to plot only the calibrated hard classifier, and not the ideal classifier. In the pipeline
we calculated results for two soft classifiers. If we want
to plot horizontal lines for only a subset of the classifiers, we need to make a new instance of ClassifierResults
using a subset
of the columns from the dataframe in the `.classifier_scores` object.


.. code-block:: python3

    fig, ax = plt.subplots()
    fig.set_size_inches(8.5, 10.5)

    pl = Plot(ax,
              pipeline2.expert_power_curve,
              classifier_scores=ClassifierResults(pipeline2.classifier_scores.df[['calibrated hard classifier']]),
              color_map=color_map,
              y_axis_label='information gain ($c_k - c_0$)',
              center_on=pipeline2.expert_power_curve.values[0],
              y_range=(0, 0.4),
              name='running example: ABC + cross entropy',
              legend_label='k raters',
              generate_pgf=True
              )

    pl.plot(include_classifiers=True,
            include_classifier_equivalences=True,
            include_droplines=True,
            include_expert_points='all',
            connect_expert_points=True,
            include_classifier_cis=True ##change back to false
            )
    # pl.add_state_distribution_inset(ds.ds_generator)
    pgf = None
    if pl.generate_pgf:
        pgf = pl.template.substitute(**pl.template_dict)
    save_plot(fig, 'runningexampleABC+cross_entropy', pgf)



FrequencyCombiner plus CrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code for the last combiner and scorer is very similar and is omitted.

Where to Find the Results
^^^^^^^^^^^^^^^^^^^^^^^^^

In config.py, you will specify a ROOTDIR.

Directory f'{ROOT_DIR}/plots' is where you'll find the plots, as .png and .tex files.

Directory f'{ROOT_DIR}/saved_analyses is where you'll find text summaries and .csv files produced by calls to
:meth:`surveyequivalence.surveyequivalence.equivalence.AnalysisPipeline.save`

Toxicity
--------