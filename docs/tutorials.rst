Tutorials
=========

Synthetic Running Example
-------------------------

The synthetic running example, as described in the paper, is implemented in the file `examples/paper_running_example.py`.

We first load the data that we created in :ref:`generating-data-for-running-example`

.. code-block:: python3

    path = f'{ROOT_DIR}/data/running_example'

    # read the reference rater labels from file
    W = pd.read_csv(f"{path}/ref_rater_labels.csv", index_col=0)

    # read the predictions from file
    def str2prediction_instance(s):
        # s will be in format "Prediction: [0.9, 0.1]" or "Prediction: neg"
        suffix = s.split(": ")[1]
        if suffix[0] == '[':
            pr_pos, pr_neg = suffix[1:-1].split(',')
            return DiscreteDistributionPrediction(['pos', 'neg'], [float(pr_pos), float(pr_neg)])
        else:
            return DiscretePrediction(suffix)
    classifier_predictions = pd.read_csv(f"{path}/predictions.csv", index_col=0).applymap(str2prediction_instance)

    hard_classifiers = classifier_predictions.columns[:1] # ['mock hard classifier']
    soft_classifiers = classifier_predictions.columns[1:] # ['calibrated hard classifier', 'h_infinity: ideal classifier']

Note that the .csv file has the synthetic mock classifier predictions as strings.
We have to turn them into appropriate instances of :class:`surveyequivalence.combiners.Prediction`.
You will have to do something similar to convert numeric predictions of a real classifier for a real dataset into
instances of an appropriate subclass of Prediction.

Also note that W (the reference rater's labels) and classifier_predictions are two dataframes that must have the same
index, with the corresponding rows providing information about the same items.

Run the analysis pipeline for PluralityVote plus AgreementScore
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Next, we run the analysis pipeline and generate plots. We will do this three times in all, with three different
pairings of combiner function with scoring function. We start with a scoring function that expects discrete label
predictions (i.e., hard classifiers). That requires a combiner function that produces a single predicted label from
a set of other labels.

We make an instance of the class :class:`surveyequivalence.equivalence.AnalysisPipeline`. We pass:

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
    pipeline.save(path = pipeline.path_for_saving("running_example/plurality_plus_agreement"),
        msg = f"""
    Running example with {num_items_per_dataset} items and {num_labels_per_item} raters per item
    {num_bootstrap_item_samples} bootstrap itemsets
    Plurality combiner with agreement score
    """)



Plot the results
^^^^^^^^^^^^^^^^

Next we create an instance of the class :class:`surveyequivalence.equivalence.Plot`, passing:

    -   the power curve that we calculated in the AnalysisPipeline
    -   classifier scores calculated in the AnalysisPipeline
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


Then, we call the method :meth:`surveyequivalence.equivalence.Plot.plot` to actually create the plot.

.. code-block:: python3

    pl.plot(include_classifiers=True,
            include_classifier_equivalences=True,
            include_droplines=True,
            include_expert_points='all',
            connect_expert_points=True,
            include_classifier_cis=True
            )

Finally, we save the plot, using :meth:`surveyequivalence.equivalence.Plot.save`. This saves both a PDF version and, since we specified that we wanted it,
a pgf file suitable for importing into latex.

.. code-block:: python3

    pl.save(pipeline.path_for_saving("running_example/plurality_plus_agreement"), fig=fig)


AnonymousBayesianCombiner plus CrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next we consider a scorer for soft classifiers, which predict a probability for each possible label, rather than
outputting a single label. The Anonymous Bayesian Combiner, as described in the paper, is one such combiner.
Essentially, it estimates the probability of a pos or neg next label conditional on having observed the labels
that have been seen so far.

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

    pipeline2.save(path=pipeline.path_for_saving("running_example/abc_plus_cross_entropy"),
                   msg = f"""
    Running example with {num_items_per_dataset} items and {num_labels_per_item} raters per item
    {num_bootstrap_item_samples} bootstrap itemsets
    Anonymous Bayesian combiner with cross entropy score
    """)

The plotting is similar, with a couple twists.

Here we specify centering of y-axis values, subtracting out the score for a survey of k=0 people.
With the cross entropy scoring
function these centered values have a natural interpretation, as explained in the paper. The cross entropy of a
baseline classifier that predicts the overall empirical frequency of the labels (i.e., Anonymous Bayesian Combiner
with k=0) against
a reference rater's labels will approach the
entropy of the distribution from which reference raters are drawn, as the number of items grows. Thus,
the cross-entropy of any other classifier minus this score estimates the
information gain of the classifier (mutual information of the classifier with a random reference rater's predictions).

Note that we are choosing to plot only the calibrated hard classifier, and not the ideal classifier. In the pipeline
we calculated results for two soft classifiers. Because here we choose
to plot a horizontal line for only one of those two classifiers, we need to make a new instance of ClassifierResults
passing in only that column from the dataframe in the `.classifier_scores` object.

You may find it instructive to change the code to :code:`classifier_scores=pipeline2.classifier_scores`, and notice that the
resulting graph adds an extra horizontal line for the ideal classifier.

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
    pl.save(path=pipeline.path_for_saving("running_example/abc_plus_cross_entropy"), fig=fig)



FrequencyCombiner plus CrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code for the last combiner and scorer is very similar and is omitted.

Where to Find the Results
^^^^^^^^^^^^^^^^^^^^^^^^^

In config.py, you will specify a ROOTDIR.

Directory f'{ROOT_DIR}/saved_analyses' will have a folder named with a timestamp for the start of your AnalysisPipeline
run. Look inside that to find three subdirectories, one for each combiner+scorer pairing.

.. _generating-data-for-running-example:

Generating Data for the Running Example
---------------------------------------

The dataset use in the running example is synthetic. We generated it using the function :func:`surveyequivalence.synthetic_datasets.make_running_example_dataset`.

.. code-block:: python3

    num_items_per_dataset=1000
    num_labels_per_item=10
    num_bootstrap_item_samples = 500

    ds = make_running_example_dataset(minimal=False, num_items_per_dataset=num_items_per_dataset,
                                       num_labels_per_item=num_labels_per_item,
                                       include_soft_classifier=True, include_hard_classifier=True)

    ds.save(dirname='running_example')

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

Two .csv files are generated, predictions.csv and ref_rater_labels.csv. They are stored in a subdirectory of
data/running_example.

Jigsaw Personal Attacks Dataset Analysis
----------------------------------------

Calculating the survey equivalence of an real world item and rater set is easy with this package. Here we focus on the
Jigsaw Toxcitiy Dataset. This dataset is originally discussed in this paper:

Wulczyn, E., Thain, N., & Dixon, L. (2017, April). Ex machina: Personal attacks seen at scale. In Proceedings of the
26th international conference on world wide web (pp. 1391-1399).

The dataset can be found in `data/wiki_attack_labels_and_predictor.csv`. It contains raters labels of whether or not
some comment on Wikipedia is a personal attack. The header and first row of the dataset are:

`rev_id,perc_labelled_attack,n_labelled_attack,n_labels,predictor_prob`
`155243,0.222222222,2,9,0.037257579`

where the columns represent the Wikipedia comment ID, the percentage of labels that indicated the the comment was an
attack, the number of labeled attacks, the number of total labels -- where the percentage is equal to the number of
attacks divided by the number of labels, and the probability that the Jigsaw predictor returned.

We load and perform surveyequivalence analysis in `examples/toxicity.py`

Example Driver
^^^^^^^^^^^^^^

The main function servers as a driver for four combinations of scoring and combiner functions. AnonymousBayesianCombiner
with CrossEntropy, which has several desirable properties as discussed in the paper. Combinations of FrequencyCombiner
and AUCScore are also performed.

The first four lines of the main method are very important for the execution of the analysis. The `max_k` parameter
limits the number of raters to consider. The `max_items` parameter truncates the dataset -- large datasets take a long
time to run; full experiments must carefully weight limiting the dataset. The `bootstrap_samples` parameter indicates
how many times to sample the surveyequivelance to generate confidence intervals. The `num_processors` indicates how many
processors to use for computing.

In the reported experimental results, we set `max_k` = 10, `max_items` = 2000, `bootstrap_samples` = 500, and we had a
20 core compute server available to us. With these parameters, the full dataset took about 12 hours to compute each
combiner/score pair (two days for the whole driver to complete).

.. code-block:: python3

    max_k = 10
    max_items = 20
    bootstrap_samples = 2
    num_processors = 3

    # Next we iterate over various combinations of combiner and scoring functions.
    combiner = AnonymousBayesianCombiner(allowable_labels=['a', 'n'])
    scorer = CrossEntropyScore()
    run(combiner=combiner, scorer=scorer, max_k=max_k, max_items=max_items, bootstrap_samples=bootstrap_samples,
        num_processors=num_processors)

    combiner = FrequencyCombiner(allowable_labels=['a', 'n'])
    scorer = CrossEntropyScore()
    run(combiner=combiner, scorer=scorer, max_k=max_k, max_items=max_items, bootstrap_samples=bootstrap_samples,
        num_processors=num_processors)

    combiner = AnonymousBayesianCombiner(allowable_labels=['a', 'n'])
    scorer = AUCScore()
    run(combiner=combiner, scorer=scorer, max_k=max_k, max_items=max_items, bootstrap_samples=bootstrap_samples,
        num_processors=num_processors)

    combiner = FrequencyCombiner(allowable_labels=['a', 'n'])
    scorer = AUCScore()
    run(combiner=combiner, scorer=scorer, max_k=max_k, max_items=max_items, bootstrap_samples=bootstrap_samples,
        num_processors=num_processors)

Loading the Dataset
^^^^^^^^^^^^^^^^^^^

The first step is to load the dataset. Importantly, the surveyequivalence functions assume that the data exists in a
maxtrix form with n rows for each item (Wiki-comment in this case), and m columns for each rater. However, the dataset,
as exists, only provides counts. So it is important that we reverse-engineer each item and estimate what each rater
might have done. This is ok, because the rater ids (i.e, individual columns) are not important -- although this might
be something for future work.

.. code-block:: python3

    # Load the dataset as a pandas dataframe
    wiki = pd.read_csv(f'{ROOT_DIR}/data/wiki_attack_labels_and_predictor.csv')
    dataset = dict()

    # X and Y for calibration. These lists are matched
    X = list()
    y = list()

    # Create rating pairs from the dataset
    for index, item in wiki.iterrows():

        raters = list()

        n_raters = int(item['n_labels'])
        n_labelled_attack = int(item['n_labelled_attack'])

        for i in range(n_labelled_attack):
            raters.append('a')
            X.append([item['predictor_prob'], n_raters])
            y.append(1)
        for i in range(n_raters - n_labelled_attack):
            raters.append('n')
            X.append([item['predictor_prob'], n_raters])
            y.append(0)

        shuffle(raters)

        # This is the predictor i.e., score for toxic comment. It will be at index 0 in W.
        dataset[index] = [item['predictor_prob']] + raters

At this point the `dataset` variable will have one row for each item (i.e., Wiki-comment) and a shuffled listing of 'a'
and 'n' indicating attack or not-attack.

This dataset is not yet in matrix form. We need to convert what is essentially an adjacency list into an adjacency
matrix. To do this we find the max number of raters and set the number of columns to that number and pad the dataset
with Nones for items with less than the max number of raters.

.. code-block:: python3

    # Determine the number of columns needed in W. This is the max number of raters for an item.
    length = max(map(len, dataset.values()))

    # Pad W with Nones if the number of raters for some item is less than the max.
    padded_dataset = np.array([xi + [None] * (length - len(xi)) for xi in dataset.values()])

    print('##Wiki Toxic - Dataset loaded##', len(padded_dataset))

    # Trim the dataset to only the first max_items and recast W as a dataframe
    W = pd.DataFrame(data=padded_dataset)[:max_items]

    # Recall that index 0 was the classifier output, i.e., toxicity score. We relabel this to 'soft classifier' to keep
    # track of it.
    W = W.rename(columns={0: 'soft classifier'})

Here `W` is the item-rater matrix. We trim it to `max_items` to reduce the size of the dataset. There are very many
items, and it would be difficult to consider all of them in a tutorial.

Calibrating the Predictor
^^^^^^^^^^^^^^^^^^^^^^^^^

Next we are concerned with calibrating our classifier.

The Wiki-toxicity predictor was labeled `predictor_prob` in the dataset, and was loaded, for each rater, into `X`,
which is associated with a 1 or a 0 in `y` if the rater labeled attack or not attack respectively. The goal of this
predictor is to not necessarily predict attack or not attack, but rather to give a probability of the label. This
probability is a kind of confidence about the prediction.

Calibrating the predictor provides a way for the `predictor_prob` to be directly interprid as a confidence level. That
is, if `predictor_prob` is well calibrated then for Wiki-comments it gave an attack value of 0.2, then about 20% of the
items it labelled as attack are actually attacks.

We use sklearns CalibratedClassifierCV class and the isotonic regressor to fit a calibrator.

.. code-block:: python3

    # Calculate calibration probabilities. Use the current hour as random seed, because these lists need to be matched
    calibrator = CalibratedClassifierCV(LinearSVC(max_iter=1000), method='isotonic').fit(pd.DataFrame([x for x, y in X]), y,
                                                                       sample_weight=[1 / y for x, y in X])

Then we create our classifiers (predictors technically). For each item in W, we create a DiscreteDistributionPrediction
with attack and not-attack labels and 'a' and 'n' respectively. These are associated with the uncalibrated and
calibrated normalized Jigsaw predictor-probabilities.

.. code-block:: python3

    # Let's keep one classifier uncalibrated
    uncalibrated_classifier = pd.DataFrame(
        [DiscreteDistributionPrediction(['a', 'n'], [attack_prob, 1 - attack_prob], normalize=True)
         for attack_prob
         in W['soft classifier']], columns=['Uncalibrated Jigsaw Toxicity Classifier'])

    # Create a calibrated classifier
    calibrated_classifier1 = pd.DataFrame(
        [DiscreteDistributionPrediction(['a', 'n'], [a, b], normalize=True)
         for b, a
         in calibrator.predict_proba(W.loc[:, W.columns == 'soft classifier'])
         ], columns=['Calibrated Jigsaw Toxicity Classifier'])

    # The classifier object now holds the classifier predictions. Let's remove this data from W now.
    W = W.drop(['soft classifier'], axis=1)

    classifiers = uncalibrated_classifier.join(calibrated_classifier1, lsuffix='left', rsuffix='right')

The last line concatenates these classifiers together into a single object so they can be passed together into the
plot function later on.

Calculating a Prior Score
^^^^^^^^^^^^^^^^^^^^^^^^^

In certain cases, like with the CrossEntropyScore, we don't care about the raw values, but rather about the
information gain that is provided by more and more raters. To measure the gain we first need to create a baseline
(i.e., prior) score from which we can (hopefully) improve.

.. code-block:: python3

    # Here we create a prior score. This is the c_0, i.e., the baseline score from which we measure information gain
    # Information gain is only defined from cross entropy, so we only calculate this if the scorer is CrossEntropyScore
    if type(scorer) is CrossEntropyScore:
        # For the prior, we don't need any bootstrap samples and K needs to be only 1. Any improvement will be from k=2
        # k=3, etc.
        prior = AnalysisPipeline(W, combiner=AnonymousBayesianCombiner(allowable_labels=['a', 'n']), scorer=scorer,
                                 allowable_labels=['a', 'n'], num_bootstrap_item_samples=0, verbosity=1,
                                 classifier_predictions=classifiers, max_K=1, procs=num_processors)
    else:
        prior = None

The AnalysisPipeline
^^^^^^^^^^^^^^^^^^^^

Now that we have the calibrated and uncalibrated predictors, the prior (if needed), and our item-rater dataset matrix
`W`, we can begin the surveyequivalence analysis

.. code-block::python3

    p = AnalysisPipeline(W, combiner=combiner, scorer=scorer, allowable_labels=['a', 'n'],
                         num_bootstrap_item_samples=bootstrap_samples, verbosity=1,
                         classifier_predictions=classifiers, max_K=max_k, procs=num_processors)

The AnalysisPipline takes in `W`, the combiner, scorer, and labels, and classifiers. The number of bootstrap samples
indicates how many tests to perform to create confidence intervals around the survey power curve. `Max_k` indicates
how large to grow the power curve, i.e., how many raters to consider in the limit. The AnalysisPipline does run in
parallel, so you can set the number of CPU cores to use with the `num_processors` parameter.

From here the plotting is very similar to the SyntheticRunningExample.
>>>>>>> origin/weninger_dev

Guess the Karma Dataset Analysis
--------------------------------

CredBank Dataset Analysis
-------------------------

