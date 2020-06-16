import pandas as pd
import numpy as np
from surveyequivalence import AnalysisPipeline, DiscreteDistributionPrediction, FrequencyCombiner, F1Score, CrossEntropyScore, AnonymousBayesianCombiner

gtk = pd.read_csv('./data/vote_gtk2.csv')

rater_ids = {v:i for (i,v) in enumerate(set(gtk['user_id']))}
item_ids = {v:i for (i,v) in enumerate(set(gtk['image_pair']))}
predict_W = np.zeros((len(rater_ids),len(item_ids)*2), dtype=str)
prefer_W = np.zeros((len(rater_ids),len(item_ids)*2), dtype=str)

for index, rating in gtk.iterrows():
    # get the x and y in the W
    rater_id = rater_ids[rating['user_id']]
    item_id = item_ids[rating['image_pair']]

    # now get the preference
    rater_opinion = rating['opinion_choice']
    if rater_opinion == 'A':
        prefer_W[rater_id, item_id] = 'p'
    elif rater_opinion == 'B':
        prefer_W[rater_id, item_id] = 'n'
    else:
        pass
        # print(rater_opinion)

    rater_prediction = rating['prediction_choice']
    if rater_prediction == 'A':
        predict_W[rater_id, item_id] = 'p'
    elif rater_prediction == 'B':
        predict_W[rater_id, item_id] = 'n'
    else:
        pass
        # print(rater_prediction)

for index, rating in gtk.iterrows():
    # get the x and y in the W
    rater_id = rater_ids[rating['user_id']]
    item_id = item_ids[rating['image_pair']] + len(item_ids)

    # now get the preference
    rater_opinion = rating['opinion_choice']
    if rater_opinion == 'A':
        prefer_W[rater_id, item_id] = 'n'
    elif rater_opinion == 'B':
        prefer_W[rater_id, item_id] = 'p'
    else:
        pass
        # print(rater_opinion)

    rater_prediction = rating['prediction_choice']
    if rater_prediction == 'A':
        predict_W[rater_id, item_id] = 'n'
    elif rater_prediction == 'B':
        predict_W[rater_id, item_id] = 'p'
    else:
        pass
        # print(rater_prediction)



# prefer might have empty rows because GTK didn't always ask the preference question.
mask = np.all(prefer_W == '', axis=1)
prefer_W = prefer_W[~mask]

prefer_W = pd.DataFrame(data=prefer_W)

print('##Dataset loaded##')

p = AnalysisPipeline(prefer_W, AnonymousBayesianCombiner(), CrossEntropyScore.score, allowable_labels=['p', 'n'],
                     null_prediction=DiscreteDistributionPrediction(['p', 'n'], [.5, .5]),
                     num_runs=1)
results = pd.concat([p.power_curve.means, p.power_curve.cis], axis=1)
results.columns = ['mean', 'ci_width']
print (results)
for i in range(15):
    thresh = .65 + .01 * i
    print(f"\tsurvey equivalence for {thresh} is ", p.power_curve.compute_equivalence(thresh))