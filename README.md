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
