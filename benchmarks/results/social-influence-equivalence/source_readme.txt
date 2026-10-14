#######################

DATASET

The file "becker_brackbill_centola_supporting_data.csv" contains the full data for replicating the analysis in:

Becker, J., Brackbill, D., & Centola, D. (2017). Network dynamics of social influence in the wisdom of crowds. Proceedings of the National Academy of Sciences, 201615978.

#######################

MISSING DATA

There are two types of missing data.  When a user had a connection open to the server, but failed to enter a response, their response is indicated as NA.  When a user lost their connection with the server (either due to attrition or technical problems) there is no entry.

#######################

COLUMNS

group_number - a unique identifier for a given set of individuals.  no individual is a member of more than one group.

network - either centralized, decentralized, or solo.  the "solo" network corresponds to the control condition

task - corresponds to a particular image prompt and question.  task details are indicated in the Appendix for Becker, Brackbill, and Centola (2017).

subject_id - a unique identifier for each experimental subject

response_1, response_2, response_3 - the users response to the estimation task for round 1, 2, or 3

truth - the true value for the estimation task

sd_pool - the pooled standard deviation of ALL independent (round 1) responses to a question

mean.neighbor.time1 - the average response of that user's network neighbors for that task at round 1

mean.neighbor.time2 - the average response of that user's network neighbors for that task at round 2
is_central - whether the user is a central node.  only valid for centralized networks


