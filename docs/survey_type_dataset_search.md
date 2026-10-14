# Individual-response data for comparing survey types

Checked September 23, 2026. People are the raters; question types define different
surveys. This follows the professor's clarification and is separate from the
earlier AI-judge dataset search and superseded questions-as-raters illustration.

**Closest substantive match: LISS election surveys. Immediately usable open
pilot: Honda et al.'s numerical-estimation experiments.** Honda changes the
response format. LISS changes what the respondent is asked about.

| Dataset | Individual responses and survey types | Outcome/reference | Access verified |
| --- | --- | --- | --- |
| Honda et al. (2022), Experiment 1 | 901 participant rows; nine shared dot-count questions; typed number plus six scale formats | Correct counts supplied in the supplement | Both experiment workbooks downloaded and inspected |
| LISS Tilburg Election Study 2017, week 3 | Archive reports 3,672 respondents; own-vote probabilities and social-circle vote shares; encrypted person ID | Official election vote shares, to be joined | Public variables/codebook verified; microdata download redirects to login |
| USC Understanding America Study, 2016 election/social-circle study | Individual own intentions, social-circle expectations and winner expectations; longitudinal person IDs | Election outcomes at the chosen geographic level, to be joined | Public documentation verified; individual files require registration/login |

Sources: [Honda study](https://www.nature.com/articles/s41598-022-11900-7),
[Honda data](https://osf.io/sxdf9/),
[LISS archive](https://www.dataarchive.lissdata.nl/study-units/view/855),
[UAS election data](https://uasdata.usc.edu/index.php?pid=UAS+2016+Presidential+Election&type=3),
[UAS social-circle study](https://eprints.whiterose.ac.uk/id/eprint/126237/1/SocialCircles_ElectionPolls_inpress.pdf).

## Open pilot: verified files

Original workbooks are in `.cache/survey-type-data/nonpolitical/`. Both SHA-256
hashes match the authors' OSF metadata. The
[inspection record](../benchmarks/results/survey-type-data-search/inspection.json)
records URLs, hashes, sheets, ranges and completeness checks.

Experiment 1 has one person per row, nine estimate columns, nine confidence
columns and a format-group column. Rows 4–904 contain **8,109 estimates without
missing values**: nine measurements per person, not 8,109 people.

| Released group | People | Response method |
| --- | ---: | --- |
| Number | 128 | Type a number |
| Scale 1 | 128 | Drag from zero; endpoints labeled |
| Scale 2 | 133 | Click a scale; endpoints labeled |
| Scale 3 | 126 | Drag from zero; endpoints and midpoint labeled |
| Scale 4 | 126 | Click a scale; endpoints and midpoint labeled |
| Scale 5 | 130 | Drag from zero; endpoints and quartiles labeled |
| Scale 6 | 130 | Click a scale; endpoints and quartiles labeled |

The supplement maps Q1–Q9 to correct counts of 183, 287, 360, 453, 554, 633, 719,
807 and 986. All estimates are within the specified 0–1,000 range.

Formats are between-person conditions. Each row preserves a person's answers
across questions. There are no explicit person IDs. The prepared long-form
JSONL uses file/sheet/row keys, which identify released records only and cannot
link people across experiments. Original workbooks were not modified.

Experiment 2 has **904 participant rows and 9,944 nonmissing estimate cells**
across seven formats. Its eleven response columns cover ten general-knowledge
prompts: minutes and seconds for one prompt occupy separate columns. Historical
reference values require further validation. Experiment 1 has simpler truth
mapping and common units, making it the better initial pilot.

Original files: [Experiment 1](https://osf.io/download/uxrhd/) and
[Experiment 2](https://osf.io/download/bjrkg/). Prepared data:
`.cache/survey-type-data/nonpolitical/honda_experiment1_long.jsonl`.

## Closest question-type comparison: LISS

The public week 3 archive identifies:

- `nomem_encr`: encrypted respondent ID.
- `nf17c004`–`nf17c017`: own-vote probabilities across parties.
- `nf17c066`–`nf17c079`: expected social-circle vote shares across parties.
- `nf17c002` and `nf17c064`: own and social-circle turnout expectations.

This supports asking how many own-intention respondents match a specified
social-circle panel when estimating election vote shares. Aggregation must
handle the response definitions and turnout consistently.

The published analysis uses 2,230 valid paired respondents; the archive's 3,672
is its broader response count, not a measurement from downloaded microdata.
The [English codebook](https://www.dataarchive.lissdata.nl/hosted-files/download/5429)
was downloaded. The [Stata file](https://www.dataarchive.lissdata.nl/hosted-files/download/4289)
redirects to login. No registration or author contact was performed.

The related Dutch/Swedish paper's
[OSF metadata](https://api.osf.io/v2/preprints/8g5ce/) specifies registration for
Dutch data and author request for Swedish data. These remain documented
candidates rather than downloaded individual records.

## Proposed first visualization and limits

For Honda Experiment 1, sample k distinct participant rows within each format,
average their estimates per image and compare with the published counts. Plot
mean absolute error against k. Preserve all nine responses when sampling a
person. Fix aggregation and scoring before comparing curves, and report an
equivalence only where the comparison is supported within the observed range.

This evaluates elicitation formats on nine estimation problems. It does not
establish future-event forecasting performance or general equivalence across
question content. Nine images provide limited item diversity. Similarly, LISS
survey weeks and party categories do not create independent elections.
Resampling people cannot add independent outcome cases.

No empirical survey-type equivalence was calculated during the search itself.
The subsequent [Honda analysis](honda_survey_format_equivalence.md) now provides
real-response figures and panel-size equivalences using RMS error, with the
scoring choice and uncertainty documented separately.
