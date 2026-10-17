# ML assignment 4: Ensemble learning for Classification

## About this repository
This repository stores the code, notebooks, and files used to develop ensemble
models for network intrusion detection. This was created as part of a university
assignment (Machine Learning 741 at Stellenbosch University).

## Assignment description
The main objective of this assignment is to explore ensemble learning as a means
of predicting cyberattacks.

Using the provided dataset, select an appropriate ensemble method and tune this
method accordingly. Ensure that the choice of ensemble method and the tuning
performed are well justified. Any necessary data transformations should also be
performed where well justified. Evaluate the tuned ensemble method on the
dataset and compare the results with non-ensemble models.

You may select any ensemble learning model, but approach this decision with
intention. What do you wish the final evaluation to reflect? Do you want to
compare ensemble methods to non-ensemble models, or do you want to compare
dissimilar methods? Neither option is correct, but a sound argument must be
supplied.

The tuning of your chosen ensemble method should also be performed with an aim
to answer specific questions that are set out before evaluation. Not only do you
want a model that performs well, but you also want to gain insight from the
tuning performance.

You report should include the necessary background, detail on your chosen
ensemble method(s) and the implementation thereof, detail on the empirical
process followed and then finally your results and a discussion of the results.
Remember that you must include all necessary background and implementation
details for the non-ensemble models that you compare against.

Report marking rubric:

| Aspect | Mark |
| --- | --- |
| Abstract             |   4 |
| Introduction         |   8 |
| Background           |  24 |
| Implementation       |   8 |
| Empirical process    |  18 |
| Results & discussion |  48 |
| Conclusions          |   4 |
| References           |   4 |
| Linguistic quality   |  32 |

Total: 150

## Stages to model building
### 1. Data gathering [x]
We were given a subset of the UNSW-NB15 Network Intrusion Detection dataset.

### 2. Data cleaning [ ]
#### Handle missing values [x]
I found that the `service` feature had missing values, and that there was a
strong correlation between `service`'s missingness and certain attacks:
```
=== Attack Categories vs Service Missingness ===
attack_cat         0      1      2      3      4      5      6      7      8      9
service_is_NaN                                                                 
False           31.3  15.78   4.72  15.34  48.24  21.07   9.35  85.06    0.0  98.45
True            68.7  84.22  95.28  84.66  51.76  78.93  90.65  14.94  100.0   1.55
```

So the feature was kept, and missing values were replaced with the string
`'other'`.

#### Remove features with unusual cardinality [x]
The `id` column was removed from the dataset

#### Remove duplicate features [x]
From past work, we learnt that the `is_ftp_login` column perfectly correlates
with `ct_ftp_cmd`, so `is_ftp_login` was dropped.

The following features have very strong correlations:
```
     feature_1        feature_2  spearman   pearson  strength
   trans_depth ct_flw_http_mthd  0.999563  0.251975  0.999563
        dbytes            dloss  0.947647  0.996711  0.996711
        tcprtt           synack  0.996267  0.943053  0.996267
        sbytes            sloss  0.830944  0.995772  0.995772
        tcprtt           ackdat  0.992658  0.920218  0.992658
         dpkts           dbytes  0.989424  0.973445  0.989424
        synack           ackdat  0.981722  0.737615  0.981722
          swin             dwin  0.980518  0.980458  0.980518
         dpkts            dloss  0.957333  0.979612  0.979612
    ct_srv_src       ct_srv_dst  0.916672  0.979467  0.979467
         spkts            sloss  0.929290  0.971859  0.971859
         spkts           sbytes  0.897137  0.964393  0.964393
         dload            dmean  0.963142  0.547562  0.963142
    ct_dst_ltm ct_src_dport_ltm  0.773512  0.961518  0.961518
        dbytes            dmean  0.960883  0.405260  0.960883
ct_dst_src_ltm       ct_srv_dst  0.839384  0.960321  0.960321
          rate            sload  0.959917  0.587503  0.959917
          rate           sinpkt -0.954854 -0.072414  0.954854
    ct_srv_src   ct_dst_src_ltm  0.816171  0.953952  0.953952
          dttl           dinpkt  0.953303  0.095347  0.953303
```

Two features that share a strong correlation might have very different levels of
informativeness w.r.t predicting the target feature, so I can't blindly drop
descriptive features.

I'll use all features, and if there is time run another experiment where
strongly correlated features are collapsed to one.

#### Handle imbalanced classes [ ]
Oversampling of minority classes
Undersampling of majority classes

### 3. Model training [ ]

### 4. Model evaluation [ ]
Potential evaluation metrics:
- accuracy
- confusion matrix
- precision
- recall
- F1-score
- latency
- throughput

For fun: describe what needs to be done to deploy the best model

I've added saving functionality to my id3 implementation. This is how the tree
can then be loaded:

```py
import pickle
from assignment4.dec_tree.id3 import predict

with open("tree.pkl", "rb") as file:
    tree = pickle.load(file)

# observation is a dictionary or pandas Series of feature values
prediction = predict(tree, observation)
```

## Generic classification ensembles

`assignment4.ensemble` provides `bagging` and `boosting`. Both accept:

- `train_model(sample_df)`: construct and return a fresh fitted classifier.
- `predict_model(model, row)`: predict a single class label from a dictionary
  or pandas Series of feature values.

The training DataFrame must have its target column last. The callbacks can
wrap ID3, k-nearest neighbours, or another classifier. Any model-specific
preprocessing belongs in those callbacks and must be applied consistently at
prediction time. Class labels must be hashable, nonmissing scalar values.

For example, wrap the existing ID3 implementation using the original training
set's categorical domains. This supplies fallback branches for categories
omitted by a particular bootstrap sample:

```python
from functools import partial

from assignment4.dec_tree.id3 import id3, predict
from assignment4.dec_tree.info_metrics import information_gains
from assignment4.ensemble import bagging, boosting

# training_df is already preprocessed; its target column is last.
# numerical_cols contains both discrete and continuous numerical predictors.
domains = {
    col: training_df[col].unique().tolist()
    for col in training_df.columns[:-1]
    if col not in numerical_cols
}
train_tree = partial(
    id3,
    info_metric=information_gains,
    continuous_features=numerical_cols,
    domains=domains,
)

bagged = bagging(
    training_df, train_tree, predict, n_estimators=20, random_state=42,
)
boosted = boosting(
    training_df, train_tree, predict, n_estimators=20,
    learning_rate=1.0, random_state=42,
)

# test_df uses the same preprocessing and feature names.
predictions = bagged.predict_many(test_df)
one_prediction = boosted.predict(test_df.iloc[0])
```

Bagging uses equal-weight majority voting. By default, each model receives
`len(training_df)` draws with replacement. For sub-bagging, pass
`bootstrap=False` and a smaller integer `sample_size`. Vote ties are resolved
by the class's first appearance in the original training set.

Boosting implements multiclass AdaBoost (SAMME) through weighted resampling,
so base learners need not implement sample weights. Each round draws a full
replacement sample and evaluates its learner on all original training rows.
This is training error, not validation error. For weighted error `error` and
`K` classes, the learner's vote weight is
`learning_rate * (log((1-error)/error) + log(K-1))`. Misclassified observations
receive a factor `exp(alpha)` before weights are normalised. In the binary
case, this gives the same normalised updates and voting decisions as the
usual symmetric half-log formulation. Confidence increases as error decreases.
See the [SAMME paper](https://web.stanford.edu/~hastie/Papers/samme.pdf).

A perfect learner ends boosting and is returned alone as the dominant vote.
A learner with error at least `1-1/K` is rejected: the first such failure
raises `ValueError`; a later failure warns and returns the accepted learners.
Unrestricted trees may fit perfectly and terminate boosting early. The generic
callback allows a restricted or pruned learner to be substituted. Resampling
is stochastic and is not identical to fitting directly with sample weights.

`random_state` controls ensemble sampling; independently seed any randomness
inside the base learner. Resampling uses training data only. The base learner
remains responsible for handling categories absent from the entire training
set; ID3's current predictor raises `KeyError` for those categories.

Returned objects expose `estimators`, `estimator_weights`, and, for boosting,
`estimator_errors`. They can be pickled when their models and prediction
callback are pickle-compatible. Prefer importable functions over lambdas or
locally defined functions when saving models; only load trusted pickle files.

### Training progress

Pass `progress=True` to `id3`, `bagging`, or `boosting` to display a progress
bar. ID3 counts training rows assigned to terminal leaves; bagging counts fitted
models; boosting counts evaluated rounds and may stop before the requested
number. Row completion does not measure the fraction of computation finished:
the initial split search can take time before the first leaf is completed.
Progress output is disabled by default and requires no extra dependencies.

```python
tree = id3(training_df, information_gains,
           continuous_features=numerical_cols, progress=True)
bagged = bagging(training_df, train_tree, predict,
                 n_estimators=20, random_state=42, progress=True)
```

A callback `progress(completed, total)` can be supplied instead of `True` to
connect training to a notebook display or another progress monitor.

### Two-hour screening batch

Run this in a terminal from the repository root:

```sh
uv run python scripts/run_screening.py --output experiments/two-hour-screening
```

The runner has a two-hour wall-clock budget and a six-minute limit per job.
It compares single and pruned ID3 trees, bagging, sub-bagging, and SAMME
boosting using 500, 1,000, and 2,000 training rows, three sampling seeds, and
ensembles of 5 members, followed by 15-member bagging and boosting runs as time
allows. These are preliminary learning-curve and ensemble-size experiments,
using all cleaned predictors and no class balancing. Each training subsample
retains at least one observation per class, so the smallest samples are only
approximately proportional to the original class frequencies.

The live experiment bar shows processed jobs out of 54 planned jobs, elapsed
time, remaining time budget, and the current stage. While training it also
shows completed models/rounds and an ID3 row progress bar. Pruning and scoring
are named stages without invented completion percentages. Progress is saved
in per-job `.progress.json` files; `--no-progress` hides terminal bars.

A stratified 20% final test partition is reserved and never scored. The
remaining data are split into growth, pruning, and tuning partitions. Pruning
uses its own 2,000-row sample; tuning metrics use the full tuning partition.
Results include macro-F1, balanced accuracy, accuracy, per-class metrics,
confusion matrices, fit and prediction timings, and fitted models.

The output directory contains `summary.csv`, per-job metrics and logs,
`status.json`, partition indices, and a frozen source-code snapshot. Callbacks
in saved ensembles are defined in the snapshot's `run_screening.py`; add its
directory and `code/src` to your Python import path when loading these models.
Timed-out and failed jobs are recorded, and finished results survive Ctrl-C.
Run the same command with the same output directory to resume unattempted
jobs. Each invocation receives a fresh time budget; existing snapshots and
partitions are reused. Keep the terminal running for the batch to continue.

### Checking the implementation

The tests use Python's standard-library `unittest` runner and the project's
existing dependencies; no additional testing dependency is required:

```sh
uv run python -m unittest discover -s tests -v
uv run python -m compileall src/assignment4
```
