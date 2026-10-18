# Overnight learning curves and ensemble comparison

Run from the repository root:

```sh
uv run python experiments/overnight-learning-curves/run.py
```

The default budget is **nine hours per invocation**, including preparation, with
30 seconds reserved for final reporting. A job starts only if its full timeout
fits in the remaining budget. Longer jobs are deferred while later shorter jobs
are still considered; the runner may stop before nine hours. It trains serially, limits individual tree jobs to 20 minutes and
ensemble/stacking jobs to one hour, and preserves partial results. Leave the
terminal running and prevent the machine from sleeping. No training starts when
using `--prepare-only`.

```sh
# Freeze inputs/code and check the development partitions without training.
uv run python experiments/overnight-learning-curves/run.py --prepare-only

# Inspect the results while the main runner is active, or afterwards.
uv run python experiments/overnight-learning-curves/run.py --summarize-only

# A different duration or separate rerun directory.
uv run python experiments/overnight-learning-curves/run.py --seconds 3600 --output /tmp/short-overnight
```

Results default to this directory's `results/`. The same command resumes
**unattempted jobs and unfinished attempts**, with a fresh nine-hour budget.
Only completed jobs are skipped. Timed-out, interrupted and failed jobs restart
from scratch, at most once per invocation. A saved partial ensemble is available
for prediction and inspection; the runner does not continue its training state.
Before a retry, old points, models, logs, progress and job status move to
`attempts/<job_id>/<attempt_number>/`. They remain available but are excluded
from current summaries; model paths in archived point files are relative to
that attempt directory. Current point/CSV rows identify their attempt number.
Jobs deferred for insufficient time retain their previous attempt untouched
and appear in `deferred_jobs.json`. One training runner may use an output
directory at a time; `--summarize-only` can still run concurrently.
Use a fresh `--output` for a complete rerun or different configuration. If no
pilot produces a raw-tree score, inspect the pilot logs before retrying.

## Questions and planned experiments

All experiments use **all 41 cleaned predictors**, including correlated ones.
They use the corrected, faster numerical threshold scoring. The target is
`attack_cat` (ten classes), with `proto`, `service`, `state`, and
`is_sm_ips_ports` categorical and all remaining predictors numerical. No further
feature selection or final test evaluation is performed.

| Experiment | Settings | What it tests |
| --- | --- | --- |
| Runtime pilots | Natural unrestricted trees, 5,000 and 10,000 rows, seed 42; raw and pruned | Select a practical common sample size using runtime only |
| Split criterion | Information gain (`entropy`) only for this overnight batch; gain ratio remains available for later | Establish the information-gain baseline before comparing other criteria |
| Learning curves | Natural unrestricted and pruned trees at 2,000, 5,000, 10,000, 20,000 and 50,000 rows; seeds 42–44 | Whether additional independent training observations improve generalisation and rare-class coverage |
| Class distribution | Natural proportions versus equal-class resampling at the same fitting row budget | Whether changing training class proportions helps macro-F1 and minority recall |
| Depth curves | Maximum depths 0, 1, 2, 4, 8, 12, 20 and unrestricted; both class policies; seeds 42–44 | Underfitting, overfitting and the training/validation gap as tree complexity increases |
| Bagging curves | 25 unrestricted trees; both policies; seeds 42–44 | Whether averaging trees improves generalisation, and how gains change with ensemble size |
| Boosting curves | 25 SAMME rounds; maximum base-tree depth 3; learning rates 0.5 and 1.0; both policies; seeds 42–44 | How sequential correction, ensemble size and learning rate affect error and minority classes |
| Holdout stacking | Information-gain base trees at unrestricted, depth 3 and depth 8; depth-3 entropy meta-tree; both policies; seeds 42–44 | Whether learned combination improves over its exact constituent models and their majority vote |

Pilots use the first configured criterion (information gain by default).
The common size is the **largest pilot whose raw fit takes at most 120 seconds**;
if neither meets that limit, use the smallest completed pilot. This choice does
not use validation performance. See `selection.json`. A pilot's completed raw
score remains usable if subsequent pruning times out.

The queue first completes unrestricted baselines at the common size,
for information gain, all seeds and both class policies. It then expands to smaller
learning sizes, depth sweeps, ensemble trajectories and stacking. Larger learning
sizes are last. Pilot fits are reused instead of duplicated. The default plan
has 84 unique jobs including the two pilots, for either common sample size.
The 42 depth-sweep jobs save raw trees only; they skip pruning and pruned scoring.
Unrestricted baselines still save both raw and pruned scores. **The time budget
may prevent all jobs or repetitions from finishing.** `plan.json`, per-job
statuses and point files specify what actually happened. The plots show how
many seeds contributed to each point.

## Information gain versus gain ratio

`config.json` contains `"criteria": ["entropy"]` for this overnight batch.
`entropy` means raw information gain. To enable a paired comparison later, set
`"criteria": ["entropy", "gain_ratio"]` in a new configuration/output. Both criteria are applied to categorical feature
choices and numerical threshold choices, through the existing fast scorer.
Set this list to `["gain_ratio"]` to run gain ratio alone, or `["entropy"]`
for information gain alone; use a new output directory when changing settings.
`gini` is also supported as an explicit criterion. Unknown names fail rather
than silently selecting a different scoring function.

Single-tree learning/depth curves, bagging and boosting are repeated for each
criterion. Their job IDs, model files, point JSONs, CSV rows, plots and aggregates
identify the criterion. Pairs use identical source samples, class resampling
seeds, depth limits and ensemble sampling seeds. Later boosting samples may
differ because different base predictions change the adaptive weights.
Stacking uses information gain throughout: unrestricted, depth-3 and depth-8
base trees, with a depth-3 meta-tree. It is not duplicated for each optional
criterion. Each base learner uses an independent resampling seed (`seed + i`).
Natural-policy bases bootstrap their base-training partition; balanced-policy
bases draw fresh equal-quota samples from that partition. They therefore do
not fit truncated prefixes on identical data. All remain trees, so diversity
and improvements must still be assessed empirically.

`criterion_comparison.csv` contains matched completed information-gain/gain-ratio
points, including both scores and **gain ratio minus information gain** deltas.
Positive macro-F1/accuracy deltas favour gain ratio; negative error deltas favour
gain ratio. A missing partner is omitted, never replaced with zero. Comparisons
are paired on model kind, sample size, seed, policy, depth, learning rate,
variant and ensemble round. Different ensemble stopping behavior can leave
unmatched rounds. The CSV is refreshed after each scored checkpoint.

Single-tree point files and `summary.csv` also record the root feature, root
split type and numbers of categorical/numerical internal nodes, alongside node,
leaf and depth counts. These support checking whether performance changes
coincide with changes in tree structure; root frequency alone does not prove a
causal explanation. Plots and means never pool distinct criteria.

This is **ID3-style induction using gain ratio**, not the full C4.5 algorithm:
the existing candidate-threshold rules remain in use and there is no C4.5
information-gain eligibility filter before maximising the ratio. Gain ratio is
not assumed to be better; validation results decide whether it helps here.

Enabling both criteria expands the plan to 162 jobs and 84 depth-sweep fits.
Paired single-tree baselines are then prioritized so the criterion question can
be addressed even if the remaining sweeps do not all fit inside nine hours. For an existing frozen output, use a new
`--output` path to pick up these code/config changes; snapshots are not silently
rewritten.

## Data separation and class balancing

The runner copies `development_pools.pkl` and the source manifest from
`experiments/two-hour-screening/`, reusing its fixed growth/pruning/validation
partitions. The source split reserved 51,531 final test rows; those rows are
absent from this pickle and cannot be scored here. Growth has 123,683 rows,
pruning 41,225, and validation 41,234. Stratified growth subsamples retain each
class, use seeds 42–44, and are paired across configurations. A fixed 10,000-row
stratified sample of the pruning pool (seed 2026) supports reduced-error pruning.
The validation pool is used only for scoring and tuning, never for fitting or
pruning. Pruning retains its existing **unweighted classification-error**
objective, even when growth rows were balanced.

Equal-class resampling allocates approximately `n / 10` fitting rows per class,
undersampling common classes without replacement and oversampling rare ones
with replacement. It changes both class balance and independent-row coverage;
it is not a pure test of class weighting. Duplicating rare observations creates
no new independent evidence. Single trees and boosting share an initial balanced
fitting population for a given source sample and seed. Boosting resamples it
using its learned weights. Balanced bagging instead draws a fresh equal-quota
sample directly from the **full original growth subsample for every member**,
using `seed + member_index`. There is no fixed discarded majority subset and
no additional bootstrap of a pre-balanced frame. Majority draws are without
replacement within a member; minority oversampling uses replacement. Natural
bagging retains its ordinary bootstrap sampling. Balanced members record source
counts relative to the original growth subsample; checkpoint metadata records
the cumulative distinct-source coverage.

Stacking instead splits the **original** growth subsample into stratified 80%
base-training and 20% meta-training portions, then balances separately inside
each layer's fit callback. Base callbacks use distinct seeds; natural bases
bootstrap independently, and balanced bases independently balance the original
base partition. The meta holdout is never included in any base resampling pool.
This prevents oversampled copies of an observation
crossing that split. It uses one-hot hard predictions as meta-features and does
not refit base trees on the meta holdout. Each class must have at least two
original rows; an insufficient sample fails explicitly rather than leaking
copies across partitions. Base learners are all trees, so diversity and gains
are not guaranteed. Its constituent and majority-vote comparisons use the exact
same fitted base models. Their base-training sample is smaller than that of the
full-data single-tree baseline; interpret the comparison accordingly.

Categorical branch domains come from growth predictors only. A prediction row
with a categorical value absent from that domain receives the fitted model's
global majority class, matching the screening adapter. This applies even if the
particular tree never splits on that feature. Validation-only state values
therefore have a documented fallback rather than influencing tree growth.

## Live results and checkpoints

- `summary.csv`: refreshed **after every scored checkpoint**, containing training
  and validation accuracy, error, macro-F1, balanced accuracy, fitting-population
  metrics for balanced models, evaluation scope, timings, tree
  complexity, member count, model path and current job status.
- `per_class.csv`: precision, recall, F1 and support for each class and split.
- `points/*.json`: full metadata, confusion matrices, class counts and scores.
- `models/`: atomic pickle files for each scored point and a `__latest.pkl`
  updated **after every accepted ensemble member**, before checkpoint scoring.
- `attempts/`: preserved historical retries, excluded from current score tables.
- `deferred_jobs.json`: jobs deferred because their full timeout would not fit.
- `progress/*.json`, `status.json`, `jobs/*.json`, `logs/*.log`: live stages,
  tree row progress, accepted members, timeouts and failures. Logs retain errors.
- `plots/*.png`, `aggregate.csv`, `RESULTS.md`: generated at the end or using
  `--summarize-only`. CSVs and plots include usable points from unfinished jobs.
- `code/`, `config.json`, `manifest.json`, `source_manifest.json`: frozen code,
  settings and SHA-256 hashes, allowing results to be traced to this version.

Ensembles are scored at rounds **1, 2, 5, 10, 15 and 25**, plus their final
accepted round if training ends earlier. These are prefixes of one trajectory,
not independent refits at each size. Boosting can reject a learner or end on a
perfect learner; a perfect learner replaces earlier members. `round` and
`actual_estimators` distinguish the attempted round from the retained count.
No point is invented for an uncompleted round. If killed while scoring,
`__latest.pkl` still preserves the most recently accepted ensemble; an
incomplete score is not written. Tree fits are saved after induction; their
pruned versions are saved separately.

The terminal displays an overall job bar, current training/scoring/pruning
stage, model round, ID3 rows completed, and elapsed job time. Tree row progress
counts observations assigned to leaves and is not a reliable percentage of CPU
work. Checkpoint evaluation is an explicit stage. Fit times exclude checkpoint
serialization and evaluation, but include worker data loading/setup; pruned
points include pruning and also report `prune_seconds` separately. Stacking
points share the full stack fit time because its constituents are fitted once.

## Reusing pruning and checkpoint computations

Reduced-error pruning now carries each child's retained validation-error count
back to its parent. It sums those counts instead of predicting each subtree
again. The decision rule, majority/tie behavior, empty-branch handling and
returned tree representation are unchanged. An internal subtree without
training observations is retained and scored once. Data still have to be routed
through the tree, so this does not remove the cost of partitioning rows at each
depth. Unsupported validation categories still raise an error.

During ensemble checkpoint evaluation, each member's predictions are cached for
the fixed original training subsample and fixed validation pool. An append-only
trajectory adds the new members' votes; earlier members are not predicted again.
Thus checkpoints at 1, 2, 5, 10, 15 and 25 members need **25 member prediction
passes per evaluation split**, rather than 58. Members added between checkpoints
are evaluated together at the next checkpoint. If boosting replaces earlier
members with a perfect learner, vote totals are reset; changed weights or
reordered members also rebuild votes from the cached predictions. Class-order
tie resolution and member-order floating-point addition are preserved.

The cache is local to one worker, assumes deterministic prediction callbacks
and immutable fitted members, and is not saved inside models. Ordinary
`predict_many` and loaded-model predictions remain independent of the cache.

Classification scores remain directly comparable to full evaluation, but
`prediction_seconds` for cached checkpoints measures **incremental evaluation
work**, not full ensemble prediction latency. Point JSONs and `summary.csv`
identify this using `evaluation_mode` / `train_evaluation_mode` /
`validation_evaluation_mode` (`cached_members` or `full`), and record how many
new members were scored. Use a separate timed `predict_many` call on a saved
model to measure deployment prediction latency. These cached timings must not
be compared as if they were full inference timings from earlier runs.

## Graphs and interpretation

Each curve plots **training and validation classification error**, alongside
training and validation macro-F1. Training scores use the **original growth
subsample**, not duplicated bootstrap/balanced fitting rows. Validation scores
always use the full fixed validation pool. This makes scores comparable across
class policies, but balanced training need not monotonically reduce unweighted
source training error. Inspect macro-F1 and per-class recall alongside accuracy.

Balanced curves also include a green **Fit population** line. Point JSONs use
`fit`, and CSVs use `fit_accuracy`, `fit_error`, `fit_macro_f1`, etc.; `per_class.csv`
also includes `split=fit`. `fit_evaluation_scope` states which observations were
scored:

- `shared_balanced_training_population`: actual balanced rows used by a single
  tree, or the fixed population on which SAMME calculates its weighted error.
  The latter is the ensemble's training population, not a claim that every
  bootstrap learner saw every row.
- `cumulative_member_fit_draws`: ensemble predictions on all balanced bagging
  member fitting draws so far, including their observed multiplicities. Each
  draw was used by at least one member; it need not be in every member's sample.
  This population grows with the ensemble. Predictions are reused from the
  original-sample cache, avoiding repeated feature-row/model evaluation.
- `own_fitted_rows`: each stacking constituent's own actual balanced fitting
  rows. The meta-learner's fitting score is in `stack_fit_details.meta.fit_metrics`.
  There is no artificial combined-stack or majority-vote fit score: their two
  layers/members do not share one balanced fitting population. Missing values
  are left blank, not zero.

Keep these fit scopes separate in the report. They complement the unchanged
original-growth and independent validation scores; they are not extra validation
sets. Stacking fitting time includes recording component fit metrics.

Depth curves can show training error falling while validation error rises.
Learning curves show the effect of adding independent data. Ensemble curves
show what happens as more models are combined; bagging often stabilises rather
than producing a classic overfitting curve. These are useful evidence about
complexity and generalisation, **not a numerical bias–variance decomposition**.
Overfitting is an empirical finding to check, not an outcome guaranteed by the
experiment design.

Lines are means over available sampling seeds; shaded bands are **sample
standard deviations**, not confidence intervals. Seeds share the same
validation pool and do not represent independent dataset splits. Missing runs
can leave unequal repeat counts; compare matched seeds before claiming an
improvement. The small classes still have few independent examples at these
training sizes. Stacking's source training score covers rows used to fit either
the base or meta layer, so it is descriptive rather than an independent score.
The best validation point is a tuning result, not a final test estimate.

## Loading a saved model

Use the frozen adapters and package to reproduce its prediction behavior:

```python
import sys
from pathlib import Path
import pickle

output = Path('experiments/overnight-learning-curves/results').resolve()
sys.path.insert(0, str(output / 'code/src'))
sys.path.insert(0, str(output / 'code'))
from overnight_core import predict_tree

# Choose an actual model_file from summary.csv. Load only trusted pickle files.
with (output / 'models/YOUR_POINT.pkl').open('rb') as file:
    model = pickle.load(file)
prediction = predict_tree(model, observation) if isinstance(model, dict) else model.predict(observation)
```

## Verification

No additional dependencies are required. From the repository root:

```sh
uv run python -m unittest discover -s tests -v
uv run python -m compileall src/assignment4 experiments/overnight-learning-curves
```

The overnight tests cover categorical/numerical gain-ratio dispatch, paired
criterion plans and reports, depth bounds, deterministic ensemble-prefix
checkpoints, cached-score equality and prediction-call counts, perfect-learner
replacement, balancing provenance and a complete
synthetic frozen batch with plotting, stacking and resume behavior. Scheduling
tests cover fresh full-pool balanced member draws, independent stacking
resampling with meta-holdout exclusion, cached fit-score equality, full-timeout admission, deferral followed by shorter jobs, retry
archival, per-invocation retry limits, and skipping pruning for depth sweeps. Synthetic
scores are not assignment results.

## Completed-run analysis

See [RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md) for the completed overnight run,
including matched comparisons, per-class tradeoffs, training-label conflicts
and the stacking meta-model diagnosis. Derived tables and overview figures are
in `results/analysis/`. Reproduce the checks and figures with:

```sh
.venv/bin/python experiments/overnight-learning-curves/analyze_results.py
```

This analysis does not retrain models or evaluate the final test partition.
