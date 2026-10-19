# Decision tree and ensemble comparison

For supplementary ThinkBook runs extending bagging to 50 members and comparing
information gain with gain ratio, see [FOLLOWUPS.md](FOLLOWUPS.md).

**The final parallel experiment is ready:** see
[FINAL_EXPERIMENT.md](FINAL_EXPERIMENT.md) for its locked design, CPU allocation,
ThinkBook/Vostro commands, result transfer, and generalization evaluation.
The remainder of this README documents the earlier runtime pilot.

This directory will hold the runtime pilot and then the final experiment for
the report question: **How does the performance of a decision tree and different
ensembles of decision trees compare?**

Current stage: **runnable runtime pilot**. `config.json` records the settings.
Update this directory after reviewing the pilot timings, preserving the pilot's
configuration and results separately from the final run. The final experiment
and its generalization-set evaluation are not implemented by this pilot.

## Running the pilot

From the repository root:

```bash
uv run python experiments/final-tree-comparison/run.py --prepare-only
uv run python experiments/final-tree-comparison/run.py
```

Preparation freezes the code, configuration, dependency lock, and development
data, verifies partition sizes, and performs no training. It is optional: the
normal run also prepares new output automatically. Keep the computer awake and
the runner active until it finishes. The default output is
`experiments/final-tree-comparison/results/runtime-pilot/`.

Run the same command to resume: complete jobs are skipped; incomplete jobs
restart from round one with the previous attempt archived. The runner prevents
concurrent training in the same output directory. Interrupted, failed, timed-out,
or deferred jobs result in a nonzero exit status. Partial measurements remain
available, but only completed jobs inform forecasts.

The per-job timeout is **two hours**, configurable before preparing output.
There is no default overall time limit. An optional `--seconds SECONDS` limits
the invocation; a new job starts only when its entire configured timeout fits
in the remaining budget. A budget below two hours therefore starts no jobs
with the default configuration. The runner never truncates a job's timeout to
fit that budget.

To regenerate reports without training, including while a run is active:

```bash
uv run python experiments/final-tree-comparison/run.py --summarize-only
```

To change prepared settings or source code, use a fresh output directory:

```bash
uv run python experiments/final-tree-comparison/run.py --output experiments/final-tree-comparison/results/runtime-pilot-v2
```

Existing output always uses its frozen code. Edits to this working directory do
not change prepared jobs; edited frozen inputs are rejected by hash verification.

### Where to find results

- `FORECAST.md`: current projected duration for three and five seeds, with
  configuration coverage and limitations.
- `projections.csv`: per-configuration estimates at 5, 10, 25, and 50 members.
- `total_projections.csv`: totals counting each ensemble once to 50 members.
- `summary.csv` and `timings.csv`: completed jobs and measured phase timings.
- `jobs/<id>/progress.json`: live phase, member count, and tree row progress.
- `jobs/<id>/timings.jsonl`: phase timings written as work completes.
- `jobs/<id>/rounds.jsonl`: boosting attempts, acceptance, and weighted errors.
- `jobs/<id>/result.json` and `worker.log`: status, memory, and worker diagnostics.
- `jobs/<id>/*.pkl` and checkpoint JSON files: fitted models and evaluation metrics.
- `attempts/`: preserved incomplete attempts; excluded from projections.

The parent prints one status line when each job is processed. Live member and
tree-row counts remain available in each job's `progress.json` file.
Peak memory is measured per worker process on Linux. Saved models require the
frozen `code/` and `code/src/` directories on `PYTHONPATH` when loading, since the
prediction adapter is defined in `run_screening.py`. Only load trusted local
pickle files.

Run the focused validation suite with:

```bash
uv run python -m unittest discover -s tests -p test_final_runtime_pilot.py -v
```

## Agreed comparison

- Balanced unrestricted, unpruned decision tree.
- Balanced pruned decision tree, derived from the same fitted unrestricted tree.
- Balanced bagging with unrestricted, unpruned members.
- Balanced SAMME boosting, testing maximum depths 3 and 8 and learning rates
  0.5 and 1.0.

All methods use information gain and all 41 cleaned predictors. The final
ensemble checkpoints are **5, 10, 25, and 50 members for both methods**. Each
ensemble is fitted once to 50 members, recording its prefixes at checkpoints;
these are not four independent fits. Record early stopping explicitly.

Pruning uses the existing accuracy-based reduced-error rule on the separate
pruning subset. Ensemble members are not pruned in this comparison.

## Data and balancing

Reuse the existing fixed, stratified partitions. Do not reshuffle observations
that have already been used in development into the final test partition.

| Partition | Rows | Use |
| --- | ---: | --- |
| Growth | 123,683 | Source pool for training samples |
| Pruning | 41,225 | Pruning decisions |
| Validation | 41,234 | Settings selection and development comparison |
| Generalization / final test | 51,531 | Final comparison only |

Development data are available in
`../overnight-learning-curves/results/development_pools.pkl`. That file contains
growth, pruning, and validation data, not the final test observations. The
runtime pilot must not load or score the final test data.

For class c, sample using q_c = (1 - lambda) * p_c + lambda / K, where p_c
is its proportion in the original growth pool and K = 10. Test lambda 0.5
and 1.0. Keep the total sample size at 123,683 training draws. Compute integer
quotas by taking floors and assigning remaining draws in descending fractional
remainder order, breaking ties by sorted class label.

Within each class, sample without replacement when its quota does not exceed
the available observations, otherwise sample with replacement. Preserve source
row IDs and record class counts and distinct source observations. Shuffle the
combined sample reproducibly.

- The standalone tree and boosting share the initial sample for a given seed
  and lambda. Boosting subsequently performs its weighted resampling.
- Bagging draws a fresh sample directly from the original growth pool for each
  member, using reproducible member seeds. Do not bootstrap a single previously
  balanced sample as the source pool for all members.
- Keep pruning, validation, and final-test class proportions natural.

## Runtime pilot

Use seed 42 and full-size training samples. Run jobs sequentially, on the same
machine and with the same evaluation and checkpoint procedures intended for
the final experiment.

For **each lambda**, run:

| Configuration | Pilot length |
| --- | --- |
| Unrestricted tree | Fit, evaluate, prune, evaluate |
| Unrestricted bagging | 5 members |
| Boosting depth 3, learning rate 0.5 | Up to 5 members |
| Boosting depth 3, learning rate 1.0 | Up to 5 members |
| Boosting depth 8, learning rate 0.5 | Up to 5 members |
| Boosting depth 8, learning rate 1.0 | Up to 5 members |

This is 12 initial jobs and up to 52 accepted tree inductions; boosting may
also incur rejected learner attempts. The pilot itself can take substantial
time. Run the standalone-tree jobs first so their timings are available early.

Extend the slowest boosting configuration for each lambda to 10 members. If
later rounds slow materially, extend it to 25 before relying on a 50-member
projection. Specifically, extend to 25 if the mean cost of rounds 6-10 exceeds
that of rounds 1-5 by more than 25%, excluding checkpoint work. Measure
evaluation and saving at the longer checkpoints as well. Extensions restart
from round one, preserving all earlier runs and exposing their extra cost.

### Measurements and outputs required from the runner

Record incremental, machine-readable timings for:

- Loading, sampling, and fixed job setup.
- Every learner attempt, accepted member, and boosting weight update.
- Pruning.
- Growth, actual-fit-population, and validation evaluation separately.
- Model serialization and output writing.
- Total elapsed time, process peak resident memory, tree size, and model size.

Use monotonic wall-clock timing. Separate components without double counting.
Record machine details, Python version, source hashes, seeds, and configuration.
Use fresh processes per job so peak-memory measurements are meaningful. Provide
overall and member-level progress, logs, and durable per-job results. On resume,
skip only complete jobs; preserve earlier attempts. Mark capped or interrupted
jobs as incomplete rather than extrapolating them as completed fits.

Forecast training to 5, 10, 25, and 50 members, including evaluation and saving
at all scheduled checkpoints. Show central and conservative projections,
observed round-to-round variation, and an explicit contingency allowance.
Forecast complete totals for three and five training seeds. These are planning
estimates, not confidence intervals. Do not assume linear scaling from a smaller
training dataset, or that five early boosting rounds fully predict later cost.

Planned output layout:

```text
results/
  runtime-pilot/       # Frozen inputs, configuration, code, timings and forecasts
  final/               # Separate frozen final-run configuration and results
```

Preserve the pilot output when adapting the directory for the final experiment.
Reuse pilot models only when their settings and execution match the final
configuration; partial ensembles require verified continuation support.

## Final selection and reporting plan

Three training seeds (42, 43, 44) are provisional, pending the runtime results.
With three seeds there are 36 jobs: 6 standalone-tree jobs producing both tree
variants, 6 bagging jobs, and 24 boosting jobs. Checkpoints are outputs of those
jobs, not additional training jobs.

1. Within each lambda, select each approach's settings by mean validation
   macro-F1 across seeds.
2. Select one shared lambda by averaging those four selected approaches'
   validation macro-F1 values, giving the approaches equal weight.
3. Freeze settings and retain the fitted models. Evaluate the selected models
   on the same untouched generalization subset. Do not select a winning seed
   on the final test set or use final-test results to change settings.

Report macro-F1, balanced accuracy, per-class precision/recall/F1 and support,
overall accuracy, confusion matrices, runtime, and model size. Summarize
variation across training seeds without treating shared test observations as
independent repetitions. Retain both lambda settings' validation results.

Before final launch, review predictor overlap between partitions, the existing
unseen-category prediction fallback, tie-breaking for settings selection,
resource limits, and the runtime forecasts. These are preflight items, not
additional experimental settings that have already been agreed.
