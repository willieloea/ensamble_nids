# Correlation feature ablation

This experiment compares all cleaned predictors with two reduced sets. It tests
whether correlation-based reduction improves prediction or training cost, and
whether the retained representative matters. It uses the same partitions as
the completed two-hour screening, without scoring the final test set.

Run from the repository root:

```sh
uv run python experiments/correlation-ablation/run.py
```

Keep the terminal open. The runner shows overall and ID3 training progress,
saves each finished job, and limits each invocation to a 30-minute budget
including preparation, with a small allowance for final file writes. Each job
has a four-minute timeout. Completion of all nine fits is not guaranteed; the
previous baseline timings suggest roughly 15–20 minutes for the batch, but
reduced feature sets can produce different tree shapes and runtimes.

## Fixed design

| Setting | Value |
| --- | --- |
| Source partitions | `../two-hour-screening/development_pools.pkl` |
| Training observations per fit | 2,000 |
| Training sampling seeds | 42, 43, 44 |
| Pruning observations | Same separate 2,000-row sample, seed 2026 |
| Tuning observations | All 41,234 rows from the existing tuning partition |
| Final test observations | Never loaded by the ablation runner |
| Induction | Original screening's frozen ID3 implementation; information gain |
| Correlation screening | Numerical growth-pool predictors only |
| Pair threshold | `max(abs(Pearson), abs(Spearman)) >= 0.95` |
| Feature policies | All predictors; representative A; representative B |
| Fits / scored models if complete | Nine fits, each evaluated unpruned and pruned: 18 results |

The all-feature baseline is **refitted** here, rather than mixing historical
timings with new fits. Within each seed, all variants use the same sampled
rows, class counts, pruning rows, tuning rows, and categorical domains. Only
predictor columns change. Feature order among retained columns is preserved.
There is no class weighting, balancing, or new hyperparameter tuning.

## How the reduced sets are chosen

1. Calculate Pearson and Spearman correlations using only the original growth
   pool's numerical predictors. The target is excluded.
2. Consider strong pairs from strongest to weakest. Merge groups only if
   **every pair** in the merged group meets the threshold. This deterministic
   greedy complete-link rule avoids assuming correlation is transitive.
3. Set A retains the **first** feature in each group's original column order;
   set B retains the **last**. All ungrouped and categorical predictors remain.

These are deliberately simple, target-independent sensitivity policies. They
do not claim either representative is best. The choices are fixed before any
tuning scores are examined, and both reduced sets have the same number of
predictors. On the current source data, ten groups reduce 41 predictors to 27:

| Correlated group | Representative A | Representative B |
| --- | --- | --- |
| `spkts`, `sbytes`, `sloss` | `spkts` | `sloss` |
| `dpkts`, `dbytes`, `dloss` | `dpkts` | `dloss` |
| `rate`, `sload` | `rate` | `sload` |
| `dttl`, `dinpkt` | `dttl` | `dinpkt` |
| `dload`, `dmean` | `dload` | `dmean` |
| `swin`, `dwin` | `swin` | `dwin` |
| `tcprtt`, `synack`, `ackdat` | `tcprtt` | `ackdat` |
| `trans_depth`, `ct_flw_http_mthd` | `trans_depth` | `ct_flw_http_mthd` |
| `ct_srv_src`, `ct_dst_src_ltm`, `ct_srv_dst` | `ct_srv_src` | `ct_srv_dst` |
| `ct_dst_ltm`, `ct_src_dport_ltm` | `ct_dst_ltm` | `ct_src_dport_ltm` |

Groups are not equivalence classes. For example, `rate` correlates strongly
with `sinpkt`, but `sload` and `sinpkt` do not meet the threshold. Therefore,
`sinpkt` is not merged into the `rate`/`sload` group. Some cross-group strong
pairs can remain after reduction. These are recorded explicitly in
`results/feature_sets.json`, along with every retained feature and all strong
pairs. This experiment tests the specified reduction policies, not a guarantee
that the remaining predictors have no strong correlations.

To inspect and freeze the choices without training:

```sh
uv run python experiments/correlation-ablation/run.py --prepare-only
```

Run the normal command afterward to train using those frozen choices. The
original two-hour results are read but never overwritten.

## Outputs and interpretation

Outputs go into `experiments/correlation-ablation/results/`:

- `feature_sets.json`, `correlations.csv`: exact growth-only selection evidence.
- `plan.json`, `manifest.json`: jobs, configuration, source identity and hashes.
- `status.json`: completion, timeout, and failure counts.
- `summary.csv`: each completed raw/pruned model, timings, and within-seed
  differences from the all-feature model of the same pruning type; also
  decision-node counts, stored leaf counts, maximum depth, and model file size.
- `per_class.csv`: class-level metrics and matched recall/F1 changes for each seed.
- `aggregate.csv`: descriptive means and standard deviations for completed runs.
- `RESULTS.md`: automatically updated table of **matched** differences.
- `variants/<feature_set>/jobs/`: per-class precision/recall/F1, confusion
  matrices, progress checkpoints, and job outcomes.
- `variants/<feature_set>/models/`: trained tree dictionaries, including pruned copies.
- `variants/<feature_set>/logs/`: worker warnings or exceptions.
- `code/`: frozen worker, package implementation, runner, and dependency files.

Input development pools are copied with the chosen columns into each variant
directory so later changes to the source experiment cannot change the fits.
This uses approximately 160 MB of additional storage. Resuming verifies the
prepared input hashes and rejects changed configurations. Run the same command
to resume **unattempted** jobs; already recorded timeouts/failures are skipped.
Use a fresh output directory to rerun them:

```sh
uv run python experiments/correlation-ablation/run.py --output experiments/correlation-ablation/repeat
```

Inspect **paired macro-F1 differences across the three seeds**, followed by
per-class recall, accuracy, fit time, and model size. Stored leaf counts include
categorical fallback leaves; model size is not an induction limit.
The raw and pruned results share a fitted tree and are not independent repeats.
Small or inconsistent score changes are inconclusive; mean differences should
not hide a large loss of recall for an attack class. A configuration with
missing seeds must be compared only on seeds present in its baseline too.

- Similar prediction with faster training supports reduction for efficiency.
- A loss with only one reduced set shows representative choice matters.
- A loss with both supports retaining the correlated predictors for this setup.
- These are small-sample tree results. They cannot establish that each dropped
  feature is individually redundant, or that the same choices suit an ensemble.

The rarest fitting class still has one observation, as in the original batch.
Any feature-retention decision remains provisional until checked with more
training data. No non-inferiority margin or significance threshold is assumed.

## Verification

Tests use the existing environment and standard-library `unittest`:

```sh
uv run python -m unittest discover -s tests -v
```

The ablation tests cover non-transitive correlation groups, target-independent
representative selection, paired result alignment, and a small subprocess run
of all three policies with shared partitions.
