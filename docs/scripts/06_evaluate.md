# `scripts/06_evaluate.py`

**Roadmap step:** File 55 (Stage 7). Dependencies: Files 48-54 (metrics, cross-validation, plots, curves, comparator, voting), trained models from `05_train.py`.

## Role
The evaluation step of the project. In one resumable run it (1) cross-validates the 11 models on TRAINING rows, (2) cross-validates the majority vote from the out-of-fold predictions, (3) scores the held-out test set once per model, (4) takes the majority vote on the test set, (5) writes the comparison table, the report and the figures.

## Why it is needed
It produces the numbers the final report stands on, and it does so without letting the test set influence any choice: cross-validation uses training rows only, and each model scores the test set once (saved results are reused; `--rescore` is the only way to score again).

## Options
| Option | Meaning |
|---|---|
| `--dry-run` | print the plan, time estimates from your own training metadata, and which results are already saved; runs nothing |
| `--folds 5` | cross-validation folds (default 5) |
| `--cv-sample-size N` | cross-validate on a class-capped sample of N training rows instead of all rows (saved in a separate folder `cross_validation\sample_N`, never overwriting the full results) |
| `--skip-cv` / `--skip-test` | run only one half; `--skip-cv` reuses saved CV results in the report |
| `--models k1 k2 ...` | only these models; the others still appear in the table if they have saved results, otherwise they are listed as missing |
| `--rescore` | score the test set again |
| `--no-figures` | skip pictures |

## What it writes (under `reports\evaluation\`)
`cross_validation\cv_<model>.json` + out-of-fold predictions (and `cv_majority_vote.json`); `test\<model>.json` (metrics, per-class results, ROC and precision-recall curves, timings, fingerprint) and `test\<model>_pred.npy`; `model_comparison.md/.csv/.json`, `per_class_f1.csv`, `evaluation_report.md` (**the file to send me**), `evaluation_summary.json`, and about 30 figures in `figures\` (test macro F1, detection and false alarms, CV macro F1, fold spread, one confusion matrix and one per-class chart per model, curves for the best model and the top four).

## How it works
- **Cross-validation:** `cross_validation.run_cross_validation` (File 49). Each model's result is saved as soon as it finishes; a rerun reuses results whose fingerprint (data size, folds, seed, settings) matches. A model that fails is recorded and the others continue.
- **Vote in cross-validation:** vote over the saved out-of-fold predictions (each row was predicted by models that did not train on it), scored fold by fold on the same folds.
- **Test scoring:** the saved model is loaded with its checksum and freshness checks; `predict` gives the labels (used by the vote), `predict_scores` gives the scores for ROC-AUC and the curves. A saved test result is reused only if the model file (SHA-256), the number of test rows and the test labels are unchanged.
- **Vote on the test set:** one equal vote per model from the saved test predictions (File 53); missing models are listed with the reason.
- **Report:** computed observations only (best single model, vote versus best model, cross-validation versus test gaps above 0.05). Limits are written into the report: rare-class scores, the random split, subsampled models, test set not used for tuning.

## Time (rough; the dry run prints estimates from your own metadata)
Full 5-fold cross-validation on all rows: about 2 to 2.5 hours (AdaBoost, Random Forest, Decision Tree, Logistic Regression and XGBoost about 40 minutes together; Gradient Boosting about 40 minutes; KNN and SVC about 20 minutes each, because every fold predicts about 403,000 rows). Test-set scoring: about 25 to 30 minutes (KNN and SVC about 10 minutes each: predict and score). Close other heavy programs and keep the laptop awake and plugged in. Ctrl+C is safe: rerun the same command and finished models are reused.

## Commands (PowerShell, from `DM_Project\`, venv active)
1. Plan first (a few seconds, nothing runs):
```powershell
python scripts\06_evaluate.py --dry-run
```
2. The real run (long):
```powershell
python scripts\06_evaluate.py
```
If it stops or you press Ctrl+C, run the same command again.

## Expected output
The dry run prints a table with one line per model (estimated CV time, estimated test time, whether results are already saved) and `Dry run: nothing was run.` The real run logs each model's folds and test scores, then prints the ranked test table, the notes, `Report: ...evaluation_report.md (+ ~30 figure(s) ...)` and `Checks passed: 4/4`. Exit code 0 when all 11 models were scored and checks passed.

## Common errors
| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` / empty file | placeholder not pasted | paste the code; Length must not be 0 |
| `... model is stale` in the "Models not in the comparison" section | settings changed after training | rerun `05_train.py` for that model (or all) |
| `MemoryError` while scoring | very low free RAM | close programs; rerun (finished models are reused) |
| `Figure ... skipped` warning | one picture could not be drawn | the numbers are unaffected; send me the warning |
| exit code 2 | a model failed, is missing, or a check failed | the report's last sections say which |

## Limits
- The split is stratified and random, so flows from the same attack campaign can sit on both sides: scores may be optimistic compared with a time-based split.
- The three rarest classes have 2 to 7 test rows.
- Subsampled models (KNN, SVC, Gradient Boosting) see a different class mix; their false alarm rates are not directly comparable.
- The default run scores the baseline models; the tuned models are scored by `--stage tuned` (below).

---

# `--stage tuned` (scores the tuned models once; needs File 57 and the baseline test results)

## Role
Scores the 11 tuned models (`artifacts\models\tuned\`) on the test set **once each**, shows every tuned model beside its baseline, and takes three majority votes: the 11 baseline models, the 11 tuned models, and the **final set**. The final set is what the voting and the application will load.

## Why it is needed
Checkpoint 5 fixed which version of each model the final system uses (the *evidence rule*) before any tuned model had a test score. This step reports how the tuned models behave on unseen flows without letting the test set choose between tuned and baseline.

## The evidence rule (read from the tuned models' metadata, never from test scores)
A model uses its **tuned** version when the tuning folds showed an improvement beyond fold noise and the tuned parameters differ from the baseline; otherwise its **baseline**. Recorded in `docs\project\checkpoint_5_decisions.md`: tuned for KNN, Decision Tree, SVC, Bernoulli NB, SGD, AdaBoost, Gradient Boosting; baseline for Random Forest, XGBoost, Logistic Regression, Gaussian NB. The script has this set as a constant and an automatic check fails if the metadata gives a different set.

## Options
`--stage tuned` with `--dry-run`, `--rescore`, `--no-figures`. Not allowed with `--stage tuned`: `--models` (the final set needs all 11), `--folds`, `--cv-sample-size`, `--skip-cv`, `--skip-test` (no cross-validation is run for tuned models; the tuning step used it on training rows).

## Dependencies
Baseline test results from `python scripts\06_evaluate.py` (`reports\evaluation\test\`), accepted only if they used the same test rows (labels hash); tuned models from `07_optimize.py`.

## What it writes (under `reports\evaluation\`)
`test_tuned\<model>.json` and `_pred.npy` (one per tuned model), `test_tuned\majority_vote_baseline|tuned|final.json` and `_pred.npy`, `model_comparison_tuned.csv/.md/.json`, `evaluation_report_tuned.md` (**the file to send me**), `evaluation_summary_tuned.json`, and three figures in `figures_tuned\` (baseline vs tuned macro F1, confusion matrix and per-class chart of the final vote). Saved tuned test results are reused on a rerun if the model file and test labels are unchanged.

## How the report reads
Tuning evidence table (which version the final set uses and why), tuned vs baseline table (changed parameters, macro F1 for both, delta, false alarms, detection), the three votes, computed observations (including tuned models that scored lower than their baseline), the final vote's confusions and low-support classes, automatic checks, limits.

## Commands (PowerShell, from `DM_Project\`, venv active)
```powershell
python scripts\06_evaluate.py --stage tuned --dry-run
python scripts\06_evaluate.py --stage tuned
```

## Expected output
The dry run lists the 11 models with the version the final set uses (7 tuned + 4 baseline) and whether results are saved. The real run logs each tuned model's test macro F1, the three votes, prints the tuned-vs-baseline table, the vote table and observations, then `Report: ...evaluation_report_tuned.md (+ 3 figure(s) ...)` and `Checks passed: 7/7`. Time: about 15 to 25 minutes (Random Forest with 300 trees and SVC are the slowest to predict). Exit code 0 when all checks pass.

## Common errors
| Problem | Cause | Fix |
|---|---|---|
| `Baseline test results are needed` | the default run was not done, or used other test rows | run `python scripts\06_evaluate.py --skip-cv` first |
| `no saved tuned model` in the missing list | `07_optimize.py` did not finish that model | rerun `07_optimize.py` |
| check `final set equals the set recorded in Checkpoint 5` fails | tuning results were rerun and changed | send me the report; do not edit the rule to fit test scores |
| `--models ... do not apply` | options not valid for the tuned stage | remove them |

## Limits
Same limits as the default run, plus: parameters were searched on at most 100,000 rows (KNN 50,000; SVC and Gradient Boosting 30,000); some winners sit at the edge of their grid; the tuned and baseline scores are single test-set measurements on a random split, so small differences are not evidence of a real change.