# `src/nids/evaluation/metrics.py`

**Location:** `DM_Project/src/nids/evaluation/metrics.py`
**Roadmap step:** File 48 (Stage 7, evaluation)

## Purpose

Computes every evaluation metric of the project from arrays (true labels, predicted labels, scores), and gives the results as plain JSON-safe dictionaries and tables.

## Why it is needed

The cross-validation (File 49), the plots and curves (50-51), the comparison table (52) and the evaluation script (53) must all score models the same way. This file is that single definition. Accuracy alone is misleading here: BENIGN is 83% of the traffic, so a weak detector can still look good. The metrics below expose weak classes and report the numbers an intrusion detection system is judged by.

## Dependencies

`numpy`, `pandas`, `scikit-learn.metrics`, `nids.preprocessing.target_builder.LabelCodec`.

## What is computed

**Overall:** accuracy, balanced accuracy, macro precision / recall / F1, weighted F1, Matthews correlation coefficient (MCC), macro ROC-AUC. Macro scores give every class the same weight, so they reveal weak rare classes. They are averaged over the classes that exist in the true labels (`macro_over_classes`).

**Per class:** precision, recall, F1, ROC-AUC (one class against the rest), the number of rows predicted as that class and the **support** (true rows). A class with fewer than **30** true rows is flagged `low_support` and labelled "low support: unreliable" in the table. On your test set that means Heartbleed (2 rows), SQL Injection (4) and Infiltration (7), as accepted at Checkpoint 3. The flag always accompanies the numbers.

**Confusion matrix** (rows = true, columns = predicted), a normalised version, and the 10 most frequent confusions (with the share of the true class they represent).

**Attack versus normal view:** a flow counts as an attack if its class is not BENIGN. *Detection rate* = attack flows predicted as any attack; *false alarm rate* = BENIGN flows predicted as any attack; plus attack precision, F1 and the counts of missed attacks and false alarms. Naming the wrong attack type still counts as a detection in this view.

**ROC-AUC:** one-versus-rest per class, from the model's scores, macro-averaged. For SVC and SGD the scores are decision scores, not probabilities; AUC only uses the ranking, so it is valid, and the result records `score_kind` so the table can say so. A class with no true rows (or no other-class rows) gets `null` and is excluded from the mean. For models with probabilities the macro value equals scikit-learn's `roc_auc_score(multi_class="ovr")` (verified).

Predicted labels always come from `model.predict`, never from the scores (they can differ in rare SVC ties).

## Functions

| Function | What it does |
|---|---|
| `compute_metrics(y_true, y_pred, codec, scores=None, score_kind=None)` | all the metrics above for one model on one set of rows |
| `evaluate_model(model, X, y, codec, with_scores=True)` | predicts with a fitted model, measures `predict_seconds` and `score_seconds`, and computes the metrics. `with_scores=False` skips the scores (and AUC): KNN and SVC need as long again for them |
| `per_class_table(metrics)` | per-class DataFrame including support and the low-support note |
| `confusion_frame(metrics, normalize=False)` | the confusion matrix with class names |
| `per_class_auc`, `binary_view`, `top_confusions` | the building blocks |
| `LOW_SUPPORT_THRESHOLD` | 30 |

## Input checks

Clear `ValueError`s for: different lengths, non-integer class codes, codes outside the class range, scores with the wrong shape, scores that are missing or infinite, and no rows.

## How it was checked (sandbox)

On a hand-computed example (12 rows, 4 classes): the confusion matrix, false alarms (2), missed attacks (1), detection rate (5/6), false alarm rate (2/6) and attack precision (5/7) equal the values worked out by hand; accuracy, macro and weighted F1 and macro precision equal scikit-learn's; a class missing from the true labels is excluded from the macro mean; a class that is never predicted gets precision 0, not an error. On synthetic data with Random Forest, SGD, SVC and KNN, the macro AUC of the probability models equals scikit-learn's, decision-score models give an AUC, and `with_scores=False` gives no AUC. All six invalid-input cases raised the intended error.

## How to test on your real data

**This test deliberately does not use the test set.** You will see test-set scores once, in the final comparison (Files 52-53), so that no model choice can be influenced by them. The command evaluates the saved Random Forest on a class-balanced slice of **training** rows (up to 500 per class, 6,055 rows). The scores will be near-perfect because the model has seen these rows: it checks the code on the real model and classes, not performance. PowerShell, from `DM_Project\`, venv active:

```powershell
python -c "import numpy as np; from nids.evaluation import metrics as M; from nids.persistence import model_store as ms; from nids.training.data_access import load_training_data; d = load_training_data(mmap=True); idx = np.sort(np.concatenate([np.flatnonzero(d.y == c)[:500] for c in range(15)])); X, y = np.asarray(d.X[idx]), d.y[idx]; r = M.evaluate_model(ms.load_model('random_forest'), X, y, d.codec); print(r['n_rows'], r['score_kind'], {k: round(v, 4) for k, v in r['overall'].items()}); print(M.per_class_table(r).round(3).to_string(index=False)); print(r['attack_vs_normal']); print(r['top_confusions'][:3])"
```

**Expected output:** `6055 probability {...}` with accuracy and macro scores close to 1 and AUC close to 1; a table of 15 classes whose support is 500 for twelve classes and 9 (Heartbleed), 17 (SQL Injection), 29 (Infiltration), where exactly those three are marked "low support: unreliable"; the attack-versus-normal dictionary (`attack_rows` 5,555 and `benign_rows` 500 in this slice, high detection rate, low false alarm rate); and the largest confusions, probably few and small.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... metrics` | placeholder empty | paste the code, check the length is not 0 |
| `ModelStoreError ... no saved trained model` | `05_train.py` has not been run | run it first |
| `StaleModelError` | settings changed after training | rerun preprocessing and training |
| `ValueError: scores must have shape ...` | the scores do not match the class count | use the model's `predict_scores` |

## Decisions and limitations

- **Macro scores and low-support flags are always shown** because accuracy hides the rare classes.
- **Rare classes:** with 2 to 7 test rows, their precision, recall and F1 move in large steps (one row changes recall of Heartbleed by 50%). They are reported with their support and never summarised as reliable.
- **The attack-versus-normal view uses the multiclass predictions**; it is not a separately trained binary model.
- **No separate test file** in the roadmap; the checks above were run in the sandbox, and these functions are exercised by the evaluation script (File 53).