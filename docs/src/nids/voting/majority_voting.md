# `src/nids/voting/majority_voting.py`

**Location:** `DM_Project/src/nids/voting/majority_voting.py`
**Roadmap step:** File 53 (moved forward from File 60 at your request, so `06_evaluate.py` and the app share ONE vote implementation). The files after it shift by one; the total stays 65.

## Role
Takes the predicted class of every available model for each flow and returns the class with the most votes.

## The rule
- One predicted label per model per flow (`model.predict`); **one equal vote per model**; no weights; **no probability averaging** (scores are never used).
- **Ties** are broken deterministically (`settings.TIE_BREAK_RULE`, default `prefer_non_benign`): among tied classes an attack beats BENIGN; if several attacks are tied the lowest class code wins. Alternative rule `lowest_index`. Same input, same output, whatever the order of the models.
- **Missing models are reported**, never hidden: `VoteResult.missing` lists every expected model that did not vote and why (not loaded, failed while predicting, not supplied). `require_all=True` refuses to vote instead.

## Why it is needed
It is a mandatory part of the project, it adds the ensemble row to the comparison table (File 55) and it gives the final answer in the Streamlit app (File 63), so it must be one tested piece of code.

## Dependencies
`numpy`, `pandas`, `config.settings`, `config.model_params`, `nids.preprocessing.target_builder.LabelCodec`.

## Functions
| Function | What it does |
|---|---|
| `vote_labels(predictions, n_classes, benign_index=0, tie_break=None, expected_keys=None, missing=None, require_all=False)` | pure vote over `{key: int label array}` -> `VoteResult` |
| `collect_predictions(models, X, continue_on_error=True)` | `model.predict` for each fitted model; a model that raises is recorded in `failures` and the others continue |
| `vote_models(models, X, codec, ...)` | the two above in one call |
| `explain_vote(predictions, result, codec, row=0)` | for one flow: who voted for what, and the tally (used by the app) |
| `VoteResult` | `labels`, `counts` (rows x classes), `voters`, `missing`, `tied`, `top_votes`, `summary()` (tied and unanimous fractions) |
| `VotingError` | invalid input: no voters, non-integer or out-of-range codes, different lengths, unknown rule, missing models with `require_all` |

## How it was checked (sandbox)
Hand-worked cases (including a three-way tie and benign-versus-attack tie), a **brute-force reference** (plain Python loop, same rule) over 500 random flows with the 11 model keys: identical labels; order of models does not change the result; the missing-model report; every error path; zero rows; a model that raises; then the real-data command below on the sandbox's synthetic artifacts. A separate test file follows (File 54).

## Command (real data; the test set is NOT used)
Loads your 11 saved models, votes on a class-capped slice of 5,000 TRAINING rows and prints: the vote summary, how often each model agrees with the vote, and the explanation of one flow. About 30 to 60 seconds (KNN and SVC need a few seconds for 5,000 rows). PowerShell, from `DM_Project\`:

```powershell
python -c "import numpy as np; from config import model_params; from nids.persistence import model_store; from nids.voting import majority_voting as V; from nids.training.data_access import load_training_data, class_capped_indices; d = load_training_data(mmap=True); idx = np.sort(class_capped_indices(d.y, 5000)); X, y = np.asarray(d.X[idx]), d.y[idx]; models = {k: model_store.load_model(k, 'trained') for k in model_params.MODEL_KEYS}; preds, failed = V.collect_predictions(models, X); r = V.vote_labels(preds, d.n_classes, d.codec.benign_index, missing=failed); print(r.summary()); print({k: round(float((p == r.labels).mean()), 3) for k, p in preds.items()}); print('vote accuracy on these TRAINING rows (models saw them; optimistic):', round(float((r.labels == y).mean()), 4)); v, t = V.explain_vote(preds, r, d.codec, row=0); print(v.to_string(index=False)); print(t.to_string(index=False))"
```

**Expected output:** a summary with `n_voters: 11`, `missing: {}`, `tie_break_rule: 'prefer_non_benign'`, a `tied_fraction` and `unanimous_fraction` (I do not know the values; expect most rows not unanimous because AdaBoost, SGD and Bernoulli NB disagree often); a dictionary with each model's agreement with the vote (high for Random Forest, Decision Tree, XGBoost, low for AdaBoost and Bernoulli NB); a vote accuracy on training rows (the models have seen these rows, so it says nothing about quality: it only shows the plumbing works); then one flow's 11 votes and the tally.

## Common errors
| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError ... majority_voting` | placeholder empty | paste the code; Length must not be 0 |
| `StaleModelError` | preprocessing or model settings changed since training | rerun the training (`05_train.py`) |
| `ModelStoreError ... no saved trained model` | a model was not trained | train it; the vote may also run without it and will report it as missing |
| `VotingError ... different` | models predicted different row counts | pass the same X to all |

## Limits
- With an even number of voters (some models missing) ties are more frequent; the report shows the tied fraction.
- A vote is not guaranteed to beat its best member: if most voters are weak (AdaBoost, SGD and the Naive Bayes models are), they can outvote the strong ones. The comparison table will show whether it does.