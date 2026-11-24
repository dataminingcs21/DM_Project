# Network intrusion detection on CIC-IDS2017

A modular Python system that classifies network flows into **BENIGN or one of 14 attack types** with **eleven supervised models**, combines them by **majority vote**, and serves the result in a **Streamlit app** where you can type the features of a flow, load a real example, or upload a CSV file.

There is no clustering and no association-rule mining. Every number in this file comes from a report the scripts wrote on the real data; where a figure is not stated here, the named report has it.

## Read this first: what the results do and do not mean

1. **The split is random, so the scores are optimistic.** The 80/20 split is stratified but random. Flows of the same attack session (and near-identical flows) can land on both sides. The scores say how well the models recognise flows like those they trained on, not how they would do on another network or a new capture. Every attack class sits in one source file of the dataset, so a split by file would test attacks the models have never seen; that harder test was not run.
2. **Three classes are too small to judge.** After cleaning, the test set holds **2 Heartbleed, 4 Web Attack - Sql Injection and 7 Infiltration** flows. One flow more or less moves their scores by tens of percent. Scores for these classes, and the all-class macro F1 that includes them, are indicative only. The app and the reports show the test support next to every per-class score.
3. **Three models were trained on a class-capped subsample** (KNN 100,000 rows, SVC 50,000, Gradient Boosting 100,000 of 2,017,889). Their class mix differs from real traffic, so their predictions are shifted toward the small classes.
4. **Tuning used even smaller samples.** Hyperparameters were searched on class-capped samples of at most 100,000 training rows (KNN 50,000; SVC and Gradient Boosting 30,000) and the winners were refitted on the sizes above. Several winners sit at the edge of their grid (Gaussian NB `var_smoothing=1e-9`, Bernoulli NB `alpha=0.1`, Logistic Regression `C=10`, SGD `alpha=1e-6`, AdaBoost depth 4), so a wider grid might do better. The grids were not widened, because the selection rule was fixed beforehand.
5. **The majority vote is not clearly better than its best members.** See "Results". The vote of the final model set has the lowest all-class macro F1 of the three votes that were computed.
6. **`Destination Port` is a model feature.** Attack classes tie to specific ports in this lab capture, so the models may partly learn the lab setup. The planned comparison without this feature was **not run**.
7. **A prediction is a vote, not proof.** The input check catches impossible values (text, negatives where impossible, a port above 65535) and warns about values outside the training range. It cannot tell whether a combination of values is a realistic flow.

## Data

- **Dataset:** CIC-IDS2017 (Canadian Institute for Cybersecurity, University of New Brunswick). It is **not included**. Put the eight CSV files in `data\raw\`.
- **As found:** 2,830,743 rows, 78 feature columns, 15 classes, BENIGN 80.3%. The smallest raw classes are Heartbleed (11), Web Attack - Sql Injection (21) and Infiltration (36).
- **Cleaning** (`03_clean_data.py`): removed 308,381 exact duplicates (10.89%) and 12 identical copy columns; fixed the three web-attack labels that contain a damaged character; set impossible negatives (2,926 rows, 93% BENIGN) to missing; infinite values (2,867 rows, all with a flow duration of 0) are treated as missing. Result: **2,522,362 rows, 66 feature columns**. 698 feature vectors carry two different labels (1,396 rows, 0.055%); they were kept and are an unavoidable error floor.
- **Preprocessing** (`04_preprocess.py`): stratified split, seed 42, **2,017,889 train / 504,473 test**. The pipeline (median imputation, signed log1p, removal of constant and correlated columns at 0.95, standard scaling) is fitted on the training rows only and keeps **40 of 66 features**.

## The eleven models

K-Nearest Neighbours, Decision Tree, Random Forest, Gaussian Naive Bayes, SVC (RBF), Bernoulli Naive Bayes, Logistic Regression, SGD Classifier, AdaBoost, Gradient Boosting and XGBoost. Each exists in a **baseline** version (default-style settings, unweighted) and a **tuned** version (GridSearchCV, 5-fold, score = prior-corrected macro F1, class weighting chosen by cross-validation where the model supports it).

**Which version the final system uses** was fixed by a rule, before any tuned model was scored on the test set, and never changed afterwards: *use the tuned model if its mean paired improvement over the baseline across the tuning folds is larger than the standard deviation of the paired fold differences; otherwise use the baseline.* This is a pragmatic threshold, not a significance test.

| Final set uses | Models |
|---|---|
| Tuned (7) | KNN, Decision Tree, SVC, Bernoulli NB, SGD, AdaBoost, Gradient Boosting |
| Baseline (4) | Random Forest, XGBoost, Logistic Regression, Gaussian NB |

**Majority vote:** one equal vote per model, no weights. On a tie the class that is not BENIGN wins (`prefer_non_benign`). A weighted vote (weights from training-data cross-validation only) is a possible later variant; it was not built.

## Results (held-out test set, 504,473 flows)

Macro F1 of the three votes (all classes, and the classes with enough test flows to be reliable):

| Vote | Macro F1, all classes | Macro F1, reliable classes |
|---|---:|---:|
| Baseline models only | 0.9095 | 0.9096 |
| Tuned models only | **0.9142** | **0.9155** |
| Final set (evidence rule) | 0.9054 | 0.9138 |

The final set has the lowest all-class macro F1. The differences are small and the all-class figure is moved by the three tiny classes, so none of the three votes can be called clearly better. The final set was kept because its rule was fixed beforehand, not because it won.

Test macro F1 of each **tuned** version (for the four models where the final set uses the baseline, this number is shown for reporting only and is not what the final set uses): KNN 0.6831, Decision Tree 0.8979, Random Forest 0.9158, Gaussian NB 0.4790, SVC 0.7078, Bernoulli NB 0.3869, Logistic Regression 0.7138, SGD 0.6523, AdaBoost 0.7908, Gradient Boosting 0.6101, XGBoost 0.9183. The tree ensembles are far ahead; several single models are well below the vote and drag the equal-weight vote down, which is the reason the weighted vote is worth trying later.

Main mistakes of the final vote:

- Web Attack - XSS is labelled Web Attack - Brute Force: 90 flows (69.23% of XSS). The reverse, Brute Force labelled XSS: 38 flows (12.93%). These two classes are close in the feature space.
- Bot labelled BENIGN: 55 flows (14.07% of Bot), that is, missed attacks.
- BENIGN labelled PortScan 183, DoS Hulk 89, Bot 58 (false alarms; BENIGN is the large class, so the shares are small).

Full tables, per-class scores, confusion matrices and figures are in `reports\evaluation\` (`evaluation_report_tuned.md`, `model_comparison_tuned.md`) and `reports\optimization\tuning_report.md`.

## Install

Windows 11, PowerShell, Python 3.13 (the pinned versions in `requirements.txt` were tested on it).

```powershell
python -m venv C:\venvs\nids
C:\venvs\nids\Scripts\Activate.ps1
cd <project folder>
pip install -r requirements.txt
pip install -e .
```

XGBoost uses a GPU when one is usable and the CPU otherwise.

## Run the pipeline

From the project folder with the environment active. Each script prints what it did and checks its own output; each has a document in `docs\scripts\`.

```powershell
python scripts\01_inspect_dataset.py      # facts about the raw files
python scripts\02_run_eda.py              # exploratory analysis
python scripts\03_clean_data.py           # cleaning -> data\cleaned\
python scripts\04_preprocess.py           # split + pipeline -> data\preprocessed\
python scripts\05_train.py                # the 11 baseline models (about 18 minutes in the reference run)
python scripts\06_evaluate.py             # baseline evaluation (cross-validation is long; see --cv-sample-size, --skip-cv)
python scripts\07_optimize.py             # tuning and refit (hours; --dry-run shows the plan first)
python scripts\06_evaluate.py --stage tuned   # scores the tuned models once, three votes, the final set
```

Outputs go to `data\`, `artifacts\models\` (trained and tuned), `reports\` and `logs\`; these folders are not part of the source and are rebuilt by the scripts.

## Use the app

```powershell
streamlit run app\streamlit_app.py
```

- **Single flow:** type the 40 feature values (grouped by the first word of the name), or load a real example of any class saved from the test split, then press **Predict**. The page shows the winning class, how many models agree, every model's vote, the tally and what the test set says about that class.
- **Upload a CSV file:** one flow per row, feature names as headers (a `Label` column is ignored; at most 50,000 rows). Rows with an invalid value are refused with the reason; the others are predicted; the result can be downloaded.
- **About the models:** which version of each model is used and why, the test scores, the known mistakes and the limits.
- A missing value is refused by default; the sidebar can switch to "fill with the training median", and the page then says so.

The app loads every model it can; a model that cannot be loaded is named on the page and the vote continues with the others.

## Compare predictions with true labels

```powershell
python scripts\08_make_comparison_files.py                 # 300 random held-out flows
python scripts\08_make_comparison_files.py --per-class 20  # every class gets flows
```

It writes `reports\comparison\comparison_input.csv` (upload it in the app), `comparison_answers.csv` (the dataset's true labels) and `comparison.csv` (true and predicted side by side). The flows are real rows of the held-out test split of CIC-IDS2017, so they are unseen by the models but **not** from another network.

## Tests

```powershell
python -m pytest -q
```

Reference run: **664 passed, 1 skipped** (the GPU test is skipped on a machine with a usable GPU) and two expected convergence warnings from the registry tests. Tests use synthetic data only and never touch the real `data\`, `artifacts\` or `logs\` folders. The app is tested headlessly with Streamlit's `AppTest`.

## Project layout

```
config\       settings.py (paths, seed, split, rules), model_params.py (grids, display names)
src\nids\     data, cleaning, eda, preprocessing, models, training, evaluation, optimization,
              voting, persistence, prediction, utils
scripts\      01 ... 08 (the pipeline and the comparison files)
app\          streamlit_app.py, ui_components.py
tests\        pytest files
docs\         one .md per file (mirrors the layout) and docs\project\ for decisions
```

The decisions made after each checkpoint, with their reasons, are in `docs\project\checkpoint_1_decisions.md` to `checkpoint_5_decisions.md`.

## Not done

- The comparison without `Destination Port`.
- A split by source file or attack campaign (a harder, more honest test).
- A wider search around the edge-of-grid winners.
- A weighted vote.
- Any test on traffic from another network. No such data was available.