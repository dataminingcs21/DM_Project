# Documentation de `src/training.py`

## 1. Rôle

Cœur de l'entraînement. Il charge et prépare les données, construit le modèle SVM, l'entraîne, calcule les métriques, enregistre les résultats et sait chercher les meilleurs hyperparamètres.

API publique :

| Fonction | Rôle | Retourne |
|---|---|---|
| `load_xy(cfg, rebuild=False)` | charge, nettoie, encode, découpe et standardise | un `dict` de tableaux NumPy |
| `run_experiment(cfg, data=None)` | entraîne un modèle et enregistre le résultat | `{"name", "row", "pipeline"}` |
| `tune(cfg, search_space, n_iter, data, sample, out_file)` | recherche aléatoire d'hyperparamètres | `(meilleurs_paramètres, tableau_des_résultats)` |

Fonctions internes : `_stratified_sample`, `_prepare_task`, `_build_estimator`, `_apply_imbalance`, `_compute_metrics`, `_decision_scores`.

Flux global :

```text
cfg --> load_xy --> {X_train, X_test, y_train, y_test, feature_names}
                         |
                         +--> run_experiment : _build_estimator -> _apply_imbalance -> fit -> predict
                         |                     -> métriques -> models/<run>.joblib + results/tables/experiments.csv
                         +--> tune : RandomizedSearchCV -> configs/best_svm_params.json
```

---

## 2. `load_xy(cfg, rebuild=False)`

Étapes dans l'ordre :

1. **Cache.** Si `cache/clean_cache.pkl` existe et `rebuild=False`, il est chargé. Sinon : lecture des CSV (`cfg["dataset_dir"]`), nettoyage, puis, si `cfg["max_rows"]` est défini, échantillon stratifié, et enregistrement dans le cache.
2. **Encodage.** `encode_labels(df, binary=False)` : étiquettes de 0 à 14.
3. **Découpage et standardisation** : `split_and_scale(test_size, random_state=seed)`.
4. **`top_k`** (si défini) : importance par forêt aléatoire sur l'entraînement, garde les `k` meilleures colonnes (train et test).
5. **`pca`** (si défini) : un décimal entre 0 et 1 = variance à conserver (PCA scikit-learn) ; sinon un entier = nombre de composantes (`apply_pca`). `top_k` est appliqué **avant** la PCA.
6. **Cible de la tâche** : `_prepare_task`.
7. Retourne :

```python
{"X_train": float32, "X_test": float32, "y_train": ndarray, "y_test": ndarray, "feature_names": [...]}
```

### `_stratified_sample(df, n, seed)`
Échantillon proportionnel par classe, avec **au moins 2 lignes par classe** (indispensable pour le découpage stratifié). Une classe qui ne peut pas atteindre 2 lignes est retirée. Les échantillons sont tirés avec la graine du run.

### `_prepare_task(y_raw, task)`
- `binary` : `(y_raw == 0)` : **1 = BENIGN, 0 = attaque**.
- `multiclass` et `family` : étiquettes inchangées. La tâche `family` n'est donc pas implémentée (elle se comporte comme `multiclass`).

---

## 3. `_build_estimator(cfg)`

| `model.name` | Objet créé |
|---|---|
| `linear_svm` | `LinearSVC(max_iter=5000, **params)` |
| `svm_rbf` | `SVC(kernel="rbf", probability=False, **params)` |
| `svm_nystroem` | `Pipeline[Nystroem(n_components=300 par défaut, gamma), LinearSVC]` ; `n_components` et `gamma` sont lus dans `params`, le reste va à `LinearSVC` |

Autre nom : `ValueError("Unknown model")`.

## 4. `_apply_imbalance(cfg, estimator)`

| Mode | Effet |
|---|---|
| `none` | rien |
| `class_weight` | `estimator.set_params(class_weight="balanced")` (ignoré si l'appel échoue) |
| `undersample` | `ImbPipeline([("sampler", RandomUnderSampler), ("clf", estimateur)])` |
| `smote` | `ImbPipeline([("sampler", SMOTE), ("clf", estimateur)])` |

Les échantillonneurs n'agissent qu'à l'`fit`. Sans `imbalanced-learn`, le mode est ignoré avec un message.

---

## 5. Métriques

`_compute_metrics(y_true, y_pred, y_score, pos_label)` (tâche binaire) : `recall`, `precision`, `f1` de la classe `pos_label=1`, `macro_f1`, `pr_auc` (`average_precision_score`) et `false_alarm_rate` = flux de classe 1 prédits 0 / nombre de flux de classe 1.

`_decision_scores` : `predict_proba` si disponible (dernière colonne), sinon `decision_function` (cas des SVM), sinon la prédiction.

Pour les tâches non binaires, `recall`, `precision` et `f1` sont des **moyennes macro** ; `pr_auc` et `false_alarm_rate` valent `NaN`.

---

## 6. `run_experiment(cfg, data=None)`

1. Charge les données si besoin (`load_xy`).
2. Construit l'estimateur et applique le traitement du déséquilibre.
3. `fit` sur le jeu d'entraînement, chronométré ; `predict` sur le test, chronométré.
4. Calcule les scores et les métriques.
5. Construit une ligne de résultats :

| Colonne | Contenu |
|---|---|
| `run`, `model`, `task`, `imbalance` | identification |
| `n_train`, `n_test`, `n_features_used` | tailles |
| `fit_seconds` | durée de l'entraînement en secondes |
| `predict_seconds_per_1k` | durée de prédiction (voir le point 9 ci-dessous) |
| `recall`, `precision`, `f1`, `macro_f1`, `pr_auc`, `false_alarm_rate` | métriques |

6. Sauvegarde `models/<run_name>.joblib` avec `{"estimator", "config", "features"}`.
7. Ajoute la ligne à `results/tables/experiments.csv` (en-tête écrit à la création du fichier).

Il n'y a **pas** de validation croisée dans ce `run_experiment`, ni de matrice de confusion, ni de tableau par classe.

---

## 7. `tune(cfg, search_space, n_iter, data, sample, out_file)`

1. Prend un sous-échantillon **aléatoire** de `sample` lignes d'entraînement (non stratifié).
2. Construit l'estimateur (avec `_apply_imbalance`).
3. `RandomizedSearchCV` avec `StratifiedKFold(cv.n_splits)`, score `cv.scoring`, `n_jobs=-1`, `refit=False`.
4. Écrit les meilleurs paramètres dans `out_file` (JSON).
5. Retourne `(best, DataFrame des résultats)`.

Les noms de paramètres doivent correspondre à **l'objet réel** cherché (voir le point 4 ci-dessous).

---

## 8. Utilisation directe en Python

```python
from src import config as C
from src.training import load_xy, run_experiment

cfg = C.load_config()
cfg["max_rows"] = 50_000
cfg["model"] = {"name": "svm_rbf", "params": {"C": 10, "gamma": 0.01}}
cfg["run_name"] = "svm_rbf_essai"
data = load_xy(cfg)
res = run_experiment(cfg, data=data)
print(res["row"])
```

---

## 9. Problèmes détectés (vérifiés en exécutant vos fichiers)

| # | Problème | Effet réel | Correction proposée |
|---|---|---|---|
| 1 | **Polarité inversée** : `_prepare_task` donne `1 = BENIGN` | `recall`, `precision`, `f1`, `pr_auc` mesurent le trafic **normal**. Un SVM qui rate beaucoup d'attaques peut afficher un « rappel » élevé | `return (y_raw != 0).astype(int)` ; et dans `_compute_metrics`, le taux de fausses alertes devient `((y_pred == 1) & (y_true == 0)).sum() / (y_true == 0).sum()` |
| 2 | **`train_max_rows` et `cv_max_rows` ne sont pas utilisés** | `--train-rows` de `train_svm.py` n'a aucun effet : le SVM `rbf` s'entraîne sur **toutes** les lignes (environ 2 millions), ce qui est impraticable | dans `run_experiment`, après `X_tr, y_tr = ...` : sous-échantillonner si `cfg["train_max_rows"]` (échantillon stratifié) |
| 3 | **`max_rows` est appliqué puis mis en cache** : le cache `clean_cache.pkl` contient alors un jeu **réduit** | après un essai avec `--max-rows 50000`, un run suivant « complet » recharge silencieusement les 50 000 lignes tant qu'on n'ajoute pas `--rebuild` | mettre en cache le jeu complet nettoyé, et appliquer `max_rows` **après** le chargement du cache |
| 4 | **Noms de paramètres de `tune`** : `search_space` de `train_svm.py` utilise `clf__C`, mais `clf` n'existe que si `imbalance` vaut `undersample` ou `smote` | avec `none` ou `class_weight`, `--tune` plante : `Invalid parameter 'clf' for estimator LinearSVC` (reproduit pour les 3 variantes) | préfixer par `clf__` seulement avec un échantillonneur (correction dans `train_svm_py.md`) |
| 5 | **`params.pop(...)` modifie la configuration** dans `_build_estimator` (variante `svm_nystroem`) | `n_components` et `gamma` sont **supprimés de `cfg`** après le premier appel (reproduit : `{}`). Avec `--tune`, l'entraînement final oublie `--n-components 500` et `--gamma` | `params = dict(cfg["model"].get("params") or {})` au début de la fonction |
| 6 | **`class_weight` ignoré avec `svm_nystroem`** (le `try/except` avale l'erreur) | `--imbalance class_weight` n'a aucun effet : `linearsvc__class_weight` reste `None` (reproduit) | pour `svm_nystroem`, utiliser `set_params(linearsvc__class_weight="balanced")` |
| 7 | **`cv.scoring = "f1"` en multiclasse** | tous les scores sont `NaN` (reproduit) et la « meilleure » combinaison est arbitraire | `scoring = "f1" if task == "binary" else "f1_macro"` |
| 8 | **Le modèle sauvegardé est incomplet** : `{"estimator", "config", "features"}` | le `StandardScaler` (et `top_k`, PCA) n'est pas conservé : impossible de refaire une prédiction correcte sur de nouveaux flux. | enregistrer aussi le scaler et les transformateurs, ou mettre scaler + sélection + modèle dans un seul `Pipeline` |
| 9 | **Unité de `predict_seconds_per_1k`** : `pred_s * 1000 / n * 1000` | la valeur est en **millisecondes** pour 1 000 flux, pas en secondes | enlever un `* 1000.0` |
| 10 | **`SMOTE` sans plafond** sur les millions de lignes | mémoire et temps très élevés | réduire d'abord (`undersample`), ou utiliser `--max-rows` |
| 11 | **Pas de validation croisée** dans `run_experiment`, **pas de matrice de confusion** ni de tableau par classe | seul `experiments.csv` est produit | à ajouter si nécessaire pour le rapport |
| 12 | `tune` : la sélection de variables (`top_k`) et la standardisation sont faites **avant** la validation croisée | légère fuite dans les scores de validation croisée (jamais dans le jeu de test) | mettre ces étapes dans un `Pipeline` |

### Priorités

1. Corriger **1** (polarité) : sinon toutes les métriques du rapport sont inversées.
2. Corriger **2** et **3** : sinon `rbf` ne peut pas tourner sur les vraies données et les runs « complets » peuvent utiliser un petit jeu.
3. Corriger **4**, **5**, **7** pour pouvoir utiliser `--tune`.

---

## 10. Dépendances

`numpy`, `pandas`, `joblib`, `scikit-learn`, `imbalanced-learn` (facultatif), et les modules du projet `config` (`C`), `preprocessing` (`P`), `feature_selection` (`F`).