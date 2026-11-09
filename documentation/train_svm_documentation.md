# Documentation de `src/train_svm.py`

> Cette version **remplace** la documentation précédente : elle décrit ce que fait le script avec **vos** fichiers `config.py` et `training.py`.

## 1. Rôle

Script en ligne de commande qui entraîne et évalue un **SVM (Support Vector Machine)** sur CIC-IDS2017. Il ne contient pas d'apprentissage : il construit la configuration, puis appelle `load_xy`, `tune` et `run_experiment` de `training.py`.

## 2. Les trois variantes

| Variante (`--variant`) | Modèle (`model.name`) | Principe |
|---|---|---|
| `linear` (défaut) | `linear_svm` | `LinearSVC`, rapide |
| `nystroem` | `svm_nystroem` | noyau RBF approché (`Nystroem` + `LinearSVC`) |
| `rbf` | `svm_rbf` | `SVC` à noyau RBF exact, très lent sur de gros volumes |

## 3. Lancement

Toujours comme module, depuis le dossier qui contient `src/` :

```powershell
cd "C:\Users\LAPTOP SPIRIT\Documents\DM_Project\DM_Project"
python -m src.train_svm --max-rows 50000
```

Exemples :

```text
python -m src.train_svm --max-rows 50000
python -m src.train_svm --variant linear --imbalance class_weight
python -m src.train_svm --variant nystroem --n-components 500 --C 1
python -m src.train_svm --variant rbf --C 10 --gamma 0.01 --max-rows 50000
python -m src.train_svm --variant linear --imbalance undersample --tune
```

## 4. Arguments

| Argument | Valeurs | Défaut | Effet avec **votre** `training.py` |
|---|---|---|---|
| `--variant` | `linear`, `nystroem`, `rbf` | `linear` | choisit le modèle |
| `--task` | `binary`, `multiclass`, `family` | `binary` | `family` se comporte comme `multiclass` (non implémentée) |
| `--C` | décimal | défaut sklearn | transmis au SVM |
| `--gamma` | décimal | auto | `rbf` et `nystroem` seulement |
| `--n-components` | entier | 300 | `nystroem` seulement |
| `--imbalance` | `none`, `class_weight`, `undersample`, `smote` | `none` | voir le point 6 |
| `--train-rows` | entier | 30 000 pour `rbf` | **sans effet** : `training.py` ne lit pas `train_max_rows` |
| `--max-rows` | entier | aucun | échantillon du jeu de données, **uniquement quand le cache est créé** (voir le point 6) |
| `--top-k` | entier | aucun | garde les k meilleures variables |
| `--pca` | décimal | aucun | `0.95` = 95 % de variance ; une valeur ≥ 1 devient un nombre de composantes |
| `--tune` | drapeau | non | recherche aléatoire avant l'ajustement final |
| `--n-iter` | entier | 6 | nombre de combinaisons essayées |
| `--seed` | entier | 42 | graine |
| `--rebuild` | drapeau | non | refait le cache de nettoyage |

## 5. Fonctionnement

### `build_config(...)`
1. `cfg = C.load_config()` (configuration par défaut de `config.py`).
2. Remplit `cfg["model"]` (`name` et `params` : `C`, `gamma`, `n_components`).
3. Fixe `task`, `imbalance`, `max_rows`, `train_max_rows` (défaut 30 000 pour `rbf`) et `cv_max_rows`.
4. Fixe `features.top_k` et `features.pca` à partir des arguments (l'ancienne valeur est donc écrasée).
5. Donne un nom au run : `svm_<variante>_<tâche>_<déséquilibre>` (+ `_C<valeur>`). Ce nom est celui du fichier `.joblib` ; un même nom **écrase** le modèle précédent, alors que la ligne du CSV est **ajoutée**.

### `search_space(variant)`
| Variante | Paramètres cherchés |
|---|---|
| `linear` | `clf__C` ∈ {0.01, 0.1, 1, 10, 100} |
| `rbf` | `clf__C` × `clf__gamma` ∈ {0.001, 0.01, 0.1, 1} |
| `nystroem` | `clf__linearsvc__C` × `clf__nystroem__gamma` |

### `main()`
1. Lit les arguments, crée les dossiers (`C.ensure_dirs()`).
2. Construit la configuration, charge les données (`load_xy`).
3. Si `--tune` : `tune(...)` sur au plus 50 000 lignes, affiche les meilleurs paramètres et les 5 meilleures lignes, écrit `configs/best_svm_params.json`, puis recopie les meilleurs paramètres dans la configuration.
4. `run_experiment(cfg, data=data)`.
5. Affiche les métriques (`run`, `n_train`, `n_test`, `n_features_used`, `recall`, `precision`, `f1`, `pr_auc`, `macro_f1`, `false_alarm_rate`, `fit_seconds`, `predict_seconds_per_1k`) et le chemin du modèle.

## 6. Ce que vous obtenez

| Fichier | Contenu |
|---|---|
| `results/tables/experiments.csv` | une ligne par exécution |
| `models/<run>.joblib` | `{"estimator", "config", "features"}` (sans le scaler : voir `training_py.md`) |
| `configs/best_svm_params.json` | seulement avec `--tune` |

Pas de matrice de confusion ni de tableau par classe avec ce `training.py`.

## 7. Problèmes détectés (vérifiés)

| # | Problème | Effet | Contournement ou correction |
|---|---|---|---|
| 1 | **`--tune` plante** sauf avec `--imbalance undersample` ou `smote` : `Invalid parameter 'clf' for estimator ...` | les noms `clf__...` n'existent que si l'estimateur est dans un `ImbPipeline` | voir la correction ci-dessous |
| 2 | **`--train-rows` n'a aucun effet** | `rbf` s'entraîne sur toutes les lignes d'entraînement, donc pratiquement jamais | corriger `training.py` (point 2 de `training_py.md`) ou utiliser `--max-rows` |
| 3 | **`--max-rows` est mis en cache** | un run « complet » ultérieur recharge le petit jeu tant qu'on ne met pas `--rebuild` | utilisez `--rebuild` en changeant `--max-rows`, ou corrigez `training.py` |
| 4 | **`--imbalance class_weight` sans effet avec `nystroem`** | `class_weight` reste `None` sans message | utilisez `undersample`, ou corrigez `_apply_imbalance` |
| 5 | **`--task multiclass --tune`** : scores `NaN` | choix arbitraire des paramètres | utiliser `f1_macro` dans `config.py` / `tune` |
| 6 | **Avec `--tune` et `nystroem`**, `--n-components` et `--gamma` sont oubliés pour l'ajustement final (`_build_estimator` les supprime de la configuration) | l'ajustement final utilise 300 composantes | corriger `_build_estimator` (point 5 de `training_py.md`) |
| 7 | **Polarité** : `1 = BENIGN` dans `training.py` | `recall`/`precision`/`f1`/`pr_auc` affichés sont ceux du trafic **normal** | corriger la polarité avant d'interpréter les résultats |

Correction du problème 1 (dans `train_svm.py`) :

```python
def search_space(variant, imbalance="none"):
    p = "clf__" if imbalance in ("undersample", "smote") else ""
    if variant == "linear":
        return {p + "C": C_GRID}
    if variant == "rbf":
        return {p + "C": C_GRID, p + "gamma": GAMMA_GRID}
    return {p + "linearsvc__C": C_GRID, p + "nystroem__gamma": GAMMA_GRID}
```

et l'appel : `tune(cfg, search_space(a.variant, a.imbalance), ...)`.

## 8. Ordre de travail conseillé

1. Corriger la polarité et l'encodage (`preprocessing_py.md` et `training_py.md`).
2. Lancer un essai rapide : `python -m src.train_svm --max-rows 50000 --rebuild`.
3. Comparer `linear`, `nystroem` et `rbf` (avec `train_max_rows` corrigé) sur la même configuration.
4. Régler C (et gamma) avec `--tune` une fois les noms de paramètres corrigés.

## 9. Dépendances

`argparse`, `config.py` (`C`), `training.py` (`load_xy`, `run_experiment`, `tune`). Voir les autres fichiers de documentation pour chaque module.