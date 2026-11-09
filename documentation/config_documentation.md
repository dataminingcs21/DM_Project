# Documentation de `src/config.py`

## 1. Rôle

Point central de la configuration du projet. Il définit :

- les **chemins** de tous les dossiers (données, modèles, résultats, cache) ;
- la **configuration par défaut** d'un run (graine, tâche, modèle, options de sélection de variables, validation croisée) ;
- deux fonctions : `load_config()` et `ensure_dirs()`.

Les autres fichiers l'importent avec `from . import config as C`.

---

## 2. Chemins

```python
ROOT = Path(__file__).resolve().parent.parent
```

`ROOT` est le dossier qui **contient `src/`** (chez vous : `...\DM_Project\DM_Project`). Tous les autres chemins en sont déduits :

| Constante | Dossier | Utilisation |
|---|---|---|
| `ROOT` | racine du projet | base de tous les chemins ; `train_svm.py` y écrit `configs/best_svm_params.json` |
| `DATASET_DIR` | `dataset/` | les 8 fichiers CSV de CIC-IDS2017 |
| `CONFIGS_DIR` | `configs/` | fichiers de configuration et meilleurs paramètres |
| `MODELS_DIR` | `models/` | modèles `.joblib` sauvegardés |
| `RESULTS_DIR` | `results/` | dossier parent des résultats |
| `TABLES_DIR` | `results/tables/` | `experiments.csv` |
| `REPORTS_DIR` | `reports/` | figures (prévu, rien n'y écrit pour l'instant) |
| `CACHE_DIR` | `cache/` | jeu de données nettoyé `clean_cache.pkl` |

---

## 3. Configuration par défaut (`_DEFAULT_CONFIG`)

| Clé | Valeur par défaut | Signification | Utilisée par |
|---|---|---|---|
| `seed` | `42` | graine aléatoire (découpage, modèles, échantillonnage) | `training.py` |
| `task` | `"binary"` | `binary` / `multiclass` / `family` | `training.py` |
| `imbalance` | `"none"` | `none`, `class_weight`, `undersample`, `smote` | `training.py` |
| `max_rows` | `None` | plafond global de lignes (débogage) | `training.py` (voir les points d'attention) |
| `train_max_rows` | `None` | plafond sur les lignes d'entraînement | **aucun fichier pour l'instant** |
| `cv_max_rows` | `None` | plafond de lignes pour la validation croisée | **aucun fichier pour l'instant** |
| `test_size` | `0.3` | part du jeu de test | `training.py` |
| `dataset_dir` | chemin de `dataset/` (texte) | où lire les CSV | `training.py` |
| `model.name` | `"linear_svm"` | `linear_svm`, `svm_nystroem` ou `svm_rbf` | `training.py` |
| `model.params` | `{}` | hyperparamètres du modèle (`C`, `gamma`, `n_components`) | `training.py` |
| `features.top_k` | `None` | garder les k variables les plus importantes | `training.py` |
| `features.pca` | `None` | PCA : décimal = variance conservée, entier = nombre de composantes | `training.py` |
| `features.lda` | `None` | LDA | **non appliquée** par `training.py` |
| `preprocessing.zero_threshold` | `99.0` | une colonne est supprimée si elle contient au moins 99 % de zéros | `preprocessing.py` |
| `cv.n_splits` | `3` | nombre de plis de validation croisée | `tune` |
| `cv.scoring` | `"f1"` | score optimisé par `tune` | `tune` |
| `run_name` | `"run"` | nom du run (fichier modèle et ligne de résultats) | `training.py`, `train_svm.py` |

---

## 4. Fonctions

### `load_config() -> dict`

Retourne une **copie profonde** de la configuration par défaut. On peut donc la modifier librement sans changer les valeurs par défaut :

```python
cfg = C.load_config()
cfg["model"] = {"name": "svm_rbf", "params": {"C": 10}}
```

### `ensure_dirs() -> None`

Crée (si besoin) `models/`, `results/`, `results/tables/`, `reports/`, `cache/` et `configs/`. À appeler avant d'écrire un résultat. Ne crée **pas** `dataset/` (c'est à vous d'y mettre les CSV).

---

## 5. Points d'attention

1. **`cv.scoring = "f1"` ne convient pas à la tâche `multiclass`.** Le score `f1` suppose deux classes ; sur 15 classes tous les scores deviennent `NaN` et la recherche d'hyperparamètres choisit une combinaison au hasard (vérifié). Pour les tâches autres que `binary`, utilisez `"f1_macro"`.
2. **`train_max_rows` et `cv_max_rows` n'ont aucun effet** avec le `training.py` actuel : ils sont définis ici mais jamais lus. Voir `training_py.md`.
3. **`features.lda` n'est jamais appliqué** par `training.py`.
4. **`zero_threshold` est en pourcentage** (99.0), pas en fraction (0.99).
5. **Pas de lecture de fichier YAML** ni de surcharge en ligne de commande : on modifie le dictionnaire en Python, comme le fait `train_svm.py`.
6. **`REPORTS_DIR` est créé mais rien n'y est écrit** (pas de matrice de confusion automatique).
7. `ROOT` dépend de l'emplacement de `config.py` : ne déplacez pas ce fichier hors de `src/`.

---

## 6. Dépendances

Utilise seulement la bibliothèque standard (`pathlib`, `copy`). Il est importé par `preprocessing`/`training`/`train_svm`.