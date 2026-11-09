# Documentation de `src/feature_selection.py`

## 1. Rôle

Trois outils pour réduire le nombre de variables avant l'apprentissage :

| Fonction | Méthode | Supervisée ? | Utilisée par `training.py` ? |
|---|---|---|---|
| `compute_rf_feature_importances` | importance des variables par forêt aléatoire | oui | oui (`features.top_k`) |
| `apply_pca` | Analyse en Composantes Principales | non | oui (`features.pca` entier) |
| `apply_lda` | Analyse Discriminante Linéaire | oui | **non** |

Toutes doivent être ajustées sur le jeu d'**entraînement** uniquement (c'est ce que fait `load_xy`).

---

## 2. `compute_rf_feature_importances(X_train, y_train, n_estimators=50, random_state=0, save_fig_path=None)`

Entraîne une `RandomForestClassifier` (`n_jobs=-1`) et retourne un tableau classé des variables.

| Élément | Détail |
|---|---|
| Entrées | `X_train` (`DataFrame` standardisé), `y_train` |
| Sortie | `DataFrame` indexé par `feature`, colonne `importance` (arrondie à 3 décimales), triée décroissante |
| Graphique | si `save_fig_path` est donné : barres des 25 premières variables, enregistrées en 300 dpi |
| Affichage | imprime un titre et le chemin du graphique |

Utilisation typique :

```python
imp = compute_rf_feature_importances(X_tr, y_tr, n_estimators=50, random_state=42)
keep = imp.head(20).index.tolist()
X_tr, X_te = X_tr[keep], X_te[keep]
```

Points d'attention :

- Dans `training.py`, la forêt est entraînée avec les **15 classes** même pour la tâche binaire (`y_tr_raw`). Le classement mesure donc l'utilité pour distinguer toutes les attaques entre elles.
- Sur environ 2 millions de lignes, 50 arbres sont longs et gourmands en mémoire. Utilisez `--max-rows` pour les essais.
- L'arrondi à 3 décimales crée des **égalités** ; l'ordre entre variables à égalité est alors arbitraire, ce qui peut changer le contenu du « top k ».
- Le graphique demande `matplotlib`.

---

## 3. `apply_pca(X_train, X_test, n_components=15, save_variance_fig_path=None)`

Projette les données sur `n_components` axes de variance maximale.

| Élément | Détail |
|---|---|
| Ajustement | `fit_transform` sur `X_train`, `transform` sur `X_test` (pas de fuite) |
| Sortie | `(train_df, test_df, pca)` : deux `DataFrame` aux colonnes `PC1`, `PC2`, ... et l'objet `PCA` |
| Graphique | si `save_variance_fig_path` : barres de la variance expliquée par composante |
| Affichage | dimensions avant et après |

Points d'attention :

- `n_components` doit être un **entier** : il sert à construire les noms `PC1...PCn`. Un décimal (par exemple `0.95`) est traité à part dans `load_xy`, avec le `PCA` de scikit-learn.
- `n_components` ne doit pas dépasser le nombre de variables (le nombre de colonnes restantes après `top_k`, s'il est utilisé).
- Les colonnes obtenues sont des combinaisons de variables : elles ne sont plus interprétables telles quelles.
- L'objet `pca` est retourné mais `load_xy` ne le conserve pas : il faudra le sauvegarder pour prédire sur de nouvelles données.

---

## 4. `apply_lda(X_train, y_train, X_test, n_components=1)`

Projection supervisée qui maximise la séparation entre classes.

| Élément | Détail |
|---|---|
| Sortie | `(train_lda, test_lda, lda)` : tableaux NumPy et objet ajusté |
| Limite | `n_components` ≤ min(nombre de classes − 1, nombre de variables). Pour le binaire : 1 axe |
| Usage conseillé | après la PCA, pour un modèle linéaire |

`training.py` ne l'appelle pas, et la clé `features.lda` de `config.py` n'est jamais lue. Pour l'utiliser il faut l'ajouter dans `load_xy`.

---

## 5. Résumé des points d'attention

| # | Remarque | Conséquence |
|---|---|---|
| 1 | Forêt entraînée sur 15 classes, même en binaire | classement adapté au multiclasse |
| 2 | Importances arrondies à 3 décimales | ordre instable entre variables à égalité |
| 3 | Coût élevé sur tout le jeu de données | limiter les lignes pour les essais |
| 4 | `apply_pca` exige un entier | les décimaux passent par `load_xy` |
| 5 | Les transformateurs (PCA, sélection) ne sont pas sauvegardés avec le modèle | impossible de prédire sur de nouveaux flux sans les refaire |
| 6 | `apply_lda` inutilisée | option `features.lda` sans effet |

---

## 6. Dépendances

`numpy`, `pandas`, `matplotlib`, `scikit-learn` (`RandomForestClassifier`, `PCA`, `LinearDiscriminantAnalysis`). Importé par `training.py` sous le nom `F`.