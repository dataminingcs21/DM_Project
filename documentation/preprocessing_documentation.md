# Documentation de `src/preprocessing.py`

## 1. Rôle

Prépare les données avant tout apprentissage, en quatre étapes :

```text
8 fichiers CSV
   -> load_and_merge_datasets   (lecture + fusion + noms de colonnes nettoyés)
   -> clean_data                (infinis -> NaN, colonnes quasi nulles, lignes incomplètes)
   -> encode_labels             (texte du Label -> nombres)
   -> split_and_scale           (découpage train/test + standardisation)
```

Aucune fonction n'apprend d'un modèle ; seule la standardisation est ajustée, et uniquement sur le jeu d'entraînement.

---

## 2. `load_and_merge_datasets(dataset_dir='./dataset/')`

Lit les 8 fichiers de CIC-IDS2017 (lundi à vendredi), les concatène en un seul `DataFrame` et supprime les espaces autour des noms de colonnes (`" Label"` devient `"Label"`).

| Point | Détail |
|---|---|
| Entrée | dossier contenant les CSV (défaut relatif `./dataset/`, dépend du dossier courant) |
| Sortie | un `DataFrame` d'environ 2,83 millions de lignes sur 79 colonnes |
| Fichier absent | affiche seulement « Attention : Fichier non trouvé » et continue |
| Aucun fichier trouvé | lève `FileNotFoundError` |

Fichiers attendus (noms exacts) : `Monday-WorkingHours.pcap_ISCX.csv`, `Tuesday-WorkingHours.pcap_ISCX.csv`, `Wednesday-workingHours.pcap_ISCX.csv`, `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`, `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`, `Friday-WorkingHours-Morning.pcap_ISCX.csv`, `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`, `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`.

---

## 3. `clean_data(df, zero_threshold=99.0)`

1. Remplace `+inf` et `-inf` par `NaN` (divisions par zéro dans les débits).
2. Supprime les colonnes dont **au moins 99 %** des valeurs sont zéro (colonnes sans information).
3. Supprime les lignes contenant au moins un `NaN`.
4. Retourne une copie.

Remarques : le seuil est calculé sur **tout** le jeu de données, avant le découpage. Cette opération ne regarde pas les étiquettes et son effet est faible, mais strictement elle utilise aussi les lignes du futur jeu de test. Aucun doublon de ligne n'est supprimé.

---

## 4. `encode_labels(df, binary=True)`

Convertit la colonne `Label` (texte) en entiers selon la liste fixe de 15 classes :

| Id | Classe | Id | Classe |
|---|---|---|---|
| 0 | BENIGN | 8 | Web Attack Brute Force |
| 1 | FTP-Patator | 9 | Web Attack XSS |
| 2 | SSH-Patator | 10 | Web Attack Sql Injection |
| 3 | DoS slowloris | 11 | Infiltration |
| 4 | DoS Slowhttptest | 12 | Bot |
| 5 | DoS Hulk | 13 | DDoS |
| 6 | DoS GoldenEye | 14 | PortScan |
| 7 | Heartbleed | | |

La fonction `norm` nettoie le texte (caractères non alphabétiques remplacés par des espaces) afin que `Web Attack – Brute Force` et `Web Attack \x96 Brute Force` donnent la même clé.

| Paramètre | Sortie |
|---|---|
| `binary=False` | `y` = identifiant 0 à 14 (utilisé par `training.py`) |
| `binary=True` | `y = 1` si BENIGN, `0` si attaque |

Retourne `(X, y)` : `X` = toutes les colonnes sauf `Label`.

---

## 5. `split_and_scale(X, y, test_size=0.3, random_state=1)`

1. Découpage **stratifié** train/test (mêmes proportions de classes des deux côtés).
2. `StandardScaler` ajusté **uniquement** sur l'entraînement (pas de fuite de données), puis appliqué au test.
3. Ne garde que les colonnes numériques.

Retourne `X_train` et `X_test` (`DataFrame`, index recommencé à 0) et `y_train`, `y_test` (tableaux NumPy).

Remarques :

- `training.py` l'appelle avec les étiquettes multiclasses : le **même découpage** sert donc à toutes les tâches (binaire, multiclasse).
- Le scaler n'est **pas retourné** : il est perdu après l'appel. Or il faudra le même pour prédire sur de nouvelles données (voir `training_py.md`).

---

## 6. Problèmes détectés (vérifiés)

| # | Problème | Effet | Correction |
|---|---|---|---|
| 1 | `pd.read_csv(file_path)` lit en UTF-8. Les fichiers CIC-IDS2017 (notamment Thursday-WebAttacks) contiennent des octets non UTF-8 comme `0x96` dans « Web Attack – ... » | `UnicodeDecodeError` à la lecture (reproduit avec un octet `0x96`) | `pd.read_csv(file_path, encoding="latin-1")` |
| 2 | **Polarité** : avec `binary=True`, `1 = BENIGN` et `0 = attaque` (vérifié : `[BENIGN, DoS Hulk, BENIGN]` donne `[1, 0, 1]`) | Le « rappel » calculé par `training.py` est celui du trafic **normal**, pas des attaques | Convention du projet : attaque = 1. Remplacer par `y = (y != 0).astype(int)` ici **et** dans `_prepare_task` de `training.py` |
| 3 | Fichier manquant = simple avertissement | Le modèle s'entraîne sur un jeu incomplet sans erreur | Lever une erreur si `len(dfs) != 8` |
| 4 | Libellé inconnu : `label_map.get` renvoie `None`, puis `astype(int)` lève `TypeError: int() argument must be ... not 'NoneType'` (reproduit) | Message peu explicite | Tester `encoded_label.isna()` et nommer les libellés non reconnus |
| 5 | Pas de suppression des doublons | Un même flux peut se retrouver dans train et test | `df.drop_duplicates()` avant le découpage |
| 6 | Chemin par défaut relatif (`./dataset/`) | Dépend du dossier de lancement | Utiliser `C.DATASET_DIR` (c'est déjà ce que passe `training.py`) |

Correction minimale du problème 1 :

```python
dfs.append(pd.read_csv(file_path, encoding="latin-1"))
```

---

## 7. Dépendances

`os`, `re`, `numpy`, `pandas`, `scikit-learn` (`train_test_split`, `StandardScaler`). Utilisé par `training.py` (importé sous le nom `P`).