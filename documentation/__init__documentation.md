# Documentation de `src/__init__.py`

## 1. Rôle

Fichier **vide**. Sa seule présence indique à Python que le dossier `src/` est un **paquet** (*package*).

Sans lui :

- les imports relatifs des autres fichiers (`from . import config as C`, `from .training import ...`) ne fonctionnent pas ;
- la commande `python -m src.train_svm` ne trouve pas le module ;
- l'éditeur (Pylance) signale `Import "." could not be resolved`.

## 2. Contenu

Aucun. Il ne faut **rien** y écrire : un fichier vide est suffisant et conforme.

## 3. Ce qu'il permet dans ce projet

| Commande / import | Fonctionne grâce à `__init__.py` |
|---|---|
| `python -m src.train_svm --max-rows 50000` | oui |
| `from . import config as C` (dans `training.py`, `train_svm.py`) | oui |
| `from .training import load_xy, run_experiment, tune` | oui |
| `from . import preprocessing as P` | oui |

## 4. Règles d'utilisation

1. Gardez-le dans `src/`, au même niveau que `config.py`, `training.py`, etc.
2. Lancez toujours les scripts **comme modules**, depuis le dossier qui contient `src/` :
   ```powershell
   cd "C:\Users\LAPTOP SPIRIT\Documents\DM_Project\DM_Project"
   python -m src.train_svm --max-rows 50000
   ```
   Exécuter `python src\train_svm.py` directement provoque `ImportError: attempted relative import with no known parent package`.
3. Dans Git, un fichier vide est bien suivi (il n'est pas ignoré). Vérifiez qu'il est commité, sinon les autres membres de l'équipe auront l'erreur d'import.

## 5. Vérification

```powershell
dir src\__init__.py
python -c "import src; print('paquet reconnu')"
```

Si la seconde commande échoue, vous n'êtes pas dans le bon dossier (celui qui contient `src/`).