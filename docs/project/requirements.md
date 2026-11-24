# `requirements.txt`

**Location:** `DM_Project/requirements.txt` (project root)
**Roadmap step:** File 02 (Stage 0, setup)

## Purpose

Lists the exact Python packages and versions the project needs, so the environment can be rebuilt identically on any machine with one command. Exact versions matter here because saved models record the library versions they were trained with, and `model_store.py` refuses to load a model trained with incompatible versions.

## Contents

| Package | Version | Role |
|---|---|---|
| `numpy` | 2.3.5 | Arrays; saved matrices |
| `pandas` | 2.3.3 | Loading and cleaning data (2.3.x chosen on purpose; pandas 3.x changes defaults) |
| `pyarrow` | 22.0.0 | Parquet read/write |
| `scipy` | 1.16.3 | Dependency of scikit-learn |
| `scikit-learn` | 1.8.0 | Preprocessing, metrics, tuning, 10 of the 11 models |
| `xgboost` | 3.1.3 | The 11th model, GPU-capable |
| `joblib` | 1.5.3 | Saving and loading models |
| `matplotlib` | 3.10.8 | Figures |
| `seaborn` | 0.13.2 | Heatmaps and distribution plots |
| `streamlit` | 1.52.2 | The interface |
| `tqdm` | 4.67.3 | Progress bars |
| `pytest` | 9.0.2 | Tests |

Full reasoning for each version, and what was and was not verified, is in Deliverable 7.

## Inputs and outputs

- **Input:** none (a plain text file read by `pip`).
- **Output:** none by itself. `pip install -r requirements.txt` installs the packages into the active environment.

## Dependencies

None. It is installed into the `.venv` created in this step, and `pyproject.toml` (File 03) assumes it is installed.

## How to set up and verify

From the project root in PowerShell:

```powershell
python --version                       # note the version
python -m venv .venv
.\.venv\Scripts\Activate.ps1           # the prompt gains a (.venv) prefix
python --version                       # should still print Python 3.13.x
python -m pip install --upgrade pip
pip install -r requirements.txt
pip check
python -c "import numpy, pandas, sklearn, xgboost, streamlit, pyarrow, matplotlib, seaborn, joblib, tqdm, pytest; print('ok', sklearn.__version__, xgboost.__version__)"
```

**Expected result:**
- The install ends with a line starting `Successfully installed ...`.
- `pip check` prints `No broken requirements found.`
- The last command prints `ok 1.8.0 3.1.3`.

In VS Code or Antigravity, select the interpreter: Command Palette → "Python: Select Interpreter" → `.venv\Scripts\python.exe`.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `Activate.ps1 cannot be loaded because running scripts is disabled` | PowerShell execution policy | Run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again |
| `python` is not recognised | Python is not on the PATH in this terminal | Try `py -3.13 -m venv .venv`, or open the "Anaconda Prompt" |
| The venv reports a different Python than 3.13 | `python` resolved to another installation | Delete `.venv`, then create it with `py -3.13 -m venv .venv` (or the full path of the 3.13 `python.exe`) |
| `ERROR: Could not find a version that satisfies the requirement ...` | Python older than 3.11, or no internet access | Check `python --version` and your connection |
| `No matching distribution found for xgboost==3.1.3` | Same causes (old Python, offline) | Same fix |
| A long install or timeout | Large packages (xgboost about 72 MB, scipy, pyarrow) on a slow connection | Re-run the same command; `pip` resumes from its cache |
| `pip` installs into the wrong environment | The venv was not activated | Check that the prompt starts with `(.venv)` |

## Decisions and limitations

- **Exact pins (`==`)** instead of ranges, for reproducibility and compatible saved models.
- **Developer and runtime dependencies in one file** (`pytest` included) to keep setup to one command.
- **Tested resolution.** The set was installed in a clean Python 3.13 environment and a smoke test fitted all 11 model types. It was not run on Windows, so this step is the first real Windows test.
- **Not included on purpose:** `imbalanced-learn`, `mlxtend`, `graphviz`, `jupyter` (reasons in Deliverable 7).