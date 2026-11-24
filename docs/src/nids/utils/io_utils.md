# `src/nids/utils/io_utils.py`

**Location:** `DM_Project/src/nids/utils/io_utils.py`
**Roadmap step:** File 06 (Stage 0, foundations)

## Purpose

One place for reading and writing files safely. Every later stage (cleaning, preprocessing, training, evaluation, the Streamlit app) saves and loads Parquet files, JSON reports, NumPy arrays and joblib models through these functions, so they all behave the same way and fail with clear messages.

## Public contents

| Function | What it does |
|---|---|
| `ensure_dir(path)` | Creates the folder (and parents) if needed; returns a `Path` |
| `require_file(path, hint=None)` | Returns the path if the file exists, else raises `FileNotFoundError` with the file name and the optional `hint` (for example "Run scripts/04_preprocess.py first") |
| `write_parquet(df, path)` / `read_parquet(path, columns=None, hint=None)` | DataFrame <-> Parquet (pyarrow, snappy, no index). `columns` loads only some columns to save RAM |
| `write_json(data, path, indent=2)` / `read_json(path, hint=None)` | Standard UTF-8 JSON. NumPy numbers/arrays are converted, NaN and Infinity become `null`, sets become sorted lists, `Path` becomes text, non-string keys become strings. Unsupported objects raise `TypeError`; corrupt files raise `ValueError` naming the file |
| `save_array(array, path)` / `load_array(path, mmap_mode=None, hint=None)` | `.npy` arrays without pickling (object arrays are refused). `mmap_mode="r"` maps the file instead of loading it into RAM |
| `dump_joblib(obj, path, compress=0)` / `load_joblib(path, hint=None)` | Fitted transformers and models |
| `config_hash(config, length=16)` | Short SHA-256 fingerprint of a settings dictionary, independent of key order |

**Atomic writes.** Every `write_*`, `save_array` and `dump_joblib` first writes `<name>.tmp` and then replaces the final file in one step. If the write fails (disk full, bad data, crash), the previous file is untouched and no `.tmp` is left behind. A half-written metadata file or model can therefore never be loaded by mistake.

## Inputs and outputs

- **Inputs:** objects to save and paths (`str` or `Path`).
- **Outputs:** the saved file (the write functions return its `Path`) or the loaded object. Debug-level log lines record sizes.

## Dependencies

- `nids.utils.logger` (File 05), which itself uses `config/settings.py` (File 04).
- Third-party: `numpy`, `pandas`, `pyarrow`, `joblib` (all pinned in `requirements.txt`).

`config_hash` is used from File 25 on: `preprocessing_config()` from settings is hashed into `preprocessing_metadata.json`, so changing, for example, `SCALER_TYPE` marks old preprocessed data as stale.

## How to test

From `DM_Project/` with the `.venv` active (PowerShell):

```powershell
python -c "from nids.utils.io_utils import write_json, read_json, config_hash; write_json({'a': 1, 'b': float('nan')}, 'logs/t.json'); print(read_json('logs/t.json')); print(config_hash({'x': 1, 'y': 2}) == config_hash({'y': 2, 'x': 1}))"
python -c "from nids.utils.io_utils import read_json; read_json('logs/missing.json', hint='Run scripts/04_preprocess.py first')"
Remove-Item logs\t.json
```

**Expected output of the first command:**

```
{'a': 1, 'b': None}
True
```

**Expected output of the second command:** a traceback ending with

```
FileNotFoundError: File not found: logs\missing.json. Run scripts/04_preprocess.py first
```

(this error is the intended result). The full test suite for this file comes with the later test files; in the sandbox all round trips (Parquet keeping `float32`, JSON, `.npy`, joblib), the hash, the error messages and the failed-write behaviour passed.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'nids'` | `.venv` not active or `pip install -e .` not run | Activate `.venv`; `pip install -e .` |
| `ModuleNotFoundError: No module named 'pyarrow'` | Requirements not installed | `pip install -r requirements.txt` |
| `FileNotFoundError: File not found: ...` | The previous stage has not been run | Run the script named in the hint |
| `TypeError: Object of type X is not JSON serializable` | Data contains an unsupported object (for example a model) | Convert it to numbers/strings first |
| `ValueError: Invalid JSON in ...` | The JSON file was edited by hand or damaged | Delete it and regenerate it with its script |
| `ValueError: Object arrays cannot be saved when allow_pickle=False` | An array of Python objects was passed to `save_array` | Convert to a numeric dtype first |
| `PermissionError` when overwriting an `.npy` on Windows | The file is still memory-mapped (`mmap_mode="r"`) by a running process | Delete the array variable / restart the process, then rewrite |

## Decisions and limitations

- **Atomic writes** protect against partial files, at the cost of briefly needing room for two copies of a file.
- **NaN and Infinity become `null` in JSON,** because standard JSON has no such values and other tools reject them.
- **`allow_pickle=False` for arrays:** safer, and all project arrays are numeric.
- **`load_joblib` unpickles data, which can run code.** Only load files created by this project; never load a model file received from someone else.
- **Not here:** the Parquet column-type rules (File 07, `schema.py`) and the model metadata sidecar format (File 43, `model_store.py`).