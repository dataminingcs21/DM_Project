# `src/nids/utils/logger.py`

**Location:** `DM_Project/src/nids/utils/logger.py`
**Roadmap step:** File 05 (Stage 0, foundations)

## Purpose

Gives every module one consistent way to log. The project uses `logging` instead of scattered `print` calls, so each run leaves a readable record in `logs/` and the console shows the same progress messages.

## Public contents

| Name | What it does |
|---|---|
| `get_logger(name)` | Returns a logger for `name` (normally `__name__`). Configures logging on first use. A name that does not start with `nids` is prefixed, so `get_logger("scripts.train")` gives `nids.scripts.train` |
| `configure_logging(level=None, log_to_file=True, force=False)` | Sets up the `nids` logger once. `level` defaults to `settings.LOG_LEVEL`. `force=True` rebuilds the handlers (used by tests, or to change the level at run time) |
| `ROOT_LOGGER_NAME`, `LOG_FORMAT`, `DATE_FORMAT`, `MAX_LOG_BYTES`, `BACKUP_COUNT` | Constants: the namespace `nids`, the line format, 5 MB per log file, 3 old files kept |

**Line format:** `2026-11-14 20:08:34 | INFO     | nids.data.loader | message`

**Handlers**
- **Console:** stdout, so progress is visible. It never crashes on characters the console cannot encode; they are shown as `?` (some CIC-IDS2017 labels contain odd characters).
- **File:** `logs/nids.log`, rotating at 5 MB with 3 backups. If the file cannot be opened, the console handler keeps working and one warning is logged.

## Inputs and outputs

- **Input:** the logger name, plus `LOG_LEVEL`, `LOGS_DIR` and `LOG_FILE_NAME` from `config/settings.py`.
- **Output:** log lines on the console and in `logs/nids.log`.

## Dependencies

`config/settings.py` (File 04). It uses only the Python standard library otherwise.

## How to use it in other files

```python
from nids.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Loaded %d files", n_files)       # use %s-style arguments, not f-strings
logger.warning("Class %s has only %d rows", name, count)
```

## How to test

From `DM_Project/` with the `.venv` active:

```powershell
python -c "from nids.utils.logger import get_logger; log = get_logger('check'); log.info('hello'); log.warning('a warning'); log.debug('hidden at INFO level')"
Get-Content logs\nids.log -Tail 5
```

**Expected output:** two lines on the console (INFO and WARNING, with the debug line hidden), for example:

```
2026-11-14 20:08:34 | INFO     | nids.check | hello
2026-11-14 20:08:34 | WARNING  | nids.check | a warning
```

The second command shows the same two lines read back from `logs\nids.log` (the file and the `logs/` folder are created automatically). Running the first command again appends two more lines, each printed once, never twice.

## Common errors

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'nids'` | `.venv` not active, or File 03's `pip install -e .` not run | Activate `.venv`; run `pip install -e .` |
| Every line appears twice | Another library called `logging.basicConfig()` and the project's logger also propagated to it | Not possible here: the `nids` logger has `propagate = False` and we tested it. If you create your own `logging.getLogger(...)` outside `nids`, use `get_logger` instead |
| `Could not open the log file in ... logging to the console only` | The `logs` path exists as a file, is read-only, or is locked | Fix the folder (it must be a writable directory) |
| `PermissionError` when the log file rotates on Windows | Another process (for example a second terminal running a script) holds `nids.log` open | Close the other process; rotation only happens at 5 MB |
| `DEBUG` messages not shown | The level is `INFO` | `configure_logging(level="DEBUG", force=True)`, or set `LOG_LEVEL = "DEBUG"` in settings |
| Nothing is logged from inside joblib or multiprocessing workers | Worker processes start with no handlers | Log in the main process (our parallel code returns results to it) |

## Decisions and limitations

- **stdout, not stderr, for the console.** Some terminals show stderr in red as if it were an error, which would be misleading for ordinary progress messages.
- **One shared `nids` namespace with `propagate = False`,** so project messages are never duplicated by other logging setups and third-party libraries stay quiet by default.
- **Configured once and lazily,** on the first `get_logger` call, so importing a module has no side effect until a logger is requested.
- **Log folder read at configuration time.** The logger reads `settings.LOGS_DIR` when it is first configured. Tests that change `NIDS_PROJECT_ROOT` must do so before the first import of `config.settings`.
- **File rotation is size-based,** not time-based. Long jobs such as grid search can log many lines; the 5 MB × 3 limit prevents unbounded growth.