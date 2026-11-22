"""Logging setup for the whole project.

Every module gets its logger with ``get_logger(__name__)``.  All loggers live
under the ``nids`` namespace and share two handlers:

* the console (stdout), so progress is visible while a script runs;
* a rotating file in ``logs/`` (5 MB per file, 3 old files kept), so runs can
  be reviewed later.

The configuration is applied once, so calling ``get_logger`` many times never
produces duplicated lines.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from config import settings

ROOT_LOGGER_NAME = "nids"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3

_configured = False


def _build_console_handler() -> logging.Handler:
    """Console handler on stdout that never crashes on unencodable characters.

    Some CIC-IDS2017 labels contain odd characters; on a console with a legacy
    encoding they could otherwise raise an encoding error inside the logging call.
    """
    stream = sys.stdout
    try:
        stream.reconfigure(errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass  # stream does not support reconfigure (for example a custom object)
    return logging.StreamHandler(stream)


def configure_logging(
    level: str | int | None = None,
    log_to_file: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Configure the ``nids`` logger (idempotent).

    Parameters
    ----------
    level:
        Log level name or number.  Defaults to ``settings.LOG_LEVEL``.
    log_to_file:
        Also write to ``logs/<LOG_FILE_NAME>``.  If the file cannot be opened
        the console handler is still used and a warning is logged.
    force:
        Remove the existing handlers and configure again (used by tests, or
        to change the level at run time).
    """
    global _configured
    root = logging.getLogger(ROOT_LOGGER_NAME)

    if _configured and not force:
        return root

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(level if level is not None else settings.LOG_LEVEL)
    root.propagate = False  # avoid duplicate lines through the Python root logger

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    console = _build_console_handler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_problem: str | None = None
    if log_to_file:
        try:
            settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                settings.LOGS_DIR / settings.LOG_FILE_NAME,
                maxBytes=MAX_LOG_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as error:
            file_problem = f"{settings.LOGS_DIR} ({error})"

    _configured = True
    if file_problem:
        root.warning("Could not open the log file in %s; logging to the console only.", file_problem)
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a logger for ``name`` (normally ``__name__``), configuring logging on first use."""
    configure_logging()
    if name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")