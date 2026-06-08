"""Central logging configuration.

Logs are emitted to both the terminal (stderr) and a rotating file under
``logs/service.log``. Every module obtains its logger through ``get_logger`` so
that handlers are configured exactly once, regardless of import order or
uvicorn reloads.
"""
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import LOG_FILE_PATH, LOG_LEVEL, LOGS_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
# ISO-8601 with a trailing Z; timestamps are UTC (see formatter.converter below)
# so logs are comparable regardless of the container's host timezone.
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

_configured = False


class MultiprocessSafeRotatingFileHandler(RotatingFileHandler):
    """Keep logging when Windows blocks rotation of a shared log file.

    Uvicorn reload and test subprocesses can hold the same file open. Windows
    then refuses the rename used by RotatingFileHandler. Move only the process
    that lost the rollover race to its own rotating file instead of emitting a
    traceback for every subsequent log record.
    """

    def doRollover(self) -> None:
        try:
            super().doRollover()
        except PermissionError:
            fallback = self._process_fallback_path()
            self.baseFilename = str(fallback.resolve())
            self.stream = None if self.delay else self._open()

    def _process_fallback_path(self) -> Path:
        path = Path(self.baseFilename)
        return path.with_name(f"{path.stem}.{os.getpid()}{path.suffix}")


def setup_logging(level: "str | int | None" = None) -> None:
    """Configure root handlers once: a console stream and a rotating file."""
    global _configured
    if _configured:
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    # Emit all timestamps in UTC rather than the host's local time.
    formatter.converter = time.gmtime

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Rotate at ~5 MB, keeping the last 5 files so logs never grow unbounded.
    file_handler = MultiprocessSafeRotatingFileHandler(
        LOG_FILE_PATH, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level or LOG_LEVEL)
    root.handlers = [console_handler, file_handler]

    # Dampen chatty third-party loggers so our own logs stay readable.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring logging is configured first."""
    setup_logging()
    return logging.getLogger(name)
