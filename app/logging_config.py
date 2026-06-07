"""Central logging configuration.

Logs are emitted to both the terminal (stderr) and a rotating file under
``logs/service.log``. Every module obtains its logger through ``get_logger`` so
that handlers are configured exactly once, regardless of import order or
uvicorn reloads.
"""
import logging
from logging.handlers import RotatingFileHandler

from app.config import LOG_FILE_PATH, LOG_LEVEL, LOGS_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level: "str | int | None" = None) -> None:
    """Configure root handlers once: a console stream and a rotating file."""
    global _configured
    if _configured:
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Rotate at ~5 MB, keeping the last 5 files so logs never grow unbounded.
    file_handler = RotatingFileHandler(
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
