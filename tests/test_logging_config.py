import logging
import os

from app.logging_config import MultiprocessSafeRotatingFileHandler


def test_rollover_lock_falls_back_to_process_log(tmp_path, monkeypatch):
    log_path = tmp_path / "service.log"
    log_path.write_text("already full", encoding="utf-8")
    handler = MultiprocessSafeRotatingFileHandler(
        log_path,
        maxBytes=1,
        backupCount=1,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    def locked_rotation(source, destination):
        raise PermissionError("simulated Windows file lock")

    monkeypatch.setattr(handler, "rotate", locked_rotation)

    try:
        handler.emit(
            logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname=__file__,
                lineno=1,
                msg="server started",
                args=(),
                exc_info=None,
            )
        )
    finally:
        handler.close()

    fallback = tmp_path / f"service.{os.getpid()}.log"
    assert fallback.read_text(encoding="utf-8").strip() == "server started"
