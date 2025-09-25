import atexit
import sys
from io import TextIOBase
from pathlib import Path
from typing import IO


class _TeeStream(TextIOBase):
    def __init__(self, original: IO[str], file_handle: IO[str]):
        self._original = original
        self._file_handle = file_handle

    def write(self, data: str) -> int:
        written = self._original.write(data)
        self._original.flush()
        self._file_handle.write(data)
        self._file_handle.flush()
        return written

    def flush(self) -> None:
        self._original.flush()
        self._file_handle.flush()

    def isatty(self) -> bool:
        return self._original.isatty()


_original_streams = {}
_handles: list[IO[str]] = []
_registered = False


def _restore_streams() -> None:
    global _original_streams
    for name, stream in _original_streams.items():
        setattr(sys, name, stream)
    _original_streams = {}

    for handle in _handles:
        handle.close()
    _handles.clear()


def setup_console_logging(log_dir: str | Path, filename: str = "console.log") -> Path:
    global _registered
    target_dir = Path(log_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path = target_dir / filename
    log_handle = log_path.open("a", buffering=1)
    _handles.append(log_handle)

    for attr in ("stdout", "stderr"):
        original = getattr(sys, attr)
        if attr not in _original_streams:
            _original_streams[attr] = original
        tee_stream = _TeeStream(original, log_handle)
        setattr(sys, attr, tee_stream)

    if not _registered:
        atexit.register(_restore_streams)
        _registered = True

    return log_path
