"""Storage helpers for wavefield snapshots.

This mirrors Deepwave's snapshot storage abstraction for use in the Maxwell
propagator. Stage 1 supports snapshot storage on device/CPU/disk.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from pathlib import Path
from typing import List
from uuid import uuid4


STORAGE_DEVICE = 0
STORAGE_CPU = 1
STORAGE_DISK = 2
STORAGE_NONE = 3


def storage_mode_to_int(storage_mode_str: str) -> int:
    mode = storage_mode_str.lower()
    if mode == "device":
        return STORAGE_DEVICE
    if mode == "cpu":
        return STORAGE_CPU
    if mode == "disk":
        return STORAGE_DISK
    if mode == "none":
        return STORAGE_NONE
    raise ValueError(
        "storage_mode must be 'device', 'cpu', 'disk', 'none', or 'auto', "
        f"but got {storage_mode_str!r}"
    )


class TemporaryStorage:
    """Manages temporary files for disk storage.

    Creates a unique subdirectory for each instantiation to prevent collisions.
    """

    def __init__(self, base_path: str, num_files: int) -> None:
        self.base_dir = Path(base_path) / f"tide_tmp_{os.getpid()}_{uuid4().hex}"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.filenames: List[str] = [
            str(self.base_dir / f"shot_{i}.bin") for i in range(num_files)
        ]

    def get_filenames(self) -> List[str]:
        return self.filenames

    def close(self) -> None:
        if self.base_dir.exists():
            with contextlib.suppress(OSError):
                shutil.rmtree(self.base_dir)

    def __del__(self) -> None:
        self.close()

