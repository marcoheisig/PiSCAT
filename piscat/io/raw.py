from __future__ import annotations

import mmap
import os
import pathlib

import numpy as np

from piscat.io.fileio import FileReader, FileWriter


class RawReader(FileReader):
    _fd: int
    _mmap: mmap.mmap | None = None

    def __init__(self, path: pathlib.Path, shape: tuple[int, ...], dtype: np.dtype):
        if not path.exists or not path.is_file():
            raise ValueError(f"No raw file at {path}.")
        super().__init__(path, shape, dtype)

    def _initialize(self) -> None:
        super()._initialize()
        (f, h, w) = self.shape
        dtype = self.dtype
        nbytes = f * h * w * dtype.itemsize
        self._fd = os.open(self.path, flags=os.O_RDONLY)
        self._mmap = mmap.mmap(self._fd, nbytes, access=mmap.ACCESS_READ)

    def _finalize(self) -> None:
        assert self._mmap is not None
        self._mmap.close()
        os.close(self._fd)
        super()._finalize()

    def _read_chunk(self, array: np.ndarray, start: int, stop: int) -> None:
        (_, h, w) = self.shape
        dtype = self.dtype
        count = stop - start
        size = h * w * dtype.itemsize
        assert self._mmap is not None
        data = np.frombuffer(self._mmap[start * size : stop * size], dtype=dtype)
        array[0:count] = data.reshape(count, h, w)


class RawWriter(FileWriter):
    _fd: int
    _mmap: mmap.mmap | None = None

    def __init__(self, path: pathlib.Path, shape: tuple[int, ...], dtype: np.dtype):
        super().__init__(path, shape, dtype)

    def _initialize(self) -> None:
        super()._initialize()
        (f, h, w) = self.shape
        dtype = self.dtype
        nbytes = f * h * w * dtype.itemsize
        self._fd = os.open(self.path, flags=os.O_RDWR | os.O_CREAT)
        os.write(self._fd, nbytes * b"\0")
        os.lseek(self._fd, 0, 0)
        self._mmap = mmap.mmap(self._fd, nbytes, access=mmap.ACCESS_WRITE)

    def _finalize(self) -> None:
        assert self._mmap is not None
        self._mmap.close()
        os.close(self._fd)
        super()._finalize()

    def _write_chunk(self, array: np.ndarray, position: int) -> None:
        (_, h, w) = self.shape
        dtype = self.dtype
        bytes_per_frame = h * w * dtype.itemsize
        assert self._mmap is not None
        start_byte = position * bytes_per_frame
        stop_byte = (position + len(array)) * bytes_per_frame
        self._mmap[start_byte:stop_byte] = array.tobytes()
