from __future__ import annotations

import pathlib

import numpy as np

from piscat.io.fileio import FileReader, FileWriter


class NumpyReader(FileReader):
    _memmap: np.memmap | None = None
    _row_major: bool

    def __init__(self, path: pathlib.Path):
        if not path.exists or not path.is_file():
            raise ValueError(f"No Numpy file at {path}.")
        with open(path, "rb") as fp:
            (major, _) = np.lib.format.read_magic(fp)
            if major == 1:
                shape, forder, dtype = np.lib.format.read_array_header_1_0(fp)
            else:
                shape, forder, dtype = np.lib.format.read_array_header_2_0(fp)
            assert isinstance(forder, bool)
            self._row_major = not forder
        super().__init__(path, shape, dtype)

    def _initialize(self) -> None:
        assert self._memmap is None
        super()._initialize()
        self._memmap = np.lib.format.open_memmap(self.path, mode="r")
        assert self._memmap.dtype == self.dtype
        assert self._memmap.shape == self.shape

    def _finalize(self) -> None:
        assert self._memmap is not None
        self._memmap.flush()
        self._memmap = None
        super()._finalize()

    def _read_chunk(self, array: np.ndarray, start: int, stop: int) -> None:
        assert self._memmap is not None
        count = stop - start
        data = self._memmap[start:stop]
        if not self._row_major:
            np.transpose(data, (2, 1, 0))
        array[0:count] = data


class NumpyWriter(FileWriter):
    _memmap: np.memmap | None = None

    def __init__(self, path: pathlib.Path, shape: tuple[int, ...], dtype: np.dtype):
        super().__init__(path, shape, dtype)

    def _initialize(self) -> None:
        assert not self._memmap
        super()._initialize()
        self._memmap = np.lib.format.open_memmap(
            self.path, mode="w+", dtype=self.dtype, shape=self.shape
        )

    def _finalize(self) -> None:
        assert self._memmap is not None
        self._memmap.flush()
        self._memmap = None
        super()._finalize()

    def _write_chunk(self, array: np.ndarray, position: int) -> None:
        assert self._memmap is not None
        self._memmap[position : position + len(array)] = array
