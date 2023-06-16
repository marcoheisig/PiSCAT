from __future__ import annotations

import abc
import pathlib
from typing import Any, Literal

import numpy as np


class FileIO(abc.ABC):
    _path: pathlib.Path
    _shape: tuple[int, ...]
    _dtype: np.dtype

    _bits: np.ndarray[Any, np.dtype[np.bool_]]  # Track who has been referenced.
    _bitcount: int = 0  # Track how many have been referenced.
    _state: Literal["uninitialized", "initialized", "finalized"] = "uninitialized"

    def __init__(self, path: pathlib.Path, shape: tuple[int, ...], dtype: np.dtype):
        self._path = path
        self._shape = shape
        self._dtype = dtype
        self._bits = np.full(shape[0], False, dtype)

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def __len__(self) -> int:
        return self._shape[0]

    def _maybe_initialize(self) -> None:
        if self._state == "uninitialized":
            self._initialize()

    def _maybe_finalize(self) -> None:
        if self._bitcount == len(self) and self._state == "initialized":
            self._finalize()

    def _mark(self, start: int, stop: int) -> None:
        for index in range(start, stop):
            if self._bits[index]:
                raise RuntimeError(f"Multiple references to the index {index}.")
            self._bits[index] = True
            self._bitcount += 1

    def _initialize(self) -> None:
        """
        This method is called once, right before performing the first IO operation.
        """
        assert self._state == "uninitialized"
        self._state = "initialized"

    def _finalize(self) -> None:
        """
        This method is called once, right after performing all IO operations.
        """
        assert self._state == "initialized"
        self._state = "finalized"


class FileReader(FileIO):
    """
    The abstract base class for reading slices of data from a file.

    :param path: The absolute path to the file that is being read from.
    :param shape: The shape of items in the file.
    :param dtype: The type of each element in the file.
    """

    def read_chunk(self, array: np.ndarray, start: int, stop: int) -> None:
        """
        Read the data from the designated interval

        :param array: Where the data is written to.
        :param start: The index of the first part of the data to be read.
        :param stop: The index right after the last part of the data to be read.
        :raise RuntimeError: Raise an error if an index is read from multiple times.
        """
        assert len(array) == stop - start
        self._maybe_initialize()
        self._mark(start, stop)
        self._read_chunk(array, start, stop)
        self._maybe_finalize()

    @abc.abstractmethod
    def _read_chunk(self, target: np.ndarray, start: int, stop: int) -> None:
        ...


class FileWriter(FileIO):
    """
    The abstract base class for storing slices of data in a file.

    :param path: The absolute path to the file that is being stored to.
    :param shape: The shape of the data being written.
    :param dtype: The dtype of the data being written.
    """

    def write_chunk(self, array: np.ndarray, position: int) -> None:
        """
        Write the supplied array to the designated interval.

        :param array: The data to be written.
        :param position: The position where array[0] is written to.
        :raise RuntimeError: Raise an error if an index is written to multiple times.
        """
        self._maybe_initialize()
        self._mark(position, position + len(array))
        self._write_chunk(array, position)
        self._maybe_finalize()

    @abc.abstractmethod
    def _write_chunk(self, array: np.ndarray, position: int) -> None:
        ...
