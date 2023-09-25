from __future__ import annotations

import weakref
from abc import ABC, ABCMeta, abstractmethod
from typing import Iterable, Iterator, NamedTuple, Sequence

import numpy as np
from tqdm import tqdm

Dtype = np.dtype

Array = np.ndarray

DEBUG = True


class Chunk:
    """
    A lazily evaluated memory region.

    There are two internal representations of a chunk - either its data is
    stored directly as a NumPy array, or its data is expressed as the output of
    a particular operation.

    :param shape: A non-empty tuple describing the shape of the chunk.
    :param dtype: The dtype of each element of the chunk.
    :param data: A lazily computed ndarray containing the contents of the chunk.
    :param users: A weak set of all entities referencing the chunk.  When
        evaluating a graph of chunks, the memory of a chunk with zero users can
        be reclaimed after its last successor has been processed.
    """

    _shape: tuple[int, ...]
    _dtype: Dtype
    _data: Array | None
    _actions: list[Action]
    _users: weakref.WeakSet

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: Dtype,
        data: Array | None = None,
    ):
        # There is no need for lazy evaluation if the array has zero elements.
        if 0 in shape:
            data = Array(shape=shape, dtype=dtype)
        self._shape = shape
        self._dtype = dtype
        self._data = data
        self._actions = []
        self._users = weakref.WeakSet()

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Return a tuple that is the shape of the chunk.
        """
        return self._shape

    @property
    def dtype(self) -> Dtype:
        """
        Return the type of each element of the chunk.
        """
        return self._dtype

    @property
    def data(self) -> Array:
        """
        Return the NumPy array underlying the chunk.

        Depending on the internal representation of the chunk, this operation
        can be a simple reference to the existing array, or involve the
        evaluation of the entire compute graph whose root node is this chunk.
        In the latter case the evaluation is cached so that all future
        references to the chunk's data can directly return an existing array.
        """
        if self._data is None:
            compute_chunks([self])
            assert isinstance(self._data, Array)
            assert self._data.shape == self._shape
            assert self._data.dtype == self._dtype
        return self._data

    @property
    def actions(self) -> list[Action]:
        """
        Return the actions that need to be run to initialize the chunk.
        """
        return self._actions

    def __repr__(self) -> str:
        return f"Chunk({self.shape!s}, {self.dtype!s}, id={id(self):#x})"

    def __len__(self) -> int:
        """
        Return the dimension of the chunk's first axis.
        """
        return self._shape[0]

    def __getitem__(self, index):
        """
        Return a selection of the chunk's data.
        """
        return self.data[index]

    def __setitem__(self, index, value):
        """
        Fill a selection of the chunk's data with the supplied value.
        """
        self.data[index] = value

    def __array__(self):
        return self.data


class Batch(NamedTuple):
    """
    A batch is a selection of some part of the first axis of a chunk.
    """

    chunk: Chunk
    start: int
    stop: int


class ActionClass(ABCMeta):
    @abstractmethod
    def target_shapes(self, *source_shapes) -> tuple:
        ...

    @abstractmethod
    def target_dtypes(self, *source_dtypes) -> tuple:
        ...


class Action(ABC):
    _targets: list[Batch]
    _sources: list[Batch]

    def __init__(self, targets: list[Batch], sources: list[Batch]):
        if DEBUG:
            for schunk, sstart, sstop in sources:
                if schunk._data is not None:
                    continue
                count = sstop - sstart
                mask = np.zeros(count, np.uint8)
                for action in schunk.actions:
                    for tchunk, tstart, tstop in action.targets:
                        if tchunk == schunk:
                            istart = max(sstart, tstart)
                            istop = min(sstop, tstop)
                            for index in range(istart, istop):
                                mask[index - sstart] = 1
                for index in range(sstart, sstop):
                    if mask[index - sstart] == 0:
                        raise RuntimeError(f"Reference to undefined element {index}.")
        # Connect this action to its targets.
        for tchunk, _, _ in targets:
            tchunk._actions.append(self)
        self._targets = targets
        self._sources = sources

    @property
    def targets(self):
        return self._targets

    @property
    def sources(self):
        return self._sources

    @abstractmethod
    def run(self):
        ...


def compute_chunks(chunks: Iterable[Chunk]) -> None:
    """
    Ensure that all the supplied video chunks and their dependencies have their
    data and metadata computed.
    """
    # Build a schedule.
    schedule: list[Action] = []
    visited: set[Action] = set()

    def process(chunk: Chunk) -> None:
        if chunk._data is not None:
            return
        actions = chunk._actions
        chunk._actions = []
        for action in actions:
            if action not in visited:
                visited.add(action)
                for source, _, _ in action.sources:
                    process(source)
                schedule.append(action)

    for chunk in chunks:
        process(chunk)
    # Execute the schedule.
    for action in tqdm(schedule, delay=0.5):
        for chunk, _, _ in action.targets:
            allocate_chunk(chunk)
        action.run()


def allocate_chunk(chunk: Chunk):
    if isinstance(chunk._data, Array):
        return
    else:
        chunk._data = Array(shape=chunk.shape, dtype=chunk.dtype)


def batches_from_chunks(
    chunks: Sequence[Chunk], start: int = 0, stop: int | None = None
) -> Iterator[Batch]:
    stop = sum(chunk.shape[0] for chunk in chunks) if stop is None else stop
    length = stop - start
    if length == 0:
        return
    position = 0
    for chunk in chunks:
        size = chunk.shape[0]
        batch_start = 0 if position > start else start - position
        batch_stop = size if position + size < stop else stop - position
        if batch_start < batch_stop:
            yield Batch(chunk, batch_start, batch_stop)
        position += size
    assert position >= length
