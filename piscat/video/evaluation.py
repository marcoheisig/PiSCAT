from __future__ import annotations

import weakref
from typing import Callable, Iterable, NamedTuple, Sequence

import numpy as np
from tqdm import tqdm

Array = np.ndarray

BYTES_PER_CHUNK = 2**21


class VideoChunk:
    """
    The in-memory representation of several consecutive video frames.

    A video chunk can have two internal representations - either its data is
    stored directly as a NumPy array, or its data is expressed as the output of
    a particular video op.

    :param shape: A (frames, height, width) tuple describing the shape of the
        chunk.
    :param dtype: The NumPy dtype describing the elements of the chunk.
    :param data: A NumPy array containing the contents of the chunk.  Computed
        lazily.
    :param users: A weak set of all entities (videos, operations, ...)
        referencing this chunk.  Whenever a graph of chunks and operations is
        evaluated, and all users of a chunk are part of that evaluation, there
        is no need to retain the chunk's memory afterwards.
    """

    _shape: tuple[int, int, int]
    _dtype: np.dtype
    _data: VideoOp | Array | None
    _users: weakref.WeakSet

    def __init__(
        self,
        shape: tuple[int, int, int],
        dtype: np.dtype,
        data: VideoOp | Array | None = None,
    ):
        if isinstance(data, Array):
            assert data.shape == shape
            assert data.dtype == dtype
        # There is no need for lazy evaluation if the array has zero elements.
        if 0 in shape:
            data = Array(shape=shape, dtype=dtype)
        self._shape = shape
        self._dtype = dtype
        self._data = data
        self._users = weakref.WeakSet()

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Return the shape of the video chunk as a (frames, height, width) tuple.
        """
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """
        Return the element type of each pixel of the video chunk.
        """
        return self._dtype

    @property
    def data(self) -> Array:
        """
        Return the NumPy array underlying the video chunk.

        Depending on the internal representation of the video chunk, this
        operation can be a simple reference to the existing array, or involve
        the evaluation of the entire compute graph whose root node is this video
        chunk.  In the latter case the evaluation is cached so that all future
        references to the video chunk's data can directly return an existing
        array.
        """
        if isinstance(self._data, Array):
            return self._data
        else:
            compute_chunks([self])
            assert isinstance(self._data, Array)
            assert self._data.shape == self._shape
            assert self._data.dtype == self._dtype
            return self._data

    @staticmethod
    def from_array(array: Array):
        shape = array.shape
        if len(shape) != 3:
            raise ValueError("Only rank three arrays can be converted to chunks.")
        return VideoChunk(shape=shape, dtype=array.dtype, data=array)

    def __repr__(self) -> str:
        return f"VideoChunk({self.shape!r}, {self.dtype!r}, id={id(self):#x})"

    def __len__(self) -> int:
        """
        Return the number of frames in the video chunk.
        """
        return self._shape[0]

    def __getitem__(self, index):
        """
        Return a selection of the video chunk's data.
        """
        return self.data[index]

    def __setitem__(self, index, value):
        """
        Fill a selection of the video chunk's data with the supplied value.
        """
        self.data[index] = value

    def __array__(self):
        return self.data


class Batch(NamedTuple):
    """
    A batch is a selection of some part of a video chunk.
    """

    chunk: VideoChunk
    start: int
    stop: int


Batches = Sequence[Batch]


Kernel = Callable[[Batches, Batches], None]


def copy_kernel(targets: Batches, sources: Batches) -> None:
    assert len(targets) == 1
    (target, tstart, tstop) = targets[0]
    pos = tstart
    for source, sstart, sstop in sources:
        count = sstop - sstart
        target[pos : pos + count] = source[sstart:sstop]
        pos += count
    assert pos == tstop


class VideoOp:
    """
    A video op describes how the contents of some video chunks can be computed,
    possibly depending on the contents of several other video chunks.

    :param targets: A list of batches that will be written to by this kernel's
        kernel.
    :param sources: A list of batches that will be read from by this kernel's
        kernel.
    :param kernel: The callable that will be invoked by this kernel once the
        data of one or more of its targets is accessed.
    """

    _targets: list[Batch]
    _sources: list[Batch]
    _kernel: Kernel

    def __init__(self, kernel: Kernel, targets: list[Batch], sources: list[Batch]):
        assert len(targets) > 0
        # Connect this kernel with its source and target chunks.
        self._targets = targets
        self._sources = sources
        self._kernel = kernel
        for target, _, _ in targets:
            assert not target._data
            target._data = self
        for source, _, _ in sources:
            source._users.add(self)

    @property
    def sources(self) -> list[Batch]:
        return self._sources

    @property
    def targets(self) -> list[Batch]:
        return self._targets

    @property
    def kernel(self) -> Kernel:
        return self._kernel

    def run(self) -> None:
        # Allocate all targets.
        for target, _, _ in self.targets:
            assert target._data is not None
            if isinstance(target._data, VideoOp):
                target._data = Array(shape=target.shape, dtype=target.dtype)
            assert isinstance(target._data, Array)
        for source, _, _ in self.sources:
            assert isinstance(source._data, Array)
        # Run the kernel.
        # print(f"Kernel: {self.targets=} <- {self.sources=}")
        self.kernel(self.targets, self.sources)
        # Mark each target as read-only.
        for target, _, _ in self.targets:
            target.data.setflags(write=False)
        # Sever the connection to each source.
        for source, _, _ in self.sources:
            assert isinstance(source._data, Array)
            source._users.remove(self)


def compute_chunks(chunks: Iterable[VideoChunk]) -> None:
    """
    Ensure that all the supplied video chunks and their dependencies have their
    data and metadata computed.
    """
    # Build a schedule.
    schedule: list[VideoOp] = []
    kernels: set[VideoOp] = set()

    def process(chunk: VideoChunk) -> None:
        data = chunk._data
        if data is None:
            raise RuntimeError("Encountered a chunk with undefined contents.")
        if isinstance(data, Array):
            return
        kernel = data
        if kernel not in kernels:
            kernels.add(kernel)
            for source, _, _ in kernel.sources:
                process(source)
            schedule.append(kernel)

    for video_chunk in chunks:
        process(video_chunk)
    # Execute the schedule.
    for kernel in tqdm(schedule, delay=0.5):
        kernel.run()


def ceildiv(a: int, b: int) -> int:
    return -(a // -b)
