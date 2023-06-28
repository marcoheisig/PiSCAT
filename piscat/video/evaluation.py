from __future__ import annotations

import weakref
from typing import Callable, Iterable, Iterator, NamedTuple, Sequence

import numpy as np
from tqdm import tqdm

from piscat.io import FileReader

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
    step: int = 1


Batches = Sequence[Batch]


Kernel = Callable[[Batches, Batches], None]


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
        for target, _, _, _ in targets:
            assert not target._data
            target._data = self
        for source, _, _, _ in sources:
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
        for target, _, _, _ in self.targets:
            assert target._data is not None
            if isinstance(target._data, VideoOp):
                target._data = Array(shape=target.shape, dtype=target.dtype)
            assert isinstance(target._data, Array)
        for source, _, _, _ in self.sources:
            assert isinstance(source._data, Array)
        # Run the kernel.
        self.kernel(self.targets, self.sources)
        # Mark each target as read-only.
        for target, _, _, _ in self.targets:
            target.data.setflags(write=False)
        # Sever the connection to each source.
        for source, _, _, _ in self.sources:
            assert isinstance(source._data, Array)
            source._users.remove(self)


def copy_kernel(targets: Batches, sources: Batches) -> None:
    assert len(targets) == 1
    (target, tstart, tstop, _) = targets[0]
    pos = tstart
    for source, sstart, sstop, _ in sources:
        count = sstop - sstart
        target[pos : pos + count] = source[sstart:sstop]
        pos += count
    assert pos == tstop


def get_fill_kernel(value) -> Kernel:
    def kernel(targets: Batches, sources: Batches):
        assert len(sources) == 0
        (target, tstart, tstop, tstep) = targets[0]
        target[tstart:tstop:tstep] = value

    return kernel


def get_reader_kernel(reader: FileReader, start: int, stop: int) -> Kernel:
    def kernel(targets: Batches, sources: Batches) -> None:
        assert len(targets) == 1
        assert len(sources) == 0
        (chunk, cstart, cstop, _) = targets[0]
        assert (stop - start) == (cstop - cstart)
        reader.read_chunk(chunk.data[cstart:cstop], start, stop)

    return kernel


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
            for source, _, _, _ in kernel.sources:
                process(source)
            schedule.append(kernel)

    for video_chunk in chunks:
        process(video_chunk)
    # Execute the schedule.
    for kernel in tqdm(schedule, delay=0.5):
        kernel.run()


def chunks_from_batches(
    batches: Iterable[Batch],
    /,
    shape: tuple[int, int, int],
    dtype: np.dtype,
    kernel: Kernel = copy_kernel,
    offset: int = 0,
    count: int | None = None,
) -> Iterator[VideoChunk]:
    """
    Process the data in all supplied batches and return it as an iterator over
    same-sized chunks.

    :param batches: An iterable over batches that supply all the data.
    :param shape: The shape of each resulting chunk.
    :param dtype: The type of each element of a resulting chunk.
    :param kernel: The function that reads data from one or more batches, and
        that writes its results to a single target batch that initializes one
        resulting chunk.
    :param offset: The number of frames of the first resulting chunk that are
        left uninitialized.
    :param count: The total number of target frames that will be defined, or
        None if this function should create chunks until the supplied batches
        are exhausted.

    :returns: An iterator over chunks of the supplied shape and dtype.
    """
    batches = iter(batches)
    if count == 0:
        raise StopIteration
    (f, h, w) = shape
    group: list[Batch] = []  # The batches that constitute the next chunk.
    gstart, gstop = 0, 0  # The interval of frames in the current group.
    ccount = 0  # The amount of chunks that have already been created.

    def merge_batches(batches: list[Batch], start: int, stop: int):
        chunk = VideoChunk(shape=shape, dtype=dtype)
        assert 0 <= start <= stop <= shape[0]
        VideoOp(kernel=kernel, targets=[Batch(chunk, start, stop)], sources=batches)
        return chunk

    while count is None or gstart < count:
        cstart = max(ccount * f, offset)
        cend = (ccount + 1) * f if not count else min((ccount + 1) * f, offset + count)
        clen = cend - cstart
        # Gather enough source chunks to fill one target chunk.
        while gstop - gstart < clen:
            try:
                batch = next(batches)
                assert batch.stop <= batch.chunk.shape[0]
            except StopIteration as e:
                if count:
                    raise ValueError("Not enough input chunks.") from e
                else:
                    if gstop - gstart > 0:
                        yield merge_batches(group, start=0, stop=gstop - gstart)
                    return
            (_, ch, cw) = batch.chunk.shape
            if ch != h or cw != w:
                raise ValueError("Cannot combine chunks with different frame sizes.")
            group.append(batch)
            gstop += batch.stop - batch.start
        rest = (gstop - gstart) - clen
        (last, lstart, lstop, lstep) = group[-1]
        group[-1] = Batch(last, lstart, lstop - rest, lstep)
        yield merge_batches(group, start=cstart - ccount * f, stop=cend - ccount * f)
        # Prepare for the next iteration.
        if rest == 0:
            group = []
        else:
            group = [Batch(last, lstop - rest, lstop, lstep)]
        gstart += clen
        ccount += 1
