from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from piscat.video.evaluation import Action, Batch, Chunk, Dtype

FLOAT32 = np.dtype(np.float32)
FLOAT64 = np.dtype(np.float64)
INT8 = np.dtype(np.int8)
INT16 = np.dtype(np.int16)
INT32 = np.dtype(np.int32)
INT64 = np.dtype(np.int64)
UINT8 = np.dtype(np.uint8)
UINT16 = np.dtype(np.uint16)
UINT32 = np.dtype(np.uint32)
UINT64 = np.dtype(np.uint64)


### Copying Data


class Copy(Action):
    """
    An action that encapsulates the process of copying elements from a source
    batch to a target batch of the same size.

    :param targets: A list containing the target batch being copied to.
    :param sources: A list containing the source batch being copied from.
    """

    def __init__(self, target: Batch, source: Batch):
        (tchunk, tstart, tstop) = target
        (schunk, sstart, sstop) = source
        assert tstop - tstart == sstop - sstart
        assert tchunk.dtype == schunk.dtype
        super().__init__([target], [source])

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        [(schunk, sstart, sstop)] = self.sources
        tchunk[tstart:tstop] = schunk[sstart:sstop]


class Fill(Action):
    """
    An action that encapsulates the process of filling all entries of a target
    batch with the sole entry of a source batch.

    :param targets: A list containing the target batch being filled.
    :param sources: A list with a single batch that contains one entry that is
        the value being written to each entry of the target batch.
    """

    def __init__(self, target: Batch, source: Batch):
        (tchunk, tstart, tstop) = target
        (schunk, sstart, sstop) = source
        assert sstop - sstart == 1
        assert tstop - tstart >= 0
        assert tchunk.dtype == schunk.dtype
        super().__init__([target], [source])

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        [(schunk, sstart, _)] = self.sources
        tchunk[tstart:tstop] = schunk[sstart]


### Selecting Data


class Slice(Action):
    step: int
    hslice: slice
    wslice: slice
    count: int

    def __init__(self, target: Batch, source: Batch, step: int, hslice: slice, wslice: slice):
        super().__init__([target], [source])
        self.step = step
        self.hslice = hslice
        self.wslice = wslice

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        [(schunk, sstart, sstop)] = self.sources
        sstep, hslice, wslice = self.step, self.hslice, self.wslice
        tchunk[tstart:tstop, :, :] = schunk[sstart:sstop:sstep, hslice, wslice]


### Reversing Data


class Reverse(Action):
    def __init__(self, target, source):
        super().__init__([target], [source])

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        [(schunk, sstart, sstop)] = self.sources
        count = sstop - sstart
        rstart = sstop - 1
        rstop = rstart - count
        if rstop < 0:
            rstop = None
        tchunk[tstart:tstop] = schunk[rstart:rstop:-1]


### Decoding Numpy Arrays into Video Data


class DecodeArray(Action):
    def __init__(self, target: Batch, source: Batch):
        (_, tstart, tstop) = target
        (_, sstart, sstop) = source
        assert tstop - tstart == sstop - sstart
        super().__init__([target], [source])


class DecodeU64Array(DecodeArray):
    def __init__(self, target: Batch, source: Batch):
        assert target.chunk.dtype == UINT32
        assert source.chunk.dtype == UINT64
        super().__init__(target, source)

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        [(schunk, sstart, sstop)] = self.sources
        tchunk[tstart:tstop] = schunk[sstart:sstop] >> 32


class DecodeI8Array(DecodeArray):
    def __init__(self, target: Batch, source: Batch):
        assert target.chunk.dtype == UINT8
        assert source.chunk.dtype == INT8
        super().__init__(target, source)

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        [(schunk, sstart, sstop)] = self.sources
        tchunk[tstart:tstop] = schunk[sstart:sstop].view(np.uint8) + 2**7


class DecodeI16Array(DecodeArray):
    def __init__(self, target: Batch, source: Batch):
        assert target.chunk.dtype == UINT16
        assert source.chunk.dtype == INT16
        super().__init__(target, source)

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        [(schunk, sstart, sstop)] = self.sources
        tchunk[tstart:tstop] = schunk[sstart:sstop].view(np.uint16) + 2**15


class DecodeI32Array(DecodeArray):
    def __init__(self, target: Batch, source: Batch):
        assert target.chunk.dtype == UINT32
        assert source.chunk.dtype == INT32
        super().__init__(target, source)

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        [(schunk, sstart, sstop)] = self.sources
        tchunk[tstart:tstop] = schunk[sstart:sstop].view(np.uint32) + 2**31


class DecodeI64Array(DecodeArray):
    def __init__(self, target: Batch, source: Batch):
        assert target.chunk.dtype == UINT64
        assert source.chunk.dtype == INT64
        super().__init__(target, source)

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        [(schunk, sstart, sstop)] = self.sources
        tchunk[tstart:tstop] = (schunk[sstart:sstop].view(np.uint64) + 2**63) >> 32


class DecodeF32Array(DecodeArray):
    def __init__(self, target: Batch, source: Batch):
        assert target.chunk.dtype == UINT32
        assert source.chunk.dtype == FLOAT32
        super().__init__(target, source)

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        [(schunk, sstart, sstop)] = self.sources
        data = schunk[sstart:sstop] * 2**32
        tchunk[tstart:tstop] = np.clip(data, 0, 2**32 - 1).astype(np.uint32)


class DecodeF64Array(DecodeArray):
    def __init__(self, target: Batch, source: Batch):
        assert target.chunk.dtype == UINT32
        assert source.chunk.dtype == FLOAT64
        super().__init__(target, source)

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        [(schunk, sstart, sstop)] = self.sources
        data = schunk[sstart:sstop] * 2**32
        tchunk[tstart:tstop] = np.clip(data, 0, 2**32 - 1).astype(np.uint32)


Decoder = Callable[[Batch, Batch], Action]


def dtype_decoder_and_precision(dtype: Dtype) -> tuple[Decoder, int]:
    kind = dtype.kind
    bits = 8 * dtype.itemsize
    # Unsigned integers.
    if kind == "u":
        if bits == 8:
            return (Copy, 8)
        if bits == 16:
            return (Copy, 16)
        if bits == 32:
            return (Copy, 32)
        if bits == 64:
            return (DecodeU64Array, 32)
    # Signed integers.
    if kind == "i":
        if bits == 8:
            return (DecodeI8Array, 8)
        if bits == 16:
            return (DecodeI16Array, 16)
        if bits == 32:
            return (DecodeI32Array, 32)
        if bits == 64:
            return (DecodeI64Array, 32)
    # Floating-point numbers.
    if kind == "f":
        if bits == 32:
            return (DecodeF32Array, 32)
        if bits == 64:
            return (DecodeF64Array, 32)
    # Everything else.
    raise ValueError(f"Cannot convert {dtype} arrays to videos.")


### Changing Video Precision


def precision_dtype(precision: int) -> Dtype:
    if precision <= 8:
        return UINT8
    if precision <= 16:
        return UINT16
    if precision <= 32:
        return UINT32
    if precision <= 64:
        return UINT64
    raise ValueError(f"Invalid precision: {precision}")


def dtype_precision(dtype: Dtype) -> int:
    if dtype == UINT8:
        return 8
    if dtype == UINT16:
        return 16
    if dtype == UINT32:
        return 32
    if dtype == UINT64:
        return 64
    raise ValueError(f"Invalid dtype: {dtype}")


def precision_next_power_of_two(precision: int) -> int:
    if precision <= 8:
        return 8
    if precision <= 16:
        return 16
    if precision <= 32:
        return 32
    if precision <= 64:
        return 64
    raise ValueError(f"Invalid precision: {precision}")


class ChangePrecision(Action):
    shift: int

    def __init__(
        self,
        target: Batch,
        source: Batch,
        target_precision: int,
        source_precision: int,
    ):
        (tchunk, tstart, tstop) = target
        (schunk, sstart, sstop) = source
        assert tstop - tstart == sstop - sstart
        assert tchunk.dtype == precision_dtype(target_precision)
        assert schunk.dtype == precision_dtype(source_precision)
        super().__init__([target], [source])
        self.shift = target_precision - source_precision

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        [(schunk, sstart, sstop)] = self.sources
        if self.shift < 0:
            tchunk[tstart:tstop] = schunk[sstart:sstop] >> -self.shift
        else:
            tchunk[tstart:tstop] = schunk[sstart:sstop] << self.shift


### Calculating the Rolling Average


class Sum(Action):
    def __init__(self, target: Batch, sources: list[Batch]):
        (tchunk, tstart, tstop) = target
        assert tchunk.dtype == UINT64
        assert tstop - tstart == 1
        count = 0
        dtype = None
        shape = tchunk.shape[1:]
        for schunk, sstart, sstop in sources:
            assert schunk.shape[1:] == shape
            if dtype is None:
                dtype = schunk.dtype
            else:
                assert schunk.dtype == dtype
            count += sstop - sstart
        if count > 0:
            assert dtype is not None
            assert dtype.kind == "u"
            assert dtype.itemsize * 8 + np.log2(count) <= 64
        super().__init__([target], sources)

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        tchunk[tstart:tstop] = 0
        for schunk, sstart, sstop in self.sources:
            tchunk[tstart] += np.sum(schunk[sstart, sstop], axis=0, dtype=np.uint64)


class ForwardRollingAverage(Action):
    def __init__(
        self,
        target: Batch,
        osum: Batch,
        left: Batch,
        right: Batch,
        isum: Batch,
        divisor: int,
        factor: int = 1,
    ):
        (tchunk, tstart, tstop) = target
        (ochunk, ostart, ostop) = osum
        (lchunk, lstart, lstop) = left
        (rchunk, rstart, rstop) = right
        (ichunk, istart, istop) = isum
        assert istop - istart == ostop - ostart == 1
        assert tstop - tstart == lstop - lstart == rstop - rstart
        assert ochunk.dtype == ichunk.dtype == UINT64
        assert rchunk.dtype == lchunk.dtype
        super().__init__([target, osum], [left, right, isum])
        self.divisor = divisor
        self.factor = factor

    def run(self):
        [(tchunk, tstart, tstop), (ochunk, ostart, _)] = self.targets
        [(lchunk, lstart, _), (rchunk, rstart, _), (ichunk, istart, _)] = self.sources
        ochunk[ostart] = ichunk[istart]
        for i in range(tstop - tstart):
            tchunk[tstart + i] = (ochunk[ostart] * self.factor) // self.divisor
            ochunk[ostart] -= lchunk[lstart + i]
            ochunk[ostart] += rchunk[rstart + i]


# Mapping Over Batches


DUMMY_CHUNK = Chunk((0, 1, 1), UINT8)


def Map(action: Callable, *iterables: Iterable[Batch]) -> None:
    "Apply the supplied action on data from all the supplied batches."
    nargs = len(iterables)
    iterators = [iter(iterable) for iterable in iterables]
    chunks: list[Chunk] = [DUMMY_CHUNK] * nargs
    starts: list[int] = [0] * nargs
    stops: list[int] = [0] * nargs
    exhausted: list[bool] = [False] * nargs

    def probe(index) -> int:
        """
        Returns how many consecutive elements can be accessed in that index.  If
        there are zero remaining elements, but the iterator hasn't been
        exhausted, load the next batch from that iterator.
        """
        if exhausted[index]:
            return 0
        while (count := (stops[index] - starts[index])) == 0:
            try:
                (chunk, start, stop) = next(iterators[index])
            except StopIteration:
                exhausted[index] = True
                return 0
            chunks[index] = chunk
            starts[index] = start
            stops[index] = stop
        return count

    def pop(index, amount) -> Batch:
        """
        Returns a batch over the selected amount of consecutive elements in that
        index, and advance the start of that index accordingly.
        """
        assert stops[index] >= starts[index] + amount
        batch = Batch(chunks[index], starts[index], starts[index] + amount)
        starts[index] += amount
        return batch

    while (amount := min(probe(index) for index in range(nargs))) > 0:
        action(*[pop(index, amount) for index in range(nargs)])
    if not all(exhausted):
        raise RuntimeError("Attempt to map over sequences of varying length.")
