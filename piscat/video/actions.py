from __future__ import annotations

from typing import Callable

import numpy as np

from piscat.video.baseclass import Precision, precision_dtype
from piscat.video.evaluation import Action, Batch, Dtype

### Copying Data


class Copy(Action):
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
        assert target.chunk.dtype == np.dtype(np.uint32)
        assert source.chunk.dtype == np.dtype(np.uint64)
        super().__init__(target, source)

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        [(schunk, sstart, sstop)] = self.sources
        tchunk[tstart:tstop] = schunk[sstart:sstop] >> 32


class DecodeI8Array(DecodeArray):
    def __init__(self, target: Batch, source: Batch):
        assert target.chunk.dtype == np.dtype(np.uint8)
        assert source.chunk.dtype == np.dtype(np.int8)
        super().__init__(target, source)

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        [(schunk, sstart, sstop)] = self.sources
        tchunk[tstart:tstop] = schunk[sstart:sstop].view(np.uint8) + 2**7


class DecodeI16Array(DecodeArray):
    def __init__(self, target: Batch, source: Batch):
        assert target.chunk.dtype == np.dtype(np.uint16)
        assert source.chunk.dtype == np.dtype(np.int16)
        super().__init__(target, source)

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        [(schunk, sstart, sstop)] = self.sources
        tchunk[tstart:tstop] = schunk[sstart:sstop].view(np.uint16) + 2**15


class DecodeI32Array(DecodeArray):
    def __init__(self, target: Batch, source: Batch):
        assert target.chunk.dtype == np.dtype(np.uint32)
        assert source.chunk.dtype == np.dtype(np.int32)
        super().__init__(target, source)

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        [(schunk, sstart, sstop)] = self.sources
        tchunk[tstart:tstop] = schunk[sstart:sstop].view(np.uint32) + 2**31


class DecodeI64Array(DecodeArray):
    def __init__(self, target: Batch, source: Batch):
        assert target.chunk.dtype == np.dtype(np.uint64)
        assert source.chunk.dtype == np.dtype(np.int64)
        super().__init__(target, source)

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        [(schunk, sstart, sstop)] = self.sources
        tchunk[tstart:tstop] = (schunk[sstart:sstop].view(np.uint64) + 2**63) >> 32


class DecodeF32Array(DecodeArray):
    def __init__(self, target: Batch, source: Batch):
        assert target.chunk.dtype == np.dtype(np.uint32)
        assert source.chunk.dtype == np.dtype(np.float32)
        super().__init__(target, source)

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        [(schunk, sstart, sstop)] = self.sources
        data = schunk[sstart:sstop] * 2**32
        tchunk[tstart:tstop] = np.clip(data, 0, 2**32 - 1).astype(np.uint32)


class DecodeF64Array(DecodeArray):
    def __init__(self, target: Batch, source: Batch):
        assert target.chunk.dtype == np.dtype(np.uint32)
        assert source.chunk.dtype == np.dtype(np.float64)
        super().__init__(target, source)

    def run(self):
        [(tchunk, tstart, tstop)] = self.targets
        [(schunk, sstart, sstop)] = self.sources
        data = schunk[sstart:sstop] * 2**32
        tchunk[tstart:tstop] = np.clip(data, 0, 2**32 - 1).astype(np.uint32)


Decoder = Callable[[Batch, Batch], Action]


def dtype_decoder_and_precision(dtype: Dtype) -> tuple[Decoder, Precision]:
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


### Change Video Precision


class ChangePrecision(Action):
    shift: int

    def __init__(
        self,
        target: Batch,
        source: Batch,
        target_precision: Precision,
        source_precision: Precision,
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
