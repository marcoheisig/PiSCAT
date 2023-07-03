from __future__ import annotations

import pathlib
from typing import Iterator, Union, overload

import numpy as np
from typing_extensions import Self

from piscat.video.baseclass import Video
from piscat.video.evaluation import Array, Batches, Kernel, VideoChunk, copy_kernel
from piscat.video.map_batches import map_batches

Path = Union[str, pathlib.Path]

EllipsisType = type(Ellipsis)

Slice = Union[slice, EllipsisType]


class Video_getitem(Video):
    @overload
    def __getitem__(self, index: int) -> Array:
        ...

    @overload
    def __getitem__(self, index: Slice) -> Self:
        ...

    @overload
    def __getitem__(self, index: tuple[int, ...]) -> Array:
        ...

    @overload
    def __getitem__(self, index: tuple[Slice, ...]) -> Self:
        ...

    def __getitem__(self, index):
        """
        Return a particular part of the video.
        """
        nframes = self.shape[0]
        (csize, h, w) = self.chunk_shape
        chunks = self._chunks

        def icheck(index):
            if not (0 <= index < nframes):
                raise IndexError(f"Invalid index {index} for a video with {nframes} frames.")

        if isinstance(index, int):
            icheck(index)
            p = self.chunk_offset + index
            chunk = chunks[p // csize]
            return chunk[p % csize]
        elif isinstance(index, EllipsisType):
            return self
        elif isinstance(index, slice):
            istart = 0 if index.start is None else index.start
            istop = len(self) if index.stop is None else max(istart, index.stop)
            istep = 1 if index.step is None else index.step
            if istart == istop:
                array = np.ndarray(shape=(0, h, w), dtype=self.dtype)
                chunk = VideoChunk(shape=(0, h, w), dtype=self.dtype, data=array)
                return type(self)([chunk], 0, 0)
            if istart == 0 and istop == nframes and istep == 1:
                return self
            icheck(istart)
            total = len(range(istart, istop, istep))
            ilast = max(istart, istart + (total - 1) * istep)
            icheck(ilast)
            pstart = self.chunk_offset + istart
            plast = self.chunk_offset + ilast
            offset = pstart % csize
            if istep == 1:
                selection = chunks[pstart // csize : (plast // csize) + 1]
                return type(self)(selection, offset, total)
            else:
                shape = (total, h, w)
                chunk_size = Video.plan_chunk_size(shape=shape, dtype=self.dtype)
                return type(self)(
                    map_batches(
                        self.batches(istart, istop),
                        shape=(chunk_size, h, w),
                        dtype=self.dtype,
                        kernel=get_slice_kernel(istep),
                        step=istep,
                        count=total,
                    ),
                    0,
                    total,
                )
        elif isinstance(index, tuple):
            i0 = index[0]
            if isinstance(i0, int):
                return self[i0].__getitem__(index[1:])
            else:
                # TODO
                return np.array(self)[index]
        else:
            raise ValueError(f"Invalid video index: {index}")

    def __iter__(self) -> Iterator[Array]:
        for index in range(len(self)):
            yield self[index]


def get_slice_kernel(step) -> Kernel:
    assert step > 0
    if step == 1:
        return copy_kernel

    def kernel(targets: Batches, sources: Batches):
        assert len(targets) == 1
        (target, tstart, tstop) = targets[0]
        pos = tstart
        offset = 0
        for source, start, stop in sources:
            count = len(range(start + offset, stop, step))
            target[pos : pos + count] = source[start + offset : stop : step]
            pos += count
            offset = start + offset + count * step - stop
        assert pos == tstop

    return kernel
