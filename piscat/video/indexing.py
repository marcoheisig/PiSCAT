from __future__ import annotations

import pathlib
from typing import Any, Iterator, Union, overload

import numpy as np
from typing_extensions import Self

from piscat.video.baseclass import Video, ceildiv
from piscat.video.evaluation import Array, Batches, Kernel, VideoChunk, copy_kernel
from piscat.video.map_batches import map_batches

Path = Union[str, pathlib.Path]

EllipsisType = type(Ellipsis)

Slice = Union[slice, EllipsisType]


class Video_indexing(Video):
    @overload
    def __getitem__(self, index: Slice) -> Self:
        ...

    @overload
    def __getitem__(self, index: int) -> Array:
        ...

    @overload
    def __getitem__(self, index: tuple[Slice]) -> Self:
        ...

    @overload
    def __getitem__(self, index: tuple[int]) -> Array:
        ...

    @overload
    def __getitem__(self, index: tuple[int, int | slice]) -> Array:
        ...

    @overload
    def __getitem__(self, index: tuple[int, slice, int]) -> Array:
        ...

    @overload
    def __getitem__(self, index: tuple[int, int, slice]) -> Array:
        ...

    @overload
    def __getitem__(self, index: tuple[int, slice, slice]) -> Array:
        ...

    @overload
    def __getitem__(self, index: tuple[int, int, int]) -> Any:
        ...

    @overload
    def __getitem__(self, index: tuple[slice, slice, slice]) -> Self:
        ...

    def __getitem__(self, index):
        """
        Return a particular part of the video.
        """
        # Unpack tuples of length one.
        if isinstance(index, tuple) and len(index) == 1:
            index = index[0]
        # Handle the case of video[...].
        if index is ...:
            return self
        # Handle the case of video[N] for some integer N.
        elif isinstance(index, int):
            vsize = len(self)
            if not (0 <= index < vsize):
                raise IndexError(f"Invalid index {index} for a video with {vsize} frames.")
            csize = self.chunk_size
            p = self.chunk_offset + index
            chunk = self.chunks[p // csize]
            return chunk[p % csize]
        # Handle the case of video[A : B : C].
        elif isinstance(index, slice):
            return select(self, index)
        # Handle the case of video[A, B] where A is an integer.
        elif isntuple(index, 2) and isinstance(index[0], int):
            return self[index[0]][index[1]]
        # Handle the case of video[A, B, C] where A is an integer.
        elif isntuple(index, 3) and isinstance(index[0], int):
            return self[index[0]][index[1], index[2]]
        # Handle the case of video[A, B] where A and B are slices.
        elif isntuple(index, 2) and isslice(index[0]) and isslice(index[1]):
            return select(self, index[0], index[1])
        # Handle the case of video[A, B, C] where A, B, and C are slices.
        elif isntuple(index, 3) and isslice(index[0]) and isslice(index[1]) and isslice(index[2]):
            return select(self, index[0], index[1], index[2])
        # If none of the previous clauses managed to return a suitable result,
        # the index is invalid and we raise an error.
        raise ValueError(f"Invalid video index: {index}")

    def __iter__(self) -> Iterator[Array]:
        for index in range(len(self)):
            yield self[index]


def select(
    video: Video_indexing,
    fslice: slice = slice(0, None),
    hslice: slice = slice(0, None),
    wslice: slice = slice(0, None),
) -> Video_indexing:
    (f, h, w) = video.shape
    csize = video.chunk_size
    fstart, fstop, fstep, fsize = canonicalize_slice(fslice, f)
    _, _, hstep, hsize = canonicalize_slice(hslice, h)
    _, _, wstep, wsize = canonicalize_slice(wslice, w)
    result_shape = (fsize, hsize, wsize)
    # Handle empty videos.
    if fsize == 0 or hsize == 0 or wsize == 0:
        array = np.ndarray(shape=result_shape, dtype=video.dtype)
        chunk = VideoChunk(shape=result_shape, dtype=video.dtype, data=array)
        return type(video)([chunk], 0, 0)
    # Handle references to the entire video.
    if fsize == f and fstep == 1:
        if hsize == h and hstep == 1:
            if wsize == w and wstep == 1:
                return video
    # If the slice in axis 0 has a negative step, reverse that slice and recall
    # to later reverse the result of the selection.
    flip = False
    if fstep < 0:
        fstep = -fstep
        fstop = fstart + 1
        fstart = fstart - fstep * (fsize - 1)
        flip = True
    # Selecting full frames with a frame step size of one can be performed
    # extremely efficiently by reusing chunks of the original video.
    if fstep == 1 and h == hsize and w == wsize:
        pstart = video.chunk_offset + fstart
        plast = pstart + (fsize - 1) * fstep
        offset = pstart % csize
        cstart = pstart // csize
        cstop = (plast // csize) + 1
        result = type(video)(video.chunks[cstart:cstop], offset, fsize)
        return reverse(result) if flip else result
    # In the general case, create new chunks and use a kernel that selects only
    # the relevant parts of each Nth element of its concatenated source batches.
    result_csize = Video.plan_chunk_size(shape=result_shape, dtype=video.dtype)
    hslice = slice(0, None) if hsize == h else hslice
    wslice = slice(0, None) if wsize == w else wslice
    result = type(video)(
        map_batches(
            video.batches(fstart, fstop),
            shape=(result_csize, hsize, wsize),
            dtype=video.dtype,
            kernel=get_slice_kernel(fstep, hslice, wslice),
            step=fstep,
            count=fsize,
        ),
        0,
        fsize,
    )
    return reverse(result) if flip else result


def canonicalize_slice(slc: slice, dim: int) -> tuple[int, int, int, int]:
    """
    Returns a tuple with five values:

        1. The lowest index in the slice.

        2. The highest index of the slice plus something less than the step.

        3. The step size of the slice.

        4. The size of the slice.
    """
    if slc is ...:
        return (0, dim, 1, dim)
    elif isinstance(slc, slice):
        (start, stop, step) = slc.indices(dim)
        size = max(0, ceildiv((stop - start), step))
        if size == 0:
            return (0, 0, 1, 0)
        if size == 1:
            return (start, start + 1, 1, 1)
        else:
            return (start, stop, step, size)
    else:
        raise ValueError(f"Not a slice: {slc}")


def get_slice_kernel(step, hslice: slice, wslice: slice) -> Kernel:
    assert step > 0
    if step == 1 and full_slice_p(hslice) and full_slice_p(wslice):
        return copy_kernel

    def kernel(targets: Batches, sources: Batches):
        assert len(targets) == 1
        (target, tstart, tstop) = targets[0]
        pos = tstart
        offset = 0
        for source, start, stop in sources:
            count = len(range(start + offset, stop, step))
            print(f"{source.shape=}")
            fslice = slice(start + offset, stop, step)
            print(f"{fslice=} {hslice=} {wslice=}")
            print(source[fslice, hslice, wslice])
            target[pos : pos + count, :, :] = source[start + offset : stop : step, hslice, wslice]
            pos += count
            offset = start + offset + count * step - stop
        assert pos == tstop

    return kernel


def reverse(video) -> Video_indexing:
    """
    Return a video whose order of frames is reversed.
    """
    vsize = len(video)
    if vsize <= 1:
        return video
    old_offset = video.chunk_offset
    new_offset = video.chunk_size * len(video.chunks) - vsize - old_offset
    return type(video)(
        map_batches(
            reversed(list(video.batches())),
            shape=video.chunk_shape,
            dtype=video.dtype,
            kernel=reverse_kernel,
            count=vsize,
            offset=new_offset,
        ),
        new_offset,
        vsize,
    )


def reverse_kernel(targets: Batches, sources: Batches):
    [(target, tstart, tstop)] = targets
    pos = tstart
    for source, sstart, sstop in sources:
        count = sstop - sstart
        ssize = len(source)
        target[pos : pos + count] = source[sstop - ssize - 1 : sstart - ssize - 1 : -1]
        pos += count
    assert pos == tstop


def isntuple(obj, n: int):
    """
    Return whether the supplied object is a tuple of length N.
    """
    return isinstance(obj, tuple) and len(obj) == n


def isslice(obj):
    if obj is ...:
        return True
    elif isinstance(obj, slice):
        return True
    else:
        return False


def full_slice_p(slc: slice):
    return (
        (slc.start is None or slc.start == 0)
        and (slc.stop is None)
        and (slc.step is None or slc.step == 1)
    )
