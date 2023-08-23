from __future__ import annotations

from typing import Any, Callable, Iterator, Union, overload

from typing_extensions import Self

from piscat.video.actions import Copy, Reverse, Slice
from piscat.video.baseclass import Video, ceildiv, precision_dtype
from piscat.video.evaluation import Action, Array, Batch, Chunk
from piscat.video.map_batches import map_batches

EllipsisType = Union[type(...), type(...)]  # Pyright doesn't understand a plain type(...).


class Video_indexing(Video):
    @overload
    def __getitem__(self, index: EllipsisType) -> Self:
        ...

    @overload
    def __getitem__(self, index: slice) -> Self:
        ...

    @overload
    def __getitem__(self, index: int) -> Array:
        ...

    @overload
    def __getitem__(self, index: tuple[slice]) -> Self:
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
    chunk_size = video.chunk_size
    precision = video.precision
    fstart, fstop, fstep, fsize = canonicalize_slice(fslice, f)
    _, _, hstep, hsize = canonicalize_slice(hslice, h)
    _, _, wstep, wsize = canonicalize_slice(wslice, w)
    result_dtype = precision_dtype(video.precision)
    result_shape = (fsize, hsize, wsize)
    # Handle empty videos.
    if fsize == 0 or hsize == 0 or wsize == 0:
        array = Array(result_shape, result_dtype)
        chunk = Chunk(result_shape, result_dtype, data=array)
        return type(video)([chunk], result_shape, video.precision)
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
        offset = pstart % chunk_size
        cstart = pstart // chunk_size
        cstop = (plast // chunk_size) + 1
        result = type(video)(video.chunks[cstart:cstop], result_shape, precision, offset)
        return reverse(result) if flip else result
    # In the general case, create new chunks and use a kernel that selects only
    # the relevant parts of each Nth element of its concatenated source batches.
    result_csize = Video.plan_chunk_size(result_shape, precision)
    hslice = slice(0, None) if hsize == h else hslice
    wslice = slice(0, None) if wsize == w else wslice
    action: Callable[[Batch, Batch], Action]
    if fstep == 1 and full_slice_p(hslice) and full_slice_p(wslice):
        action = Copy
    else:
        action = lambda t, s: Slice(t, s, fstep, hslice, wslice)

    chunks = map_batches(
        video.batches(fstart, fstop),
        (result_csize, hsize, wsize),
        result_dtype,
        action=action,
        step=fstep,
        count=fsize,
    )
    result = type(video)(list(chunks), result_shape, precision)
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
        list(
            map_batches(
                reversed(list(video.batches())),
                shape=video.chunk_shape,
                dtype=video.dtype,
                action=Reverse,
                count=vsize,
                offset=new_offset,
            )
        ),
        video.shape,
        video.precision,
        new_offset,
    )


def isntuple(obj, n: int):
    """
    Return whether the supplied object is a tuple of length N.
    """
    return isinstance(obj, tuple) and len(obj) == n


def isslice(obj):
    return isinstance(obj, slice)


def full_slice_p(slc: slice):
    return (
        (slc.start is None or slc.start == 0)
        and (slc.stop is None)
        and (slc.step is None or slc.step == 1)
    )
