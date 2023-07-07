from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np

from piscat.video.evaluation import Batch, Kernel, VideoChunk, VideoOp, ceildiv, copy_kernel


def map_batches(
    batches: Iterable[Batch],
    /,
    shape: tuple[int, int, int],
    dtype: np.dtype,
    kernel: Kernel,
    offset: int = 0,
    count: int | None = None,
    step: int = 1,
) -> Iterator[VideoChunk]:
    """
    Apply the kernel to all frames in the supplied batches.

    Returns an iterator over the resulting video chunks.

    :param batches: An iterable over batches that supply all the data.
    :param shape: The shape of each resulting chunk.
    :param dtype: The type of each element of each resulting chunk.
    :param kernel: The function that reads data from one or more batches, and
        that writes its results to a single target batch that initializes one
        resulting chunk.
    :param offset: The number of frames of the first resulting chunk that are
        left uninitialized.
    :param count: The total number of target frames that will be defined, or
        None if this function should create chunks until the supplied batches
        are exhausted.
    :param step: How many input frames constitute one output frame.  Requires a
        suitable kernel.

    :returns: An iterator over chunks of the supplied shape and dtype.
    """
    batches = iter(batches)
    if count == 0:
        return
    (f, h, w) = shape
    group: list[Batch] = []  # The batches that constitute the next chunk.
    gstop = 0  # The last frame in the current group.
    cn = 0  # The amount of chunks that have already been created.

    def merge_batches(batches: list[Batch], start: int, stop: int):
        if kernel is copy_kernel and len(batches) == 1:
            (chunk, cstart, cstop) = batches[0]
            if chunk.shape == shape and cstart == start and cstop == stop:
                return chunk
        chunk = VideoChunk(shape=shape, dtype=dtype)
        assert 0 <= start <= stop <= shape[0]
        VideoOp(kernel=kernel, targets=[Batch(chunk, start, stop)], sources=batches)
        return chunk

    while count is None or (cn * f - offset) < count:
        gstart = max(0, (cn * f - offset) * step)
        cstart = max(cn * f, offset)
        cstop = (cn + 1) * f if not count else min((cn + 1) * f, offset + count)
        clen = cstop - cstart
        # Gather enough source chunks to fill one target chunk.
        while (amount := ceildiv(gstop, step) - ceildiv(gstart, step)) < clen:
            try:
                batch = next(batches)
            except StopIteration as e:
                if count:
                    raise ValueError("Not enough input chunks.") from e
                else:
                    if amount > 0:
                        yield merge_batches(group, start=0, stop=amount)
                    return
            (chunk, start, stop) = batch
            (cf, ch, cw) = chunk.shape
            assert 0 <= start <= stop <= cf
            gstop += stop - start
            if gstop > gstart:
                group.append(Batch(chunk, max(stop - (gstop - gstart), start), stop))
        # Correct the last batch.
        rest = gstop - (gstart + clen * step)
        if rest > 0:
            (chunk, start, stop) = group[-1]
            group[-1] = Batch(chunk, start, stop - rest)
        # Turn all batches of the current group into a chunk.
        yield merge_batches(group, cstart - cn * f, cstop - cn * f)
        # Prepare for the next iteration.
        if rest > 0:
            group = [Batch(chunk, stop - rest, stop)]  # type: ignore
        else:
            group = []
        cn += 1
