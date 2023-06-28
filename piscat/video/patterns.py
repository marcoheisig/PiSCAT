from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np

from piscat.video.evaluation import Batch, Kernel, VideoChunk, VideoOp


def map_batches(
    batches: Iterable[Batch],
    /,
    shape: tuple[int, int, int],
    dtype: np.dtype,
    kernel: Kernel,
    offset: int = 0,
    count: int | None = None,
) -> Iterator[VideoChunk]:
    """
    Apply the kernel to all frames in the supplied batches.

    Returns an iterator over the resulting video chunks.

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
        (last, lstart, lstop) = group[-1]
        group[-1] = Batch(last, lstart, lstop - rest)
        yield merge_batches(group, start=cstart - ccount * f, stop=cend - ccount * f)
        # Prepare for the next iteration.
        if rest == 0:
            group = []
        else:
            group = [Batch(last, lstop - rest, lstop)]
        gstart += clen
        ccount += 1
