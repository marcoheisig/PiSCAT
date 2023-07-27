from __future__ import annotations

from typing import Any, Iterable, Iterator

from piscat.video.actions import Copy
from piscat.video.baseclass import Dtype, ceildiv
from piscat.video.evaluation import Batch, Chunk


def map_batches(
    batches: Iterable[Batch],
    shape: tuple[int, int, int],
    dtype: Dtype,
    action: Any,
    offset: int = 0,
    count: int | None = None,
    step: int = 1,
) -> Iterator[Chunk]:
    """
    Apply the action to all frames in the supplied batches.

    Returns an iterator over the resulting video chunks.

    :param batches: An iterable over batches that supply all the data.
    :param shape: The shape of each resulting chunk.
    :param precision: The number of bits of information per pixel.
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

    :returns: An iterator over chunks of the supplied shape and precision.
    """
    batches = iter(batches)
    if count == 0:
        return
    group: list[Batch] = []  # The batches that constitute the next chunk.
    gstop = 0  # The last frame in the current group.
    cs = shape[0]
    cn = 0  # The amount of chunks that have already been created.

    def make_chunk(batches: list[Batch], tstart: int, tstop: int):
        assert 0 <= tstart <= tstop <= cs
        # Optimization: Eliminate superfluous copy operations.
        if action is Copy and len(batches) == 1:
            [(schunk, sstart, sstop)] = batches
            assert schunk.dtype == dtype
            if schunk.shape == shape and sstart == tstart and sstop == tstop:
                return schunk
        chunk = Chunk(shape, dtype)
        position = tstart
        for source in batches:
            count = source.stop - source.start
            action(Batch(chunk, position, position + count), source)
            position += count
        return chunk

    while count is None or (cn * cs - offset) < count:
        gstart = max(0, (cn * cs - offset) * step)
        cstart = max(cn * cs, offset)
        cstop = (cn + 1) * cs if not count else min((cn + 1) * cs, offset + count)
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
                        yield make_chunk(group, 0, amount)
                    return
            (chunk, start, stop) = batch
            assert 0 <= start <= stop <= chunk.shape[0]
            gstop += stop - start
            if gstop > gstart:
                group.append(Batch(chunk, max(stop - (gstop - gstart), start), stop))
        # Correct the last batch.
        rest = gstop - (gstart + clen * step)
        if rest > 0:
            (chunk, start, stop) = group[-1]
            group[-1] = Batch(chunk, start, stop - rest)
        # Turn all batches of the current group into a chunk.
        yield make_chunk(group, cstart - cn * cs, cstop - cn * cs)
        # Prepare for the next iteration.
        if rest > 0:
            group = [Batch(chunk, stop - rest, stop)]  # type: ignore
        else:
            group = []
        cn += 1
