from __future__ import annotations

import os
import pathlib
import tempfile
from itertools import chain

import numpy as np
import numpy.typing as npt

from piscat.video import Batch, Video, VideoChunk, VideoOp, copy_kernel


def check_chunk(chunk: VideoChunk):
    assert len(chunk.shape) == 3
    assert len(chunk) == chunk.shape[0]
    data = chunk._data
    assert data is not None
    if isinstance(data, VideoOp):
        assert any(target.chunk is chunk for target in data.targets)
    else:
        assert data.shape == chunk.shape
        assert data.dtype == chunk.dtype


def check_kernel(kernel: VideoOp):
    for target, tstart, tstop in kernel.targets:
        assert 0 <= tstart < target.shape[0]
        assert 0 <= tstop <= target.shape[0]
        data = target._data
        if isinstance(data, VideoOp):
            assert data is kernel
    for source, sstart, sstop in kernel.sources:
        data = source._data
        assert 0 <= sstart < source.shape[0]
        assert 0 <= sstop <= source.shape[0]
        assert kernel in source._users


def check_video(video: Video):
    nchunks = len(video._chunks)
    csize = video.chunk_shape[0]
    assert video.chunk_offset < csize
    (length, h, w) = video.shape
    assert (nchunks - 1) * csize < video.chunk_offset + length <= nchunks * csize
    assert length == len(video)
    assert h == video.shape[1]
    assert w == video.shape[2]
    for chunk in video._chunks:
        assert chunk.shape == video.chunk_shape
        assert chunk.dtype == video.dtype


def iota_array(
    start: int,
    stop: int,
    step: int = 1,
    h: int = 100,
    w: int = 100,
    dtype: npt.DTypeLike = np.uint16,
):
    vec = np.arange(start, stop, step, dtype=dtype)
    return np.broadcast_to(vec.reshape(len(vec), 1, 1), (len(vec), h, w))


def test_chunk():
    # Test the creation of a chunk with known contents.
    shape = (7, 5, 3)
    (f, h, w) = shape
    array = np.array(list(float(i) for i in range(f * h * w))).reshape((f, h, w))
    chunk = VideoChunk.from_array(array)
    check_chunk(chunk)
    np.array_equal(chunk.data, array)
    for i in range(f):
        np.array_equal(chunk[i], array[i])
    for i in range(f):
        np.array_equal(chunk[i : f - 1], array[i : f - 2])

    # Test the creation of a chunk by copying an existing chunk.
    copy = VideoChunk(shape=shape, dtype=array.dtype)
    kernel = VideoOp(copy_kernel, [Batch(copy, 0, f)], [Batch(chunk, 0, f)])
    check_kernel(kernel)
    assert np.array_equal(copy.data, array)


def test_video_from_frame():
    frame = np.array(list(float(i) for i in range(1, 10))).reshape((3, 3))
    video = Video.from_frame(frame, 24)
    check_video(video)
    for fr in video:
        assert np.array_equal(frame, fr)


def test_video_from_array():
    for h, w in [(2, 3), (128, 128)]:
        for length in range(1, 256):
            array = iota_array(0, length, h=h, w=w)
            video = Video.from_array(array)
            check_video(video)
            for index, frame in enumerate(array):
                assert np.array_equal(video[index], frame)


def test_video_concatenate():
    for h, w in [(13, 15), (128, 128)]:
        for count in range(1, 10, 3):
            pos = 0
            arrays = []
            for _ in range(count):
                n = np.random.randint(20)
                arrays.append(iota_array(pos, pos + n, h=h, w=w, dtype=np.float64))
                pos += n
            videos = [Video.from_array(array) for array in arrays]
            video = Video.concatenate(videos)
            pos = 0
            for array in arrays:
                n = array.shape[0]
                assert np.array_equal(video[pos : pos + n], array)
                pos += n


def test_video_indexing():
    # Reversal
    for dim in range(7):
        for h, w in [(10, 10), (50, 100), (100, 50), (640, 480)]:
            array = iota_array(0, dim, h=h, w=w)
            video = Video.from_array(array)
            assert np.array_equal(video[::-1], array[::-1])
    # Selecting parts of the first axis.
    for h, w in [(3, 4), (7, 6), (50, 70)]:
        for chunk_size in [1, 2, 3]:
            for dim in range(4):
                for start in chain(range(-dim - 2, dim + 2), [None]):
                    for stop in chain(range(-dim - 2, dim + 2), [None]):
                        for step in chain(range(-dim, -2), [None], range(1, dim)):
                            array = iota_array(0, dim, 1, h=h, w=w)
                            video = Video.from_array(array, chunk_size=chunk_size)
                            assert np.array_equal(video[start:stop:step], array[start:stop:step])
    # Selecting parts of the other axes
    for h, w in [(3, 4), (7, 6), (50, 70)]:
        for chunk_size in [1, 2, 3]:
            for f in range(4):
                array = np.random.random((f, h, w))
                video = Video.from_array(array, chunk_size=chunk_size)
                assert np.array_equal(video[:, 0:1, 0:1], array[:, 0:1, 0:1])
                assert np.array_equal(video[:, ::-2, ::-3], array[:, ::-2, ::-3])


def test_video_io():
    a1 = iota_array(0, 100, dtype=np.dtype("u1"))
    a2 = a1.astype(np.dtype("u2")) * 256
    a3 = a1 / np.array(256, np.dtype("f4"))
    v1 = Video.from_array(a1)
    v2 = Video.from_array(a2)
    v3 = Video.from_array(a3)
    with tempfile.TemporaryDirectory() as td:
        vn = 0
        for video in [v1, v2, v3]:
            for suffix in ["npy", "raw", "mp4"]:
                for flush in [True, False]:
                    path = os.path.join(td, f"video{vn}.{suffix}")
                    vn += 1
                    # Write the video to a file.
                    video.to_file(path, flush=flush)
                    assert pathlib.Path(path).exists()
                    # Read the video back in.
                    if suffix == "raw":
                        other = Video.from_raw_file(path, video.shape, video.dtype)
                    else:
                        other = Video.from_file(path)
                    assert video.dtype == other.dtype
                    assert video.shape == other.shape
                    assert np.allclose(video, other, atol=1e-5)
