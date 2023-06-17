from __future__ import annotations

import os
import pathlib
import tempfile

import numpy as np
import numpy.typing as npt

from piscat.core.video import Batch, Video, VideoChunk, VideoOp, copy_kernel


def check_chunk(chunk: VideoChunk):
    assert len(chunk.shape) == 3
    assert len(chunk) == chunk.shape[0]
    for user in chunk._users:
        assert any(source.chunk is chunk for source in user.sources)
    for video in chunk._videos:
        assert chunk in video._chunks
    data = chunk._data
    assert data is not None
    if isinstance(data, VideoOp):
        assert any(target.chunk is chunk for target in data.targets)
    else:
        assert data.shape == chunk.shape
        assert data.dtype == chunk.dtype


def check_kernel(kernel: VideoOp):
    for target, tstart, tstop, tstep in kernel.targets:
        assert tstep == 1
        assert 0 <= tstart < target.shape[0]
        assert 0 <= tstop <= target.shape[0]
        data = target._data
        if isinstance(data, VideoOp):
            assert data is kernel
    for source, sstart, sstop, sstep in kernel.sources:
        data = source._data
        assert sstep == 1
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


def iota_frames(
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
            array = iota_frames(0, length, h=h, w=w)
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
                arrays.append(iota_frames(pos, pos + n, h=h, w=w, dtype=np.float64))
                pos += n
            videos = [Video.from_array(array) for array in arrays]
            video = Video.concatenate(videos)
            pos = 0
            for array in arrays:
                n = array.shape[0]
                assert np.array_equal(video[pos : pos + n], array)
                pos += n


def test_video_indexing():
    for nchunks in range(1, 5):
        for chunk_shape in [(1, 3, 4), (2, 3, 4), (3, 3, 4)]:
            arrays = [np.random.random(chunk_shape) for _ in range(nchunks)]
            chunks = [VideoChunk.from_array(array) for array in arrays]
            array = np.concatenate(arrays, axis=0)
            for offset in range(chunk_shape[0] - 1):
                video = Video(chunks, offset)
                length = len(video)
                assert np.array_equal(video[0:], array[offset : offset + length])
                half = length // 2
                assert np.array_equal(video[half], array[offset + half])


def test_video_io():
    a1 = iota_frames(0, 100, dtype=np.dtype("u1"))
    a2 = a1.astype(np.dtype("u2")) * 256
    a3 = a1 / np.array(256, np.dtype("f4"))
    v1 = Video.from_array(a1)
    v2 = Video.from_array(a2)
    v3 = Video.from_array(a3)
    with tempfile.TemporaryDirectory() as td:
        vn = 0
        for video in [v1, v2, v3]:
            for suffix in ["npy", "raw", "mp4"]:
                path = os.path.join(td, f"video{vn}.{suffix}")
                vn += 1
                video.to_file(path)
                assert pathlib.Path(path).exists()
                if suffix == "raw":
                    other = Video.from_raw_file(path, video.shape, video.dtype)
                else:
                    other = Video.from_file(path)
                assert video.dtype == other.dtype
                assert video.shape == other.shape
                assert np.allclose(video, other, atol=1e-5)
