from __future__ import annotations

import os
import pathlib
import tempfile
from itertools import chain

import numpy as np
import numpy.typing as npt

from piscat.video import Action, Batch, Chunk, Video
from piscat.video.actions import Copy


def check_chunk(chunk: Chunk):
    # Check that __length__ behaves correctly.
    assert len(chunk) == chunk.shape[0]
    # Check that each chunk is connected to the actions writing to it.
    for action in chunk.actions:
        assert any(target.chunk is chunk for target in action.targets)
    if chunk._data is not None:
        assert chunk._data.shape == chunk.shape
        assert chunk._data.dtype == chunk.dtype


def check_action(action: Action):
    for target, tstart, tstop in action.targets:
        assert 0 <= tstart < target.shape[0]
        assert 0 <= tstop <= target.shape[0]
    for source, sstart, sstop in action.sources:
        assert 0 <= sstart < source.shape[0]
        assert 0 <= sstop <= source.shape[0]


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
    chunk = Chunk(array.shape, array.dtype, array)
    check_chunk(chunk)
    np.array_equal(chunk.data, array)
    for i in range(f):
        np.array_equal(chunk[i], array[i])
    for i in range(f):
        np.array_equal(chunk[i : f - 1], array[i : f - 2])

    # Test the creation of a chunk by copying an existing chunk.
    copy = Chunk(shape=shape, dtype=array.dtype)
    action = Copy(Batch(copy, 0, f), Batch(chunk, 0, f))
    check_action(action)
    assert np.array_equal(copy.data, array)


def test_video_from_frame():
    frame = np.array(list(float(i) for i in range(1, 10)), dtype=np.uint8).reshape((3, 3))
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


def test_video_change_precision():
    v8 = Video.from_array(iota_array(0, 2**8 - 1, 2**0, dtype=np.uint8))
    v16 = Video.from_array(iota_array(0, 2**16 - 1, 2**8, dtype=np.uint16))
    v24 = Video.from_array(iota_array(0, 2**24 - 1, 2**16, dtype=np.uint32))
    for _video, precision in [(v8, 8), (v16, 16), (v24, 24)]:
        for other_precision in (1, 2, 3, 8, 9, 10, 12, 15, 16, 17, 23, 24):
            if other_precision > precision:
                pass  # TODO
            else:
                pass  # TODO


def test_video_concatenate():
    for h, w in [(13, 15), (128, 128)]:
        for count in range(1, 10, 3):
            pos = 0
            arrays = []
            for _ in range(count):
                n = np.random.randint(20)
                arrays.append(iota_array(pos, pos + n, h=h, w=w, dtype=np.uint16))
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
    for h, w in [(3, 4), (7, 6), (50, 70), (128, 128), (512, 512)]:
        for dim in range(4):
            for start in chain(range(-dim - 2, dim + 2), [None]):
                for stop in chain(range(-dim - 2, dim + 2), [None]):
                    for step in chain(range(-dim, -2), [None], range(1, dim)):
                        array = iota_array(0, dim, 1, h=h, w=w)
                        video = Video.from_array(array)
                        assert np.array_equal(video[start:stop:step], array[start:stop:step])
    # Selecting parts of the other axes
    for h, w in [(3, 4), (7, 6), (50, 70), (128, 128)]:
        for f in range(4):
            for dtype, precision in [(np.uint8, 8), (np.uint16, 16)]:
                array = (np.random.random((f, h, w)) * 2**precision).astype(dtype)
                video = Video.from_array(array)
                assert np.array_equal(video[:, 0:1, 0:1], array[:, 0:1, 0:1])
                assert np.array_equal(video[:, ::-2, ::-3], array[:, ::-2, ::-3])


def test_video_io():
    a1 = np.zeros((100, 2, 2), dtype=np.uint8)
    a2 = np.zeros((100, 2, 2), dtype=np.uint8)
    a2.fill(255)
    a3 = np.zeros((100, 2, 2), dtype=np.uint32)
    a3.fill(2**32 - 1)
    a4 = iota_array(0, 100, dtype=np.dtype("u1"))
    a5 = a4.astype(np.uint16) * 256
    a6 = a4.astype(np.uint32) * 256**3
    with tempfile.TemporaryDirectory() as td:
        vn = 0
        for array in [a1, a2, a3, a4, a5, a6]:
            video = Video.from_array(array)
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
                    assert video.precision == other.precision
                    assert video.shape == other.shape
                    assert np.allclose(video, other, rtol=0.001)
