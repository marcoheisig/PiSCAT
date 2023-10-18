from __future__ import annotations

import math
import pathlib
import tempfile

import dask.array as da
import numpy as np

from piscat.video import Video
from piscat.video.baseclass import precision_dtype


def generate_video(
    shape: tuple[int, int, int] = (2, 3, 4), precision: int = 8, variant: str = "linear"
):
    (f, h, w) = shape
    dtype = precision_dtype(precision)
    if variant == "linear":
        stop = 2**precision - 1
        step = math.ceil(stop / f) if f > 0 else 1
        vector = np.array(range(0, stop, step), dtype=dtype)
        assert len(vector) == f
        array = np.broadcast_to(vector.reshape(f, 1, 1), (f, h, w))
    else:
        raise ValueError(f"Invalid variant: {variant}")
    return Video.from_array(array)


def test_video_change_precision():
    def test(value, precision, expected_precision, expected_value):
        array = np.array([[[value]]], dtype=precision_dtype(precision))
        video = Video(da.from_array(array), precision)
        value = video.change_precision(expected_precision)._array[0, 0, 0].compute()
        assert value == expected_value

    test(0, 8, 16, 0)
    test(1, 8, 9, 2)
    test(2, 8, 9, 4)
    test(1, 9, 8, 0)
    test(2, 9, 8, 1)
    test(255, 8, 8, 255)
    test(255, 8, 16, 255 << 8)
    test(255 << 8, 16, 8, 255)
    test(0, 8, 32, 0)
    test(2**32 - 1, 32, 31, 2**31 - 1)
    test(1, 1, 64, 2**63)


def test_video_concatenate():
    def test(*videos):
        cat = Video.concatenate(videos)
        assert len(cat) == sum(len(video) for video in videos)
        pos = 0
        for video in videos:
            n = len(video)
            expected = video.to_array()
            assert np.array_equal(cat[pos : pos + n].to_array(), expected)
            pos += n

    v1 = generate_video(shape=(2, 5, 7))
    v2 = generate_video(shape=(3, 13, 31))
    v3 = generate_video(shape=(5, 128, 128))
    v4 = generate_video(shape=(9, 128, 128))
    test(v1, v1)
    test(v2, v2)
    test(v3, v4)


def test_video_from_array():
    def test(numbers, dtype, results, lo=None, hi=None, precision=None):
        array = np.array([[numbers]], dtype)
        video = Video.from_array(array, lo=lo, hi=hi, precision=precision)
        assert isinstance(video, Video)
        if precision is not None:
            assert video.precision == precision
        for a, b in zip(video._array.compute()[0, 0], results):
            assert a == b

    test([0, 1, 2**8 - 1], np.uint8, [0, 1, 255])
    test([0, 1, 10], np.uint8, [0, 1, 15], hi=10)
    test([0, 1, 2**8 - 1], np.uint8, [0, 4, 2**10 - 1], precision=10)
    test([0, 1, 2**16 - 1], np.uint16, [0, 1, 2**16 - 1])
    test([0, 1, 2**32 - 1], np.uint32, [0, 1, 2**32 - 1])
    test([0, 1, 2**64 - 1], np.uint64, [0, 1, 2**64 - 1])
    test([-(2**7), 0, 2**7 - 1], np.int8, [0, 2**7, 2**8 - 1])
    test([-(2**15), 0, 2**15 - 1], np.int16, [0, 2**15, 2**16 - 1])
    test([-(2**31), 0, 2**31 - 1], np.int32, [0, 2**31, 2**32 - 1])
    test([-(2**63), 0, 2**63 - 1], np.int64, [0, 2**63, 2**64 - 1])
    test([0.0, 0.5, 1.0], np.float32, [0, 2**23 - 1, 2**24 - 1])
    test([0.0, 0.5, 1.0], np.float64, [0, 2**52 - 1, 2**53 - 1])
    test([-10.0, 0.0, 10.0], np.float64, [0, 2**52 - 1, 2**53 - 1], lo=-10.0, hi=10.0)


def test_video_from_file():
    pass


def test_video_from_raw_file():
    def test(raw: bytes, expected: np.ndarray):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td, "data.raw")
            path.write_bytes(raw)
            video = Video.from_raw_file(path, expected.shape, expected.dtype)
            assert np.array_equal(video.to_array(expected.dtype), expected)

    test(b"\x00\x01\x02\x03\x04\x05", np.array([[[0, 1, 2], [3, 4, 5]]], dtype=np.uint8))
    test(b"\x00\x01\x02\x03\x04\x05", np.array([[[0, 1], [2, 3], [4, 5]]], dtype=np.uint8))
    test(b"\x00\x01\x02\x03\x04\x05", np.array([[[0, 1, 2]], [[3, 4, 5]]], dtype=np.uint8))


def test_video_indexing():
    pass


def test_video_rolling_average():
    pass


def test_video_to_array():
    pass


def test_video_to_file():
    pass
