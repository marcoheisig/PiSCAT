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
        vector = np.linspace(0, 2**precision, endpoint=False, num=f, dtype=dtype)
        array = np.broadcast_to(vector.reshape(f, 1, 1), (f, h, w))
    elif variant == "random":
        array = np.random.randint(0, 2**precision - 1, shape, dtype=dtype)
    else:
        raise ValueError(f"Invalid variant: {variant}")
    return Video(da.from_array(array), precision)


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
    def test(raw: bytes, dtype, expected):
        expected = np.array(expected, dtype=np.uint64)
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td, "data.raw")
            path.write_bytes(raw)
            video = Video.from_raw_file(path, expected.shape, dtype)
            assert np.array_equal(video._array.compute(), expected)

    b1 = bytearray.fromhex("00 01 02 03 04 ff")
    test(b1, np.uint8, [[[0, 1, 2], [3, 4, 255]]])
    test(b1, np.uint8, [[[0, 1], [2, 3], [4, 255]]])
    test(b1, np.uint8, [[[0, 1, 2]], [[3, 4, 255]]])
    test(b1, np.int8, [[[128, 129, 130]], [[131, 132, 127]]])
    b2 = bytearray.fromhex("0000 0001 0002 0003 0004 0005")
    test(b2, ">u2", [[[0, 1, 2], [3, 4, 5]]])
    test(b2, ">u2", [[[0, 1], [2, 3], [4, 5]]])
    test(b2, ">u2", [[[0, 1, 2]], [[3, 4, 5]]])
    b3 = bytearray.fromhex("00000000 00000001 00000002 00000003 00000004 00000005")
    test(b3, ">u4", [[[0, 1, 2], [3, 4, 5]]])
    test(b3, ">u4", [[[0, 1], [2, 3], [4, 5]]])
    test(b3, ">u4", [[[0, 1, 2]], [[3, 4, 5]]])


def test_video_indexing():
    pass


def test_video_rolling_average():
    def ravg(array, window_size, bits):
        size = array.shape[0] - window_size + 1
        x = np.zeros((size, *array.shape[1:]), dtype=object)
        for pos in range(window_size):
            x += array[pos : pos + size]
        assert len(x) == size
        return (x << bits) // window_size

    for n in (1, 3, 7):
        for precision in (1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 62, 63, 64):
            v1 = generate_video((n, 1, 1), precision=precision, variant="linear")
            v2 = generate_video((n, 1, 1), precision=precision, variant="random")
            for video in (v1, v2):
                for window_size in (1, 2, 3, n):
                    if window_size > n:
                        continue
                    bits = math.ceil(math.log2(window_size))
                    rshift = max(0, precision + bits - 64)
                    for chunk_size in range(1, 2, 4):
                        v = Video(video._array.rechunk((chunk_size, -1, -1)), video.precision)
                        ra = v.rolling_average(window_size)
                        result = ra._array.compute()
                        expected = ravg(v._array.compute() >> rshift, window_size, bits)
                        assert np.array_equal(result, expected)
                        dra = v.differential_rolling_average(window_size)
                        result = dra._array.compute()
                        if ra.precision == 64:
                            x = expected >> 1
                            p = 64
                        else:
                            x = expected
                            p = ra.precision + 1
                        expected = (x[0:-window_size] - x[window_size:]) + 2 ** (p - 1) - 1
                        assert np.array_equal(result, expected)


def test_video_to_array():
    def test(numbers, precision, dtype, expected_numbers):
        array = np.array([[numbers]], precision_dtype(precision))
        expected = np.array([[expected_numbers]], dtype=dtype)
        video = Video.from_array(array, precision=precision)
        assert np.array_equal(video.to_array(expected.dtype), expected)

    test([0, 1, 2**8 - 1], 8, np.uint8, [0, 1, 2**8 - 1])
    test([0, 1, 2**16 - 1], 16, np.uint16, [0, 1, 2**16 - 1])
    test([0, 1, 2**32 - 1], 32, np.uint32, [0, 1, 2**32 - 1])
    test([0, 1, 2**64 - 1], 64, np.uint64, [0, 1, 2**64 - 1])
    test([0, 1, 2**8 - 1], 8, np.int8, [-128, -127, 127])
    test([0, 2**64 - 1], 64, np.int64, [-(2**63), 2**63 - 1])
    test([0, 255], 8, np.float32, [0.0, 1.0])
    test([0, 255], 8, np.float64, [0.0, 1.0])


def test_video_to_file():
    pass


def test_video_to_raw_file():
    def test(data, precision, dtype, expected):
        array = da.from_array(np.array(data, precision_dtype(precision)))
        video = Video(array, precision)
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td, "data.raw")
            video.to_raw_file(path, dtype)
            with open(path, "rb") as f:
                data = bytearray(f.read())
        assert data == expected

    b1 = bytearray.fromhex("00 01 02 03 04 ff")
    test([[[0, 1, 2], [3, 4, 255]]], 8, np.uint8, b1)
    test([[[0, 1], [2, 3], [4, 255]]], 8, np.uint8, b1)
    test([[[0, 1, 2]], [[3, 4, 255]]], 8, np.uint8, b1)
    test([[[128, 129, 130]], [[131, 132, 127]]], 8, np.int8, b1)
    b2 = bytearray.fromhex("0000 0001 0002 0003 0004 0005")
    test([[[0, 1, 2], [3, 4, 5]]], 16, ">u2", b2)
    test([[[0, 1], [2, 3], [4, 5]]], 16, ">u2", b2)
    test([[[0, 1, 2]], [[3, 4, 5]]], 16, ">u2", b2)
    data = [[[2**15 + 0, 2**15 + 1, 2**15 + 2]], [[2**15 + 3, 2**15 + 4, 2**15 + 5]]]
    test(data, 16, ">i2", b2)
