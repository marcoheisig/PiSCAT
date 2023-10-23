from __future__ import annotations

from typing import Union, overload

from typing_extensions import Self

from piscat.video.baseclass import Video
from piscat.video.to_array import array_from_video_data

EllipsisType = Union[type(...), type(...)]  # Pyright doesn't understand a plain type(...).


class VideoPart:
    _array: Video_indexing
    _precision: int

    def __init__(self, array, precision):
        self._array = array
        self._precision = precision

    def to_array(self, dtype=None):
        return array_from_video_data(self._array, self._precision, dtype).compute()

    @property
    def precision(self) -> int:
        return self._precision


class Pixel(VideoPart):
    def __init__(self, array, precision):
        if len(array.shape) != 0:
            raise ValueError(f"Invalid pixel shape: {array.shape}")
        super().__init__(array, precision)

    @overload
    def __getitem__(self, index: EllipsisType) -> Self:
        ...

    @overload
    def __getitem__(self, index: tuple[()]) -> Self:
        ...

    def __getitem__(self, index):
        selection = self._array.__getitem__(index)
        rank = len(selection.shape)
        if rank == 0:
            return Pixel(selection, self.precision)
        else:
            raise IndexError(f"Invalid pixel index: {index}")

    def to_scalar(self, dtype=None):
        return self.to_array(dtype)[()]


class VideoPart1D(VideoPart):
    def __init__(self, array, precision):
        if len(array.shape) != 1:
            raise ValueError(f"Invalid 1D shape: {array.shape}")
        super().__init__(array, precision)

    @overload
    def __getitem__(self, index: EllipsisType) -> Self:
        ...

    @overload
    def __getitem__(self, index: tuple[int]) -> Pixel:
        ...

    @overload
    def __getitem__(self, index: tuple[slice]) -> Self:
        ...

    def __getitem__(self, index):
        selection = self._array.__getitem__(index)
        rank = len(selection.shape)
        if rank == 0:
            return Pixel(selection, self.precision)
        elif rank == 1:
            return VideoPart1D(selection, self.precision)
        elif rank == 2:
            return VideoPart2D(selection, self.precision)
        else:
            raise IndexError(f"Invalid 1D index: {index}")


class VideoPart2D(VideoPart):
    def __init__(self, array, precision):
        if len(array.shape) != 2:
            raise ValueError(f"Invalid 2D shape: {array.shape}")
        super().__init__(array, precision)

    @overload
    def __getitem__(self, index: EllipsisType) -> Self:
        ...

    @overload
    def __getitem__(self, index: tuple[int, int]) -> Pixel:
        ...

    @overload
    def __getitem__(self, index: tuple[int, slice]) -> VideoPart1D:
        ...

    @overload
    def __getitem__(self, index: tuple[int, slice]) -> VideoPart1D:
        ...

    @overload
    def __getitem__(self, index: tuple[slice, slice]) -> Self:
        ...

    def __getitem__(self, index):
        selection = self._array.__getitem__(index)
        rank = len(selection.shape)
        if rank == 0:
            return Pixel(selection, self.precision)
        elif rank == 1:
            return VideoPart1D(selection, self.precision)
        elif rank == 2:
            return VideoPart2D(selection, self.precision)
        else:
            raise IndexError(f"Invalid 2D index: {index}")


class Video_indexing(Video):
    @overload
    def __getitem__(self, index: EllipsisType) -> Self:
        ...

    @overload
    def __getitem__(self, index: int) -> VideoPart2D:
        ...

    @overload
    def __getitem__(self, index: slice) -> Self:
        ...

    @overload
    def __getitem__(self, index: tuple[int]) -> VideoPart2D:
        ...

    @overload
    def __getitem__(self, index: tuple[slice]) -> Self:
        ...

    @overload
    def __getitem__(self, index: tuple[int, int]) -> VideoPart1D:
        ...

    @overload
    def __getitem__(self, index: tuple[int, slice]) -> VideoPart2D:
        ...

    @overload
    def __getitem__(self, index: tuple[slice, int]) -> VideoPart2D:
        ...

    @overload
    def __getitem__(self, index: tuple[slice, slice]) -> Self:
        ...

    @overload
    def __getitem__(self, index: tuple[int, int, int]) -> Pixel:
        ...

    @overload
    def __getitem__(self, index: tuple[int, int, slice]) -> VideoPart1D:
        ...

    @overload
    def __getitem__(self, index: tuple[int, slice, int]) -> VideoPart1D:
        ...

    @overload
    def __getitem__(self, index: tuple[slice, int, int]) -> VideoPart1D:
        ...

    @overload
    def __getitem__(self, index: tuple[int, slice, slice]) -> VideoPart2D:
        ...

    @overload
    def __getitem__(self, index: tuple[slice, int, slice]) -> VideoPart2D:
        ...

    @overload
    def __getitem__(self, index: tuple[slice, slice, int]) -> VideoPart2D:
        ...

    @overload
    def __getitem__(self, index: tuple[slice, slice, slice]) -> Self:
        ...

    def __getitem__(self, index):
        """
        Return a particular part of the video.
        """
        selection = self._array.__getitem__(index)
        rank = len(selection.shape)
        if rank == 0:
            return Pixel(selection, self.precision)
        elif rank == 1:
            return VideoPart1D(selection, self.precision)
        elif rank == 2:
            return VideoPart2D(selection, self.precision)
        elif rank == 3:
            return type(self)(selection, self.precision)
        else:
            raise IndexError(f"Invalid video index: {index}")
