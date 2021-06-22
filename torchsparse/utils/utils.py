from itertools import repeat
from typing import List, Tuple, Union

__all__ = ['make_ntuple']


def make_ntuple(x: Union[int, List[int], Tuple[int, ...]],
                ndim: int) -> Tuple[int, ...]:
    if isinstance(x, int):
        x = tuple(repeat(x, ndim))
    elif isinstance(x, list):
        x = tuple(x)

    assert isinstance(x, tuple) and len(x) == ndim, x
    return x
