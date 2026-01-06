from typing import Optional, List
import numpy as np


def to_mixed(num: int, bases: List[int]) -> List[int]:
    assert len(bases) >= 1

    for b in bases:
        assert b >= 2

    n = len(bases)
    digits: List[int] = [0] * n

    for k in range(n - 1, -1, -1):
        b = bases[k]
        digits[k] = num % b
        num //= b

    if num != 0:
        raise ValueError

    return digits


def fr_mixed(digits: List[int], bases: List[int]) -> int:
    assert len(digits) == len(bases) and len(bases) >= 1
    num = 0

    for d, b in zip(digits, bases):
        if d < 0 or d >= b:
            raise ValueError

        num = num * b + d

    return num


def flip(
    state: np.ndarray, base: int = 2, dims: Optional[List[int]] = None
) -> np.ndarray:
    if not isinstance(state, np.ndarray):
        state = state.cpu().detach().numpy()

    total_len = len(state)

    if dims is None and isinstance(base, (list, tuple, np.ndarray)):
        dims = list(base)

    if dims is None:
        if not isinstance(base, (int, np.integer)):
            raise TypeError("base must be an int when dims is None")

        n = int(np.log(total_len) / np.log(int(base)))
        if int(base) ** n != total_len:
            raise ValueError
        dims_list: List[int] = [int(base)] * n
    else:
        dims_list = [int(b) for b in dims]
        if int(np.prod(dims_list)) != total_len:
            raise ValueError

    out = np.empty_like(state)
    dims_rev = dims_list[::-1]

    for i in range(total_len):
        digits_out = to_mixed(i, dims_rev)
        j = fr_mixed(digits_out[::-1], dims_list)
        out[i] = state[j]

    return out
