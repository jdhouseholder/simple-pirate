import numpy as np
from typing import Sequence, List


def bytes_to_uint64_list(
    data: bytes, byteorder: str = "little", enforce_64_multiple: bool = True
) -> List[np.uint64]:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    n = len(data)
    if enforce_64_multiple and n % 64 != 0:
        raise ValueError(f"length must be a multiple of 64 bytes (got {n})")
    if n % 8 != 0:
        raise ValueError(
            f"length must be a multiple of 8 bytes to form uint64 (got {n})"
        )
    if byteorder not in ("little", "big"):
        raise ValueError("byteorder must be 'little' or 'big'")

    dt = np.dtype("<u8") if byteorder == "little" else np.dtype(">u8")
    arr = np.frombuffer(data, dtype=dt)  # zero-copy view of the buffer as 64-bit words
    return list(arr)  # list of np.uint64 scalars


def uint64_list_to_bytes(
    values: Sequence[int], byteorder: str = "little", enforce_64_multiple: bool = True
) -> bytes:
    if byteorder not in ("little", "big"):
        raise ValueError("byteorder must be 'little' or 'big'")

    dt = np.dtype("<u8") if byteorder == "little" else np.dtype(">u8")
    arr = np.asarray(values, dtype=dt)
    if enforce_64_multiple and (arr.size % 8 != 0):
        raise ValueError(
            f"number of uint64 values must be a multiple of 8 (got {arr.size}) to yield bytes multiple of 64"
        )
    return arr.tobytes(order="C")
