import numpy as np
from typing import Sequence, List


def pad_zeros(data: bytes, multiple: int) -> bytes:
    pad_len = (-len(data)) % multiple
    if pad_len == 0:
        return data
    return data + b"\x00" * pad_len


def unpad_zeros(padded: bytes, *, require_multiple: int | None = None) -> bytes:
    return padded.rstrip(b"\x00")


def bytes_to_uint64_list(data: bytes, N: int) -> List[np.uint64]:
    data = pad_zeros(data, N)
    return np.frombuffer(data, dtype="<u8")


def str_to_uint64_list(s: str, N: int) -> List[np.uint64]:
    return bytes_to_uint64_list(s.encode("utf8"), N)


def uint64_list_to_bytes(values: Sequence[int]) -> bytes:
    b = np.asarray(values, dtype="<u8").tobytes(order="C")
    return unpad_zeros(b)


def uint64_list_to_str(values: Sequence[int]) -> bytes:
    return uint64_list_to_bytes(values).decode("utf8")
