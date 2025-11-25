import numpy as np
import hashlib
import struct
import secrets

from gauss_sample import gauss_sample


def random_key():
    return secrets.token_bytes(32)


def rand_mat(rows: int, cols: int, logmod: int) -> np.ndarray:
    if not (0 <= logmod <= 32):
        raise ValueError("logmod must be between 0 and 32 for uint32 output.")
    n = rows * cols
    buf = secrets.token_bytes(n * 4)
    arr = np.frombuffer(buf, dtype="<u4", count=n)
    if logmod < 32:
        mask = (1 << logmod) - 1
        arr &= mask
    return arr.reshape(rows, cols)


def shake_rand_mat(rows: int, cols: int, logmod: int, key: bytes) -> np.ndarray:
    xof = hashlib.shake_256(key)
    buf = xof.digest(rows * cols * 4)
    arr = np.frombuffer(buf, dtype="<u4").reshape(rows, cols)
    mask = np.uint32((1 << logmod) - 1)
    return arr & mask


def gauss_mat(rows: int, cols: int) -> np.ndarray:
    e = np.empty((rows, cols), dtype=np.int32)
    for i in range(rows):
        for j in range(cols):
            e[i][j] = gauss_sample()
    return e


def squish(x: np.ndarray, basis: int, delta: int) -> np.ndarray:
    rows, cols = x.shape
    squished_cols = (cols + delta - 1) // delta
    padded_cols = squished_cols * delta
    x_padded = np.zeros((rows, padded_cols), dtype=np.uint64)
    x_padded[:, :cols] = x
    grouped = x_padded.reshape(rows, squished_cols, delta)
    multipliers = 1 << (np.arange(delta, dtype=np.uint64) * basis)
    squished = np.sum(grouped * multipliers, axis=2)
    return squished


def unsquish(x: np.ndarray, basis: int, delta: int, cols: int) -> np.ndarray:
    rows = x.shape[0]
    mask = (1 << basis) - 1
    shifts = np.arange(delta, dtype=np.uint64) * basis
    unpacked = (x[:, :, np.newaxis] >> shifts[np.newaxis, np.newaxis, :]) & mask
    out_full = unpacked.reshape(rows, -1)
    return out_full[:, :cols].astype(np.uint64)


def expand(x: np.ndarray, mod: int, delta: int) -> np.ndarray:
    rows, cols = x.shape
    divisors = mod ** np.arange(delta, dtype=np.uint64)
    digits = (x[:, :, np.newaxis] // divisors) % mod
    stacked_digits = digits.transpose(0, 2, 1).reshape(rows * delta, cols)
    return stacked_digits.astype(np.int64) - (mod // 2)


def contract(x: np.ndarray, mod: int, delta: int) -> np.ndarray:
    rows, cols = x.shape
    if rows % delta != 0:
        raise ValueError("x must be a multiple of delta.")

    original_rows = rows // delta
    digits = (x + (mod // 2)).astype(np.uint64)
    grouped_digits = digits.reshape(original_rows, delta, cols)
    multipliers = mod ** np.arange(delta, dtype=np.uint64)
    reconstructed = np.sum(
        grouped_digits * multipliers[np.newaxis, :, np.newaxis], axis=1
    )
    return reconstructed


def matrix_mul_vec_packed(
    a: np.ndarray, b: np.ndarray, plaintext_modulus: int, basis: int, compression: int
) -> np.ndarray:
    a_unpacked = unsquish(
        a,
        basis=basis,
        delta=compression,
        cols=a.shape[1] * compression,
    )
    output_vec = a_unpacked @ b.astype(np.uint64)
    return output_vec


def round(x, delta, plaintext_modulus):
    v = (x + (delta // 2)) // delta
    return v % plaintext_modulus


def base_p(m, i, p):
    for j in range(i):
        m = m // p
    return m % p


def reconstruct_from_base_p(vals, p):
    res = np.uint64(0)
    coeff = np.uint64(1)
    for v in vals:
        res += coeff * v
        coeff *= p
    return res


def reconstruct_elem(vals, index, parameters):
    q = np.uint64(1 << parameters.logq)
    pd2 = np.uint64(parameters.plaintext_modulus // 2)
    for i in range(len(vals)):
        tmp = (vals[i] + pd2) % q
        vals[i] = tmp % parameters.plaintext_modulus

    val = reconstruct_from_base_p(vals, parameters.plaintext_modulus)

    if parameters.num_db_entries_per_zp_element > 0:
        return base_p(
            m=val,
            i=np.uint64(index % parameters.num_db_entries_per_zp_element),
            p=np.uint64(1 << parameters.bits_per_entry),
        )
    else:
        return val
