import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def matmul_u32_tiled(A, B, Tk=256, Tn=256):
    m, k = A.shape
    n = B.shape[1]
    out = np.empty((m, n), dtype=np.uint32)

    for i in prange(m):
        # Reuse tmp buffer across j-tiles
        tmp = np.empty(Tn, dtype=np.uint64)
        for jj in range(0, n, Tn):
            j_max = jj + Tn
            if j_max > n:
                j_max = n
            seglen = j_max - jj
            # zero only the active segment
            tmp[:seglen] = 0

            for kk in range(0, k, Tk):
                k_max = kk + Tk
                if k_max > k:
                    k_max = k
                for t in range(kk, k_max):
                    a_it = np.uint64(A[i, t])
                    # iterate contiguous slice of B[t, :]
                    b_row = B[t, jj:j_max]
                    for s in range(seglen):
                        tmp[s] += a_it * b_row[s]

            for s in range(seglen):
                out[i, jj + s] = np.uint32(tmp[s])

    return out


@njit(parallel=True, cache=True)
def matvec_packed_fused(a, b, basis, compression):
    rows, squished_cols = a.shape
    out = np.zeros((rows, 1), dtype=np.uint64)
    mask = np.uint64((1 << basis) - 1)
    # b is shaped (squished_cols * compression, 1)
    for i in prange(rows):
        acc = np.uint64(0)
        for c in range(squished_cols):
            x = np.uint64(a[i, c])
            base = c * compression
            # extract chunks and accumulate
            for k in range(compression):
                chunk = (x >> np.uint64(k * basis)) & mask
                bk = np.uint64(b[base + k, 0])
                acc += chunk * bk
        out[i] = acc
    return out
