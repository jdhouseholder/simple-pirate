from dataclasses import dataclass
from typing import List
import numpy as np

from parameters import Parameters, solve_system_parameters
import lib


@dataclass
class OfflineData:
    A_key: bytes
    hint: np.ndarray


def process_database(parameters: Parameters, og_db: np.ndarray):
    db = np.zeros(
        (
            parameters.db_rows,
            parameters.db_cols,
        ),
        dtype=np.uint32,
    )

    if parameters.db_entries_per_zp_element > 0:
        # Multiple db entries per zp element
        at = np.uint64(0)
        cur = np.uint64(0)
        coeff = np.uint64(1)
        for v_index, v in enumerate(og_db):
            cur += v * coeff
            coeff *= np.uint64(1 << parameters.bits_per_entry)
            if ((v_index + 1) % parameters.db_entries_per_zp_element == 0) or (
                v_index == len(og_db) - 1
            ):
                db[
                    at // parameters.db_cols,
                    at % parameters.db_cols,
                ] = cur

                at += 1
                cur = np.uint64(0)
                coeff = np.uint64(1)
    elif parameters.db_entries_per_logical_entry == 1:
        # Multiple zp elements per db entry

        # Each entry needs to be split across zp elements, so we'll have to calculate the number of bits
        # per zp and then split.
        for v_index, v in enumerate(og_db):
            for zp_element_index in range(parameters.zp_elements_per_db_entry):
                i = (
                    v_index // parameters.db_cols
                ) * parameters.zp_elements_per_db_entry + zp_element_index
                j = v_index % parameters.db_cols

                db[i, j] = lib.base_p(
                    m=v,
                    i=zp_element_index,
                    p=parameters.plaintext_modulus,
                )
    else:
        for logical_index, logical_chunk in enumerate(
            lib.chunk(og_db, n=parameters.db_entries_per_logical_entry)
        ):
            for db_entry_index, db_entry in enumerate(logical_chunk):
                for zp_element_index in range(parameters.zp_elements_per_db_entry):
                    i = (
                        (logical_index // parameters.db_cols)
                        * parameters.zp_elements_per_db_entry
                        * parameters.db_entries_per_logical_entry
                        + parameters.zp_elements_per_db_entry * db_entry_index
                        + zp_element_index
                    )
                    j = logical_index % parameters.db_cols

                    db[i, j] = lib.base_p(
                        m=db_entry,
                        i=zp_element_index,
                        p=parameters.plaintext_modulus,
                    )

    return db


class SimplePirServer:
    def __init__(self, parameters: Parameters, db: np.ndarray):
        self.parameters = parameters

        self.A_key = lib.random_key()
        self.A = lib.shake_rand_mat(
            rows=self.parameters.db_cols,
            cols=self.parameters.lwe_secret_dimension,
            logmod=self.parameters.logq,
            key=self.A_key,
        )

        self.db = process_database(parameters, db)

        self.db -= parameters.plaintext_modulus // 2

        self.hint = (self.db @ self.A).astype(np.uint32)

        self.db += self.parameters.plaintext_modulus // 2

        self.db = lib.squish(
            self.db,
            basis=self.parameters.compression_basis,
            delta=self.parameters.compression_squishing,
        )

        if (
            self.parameters.plaintext_modulus > (1 << self.parameters.compression_basis)
        ) or (
            self.parameters.logq
            < self.parameters.compression_basis * self.parameters.compression_squishing
        ):
            raise ValueError("Unable to squish db.")

    def get_offline_data(self):
        return OfflineData(A_key=self.A_key, hint=self.hint)

    def answer(self, queries: List[np.ndarray]):
        # The client knows the indicies that it is querying so it can
        # send multiple disjoint queries according to this uniform split
        # policy and get batching for free.

        n_queries = len(queries)
        rows = self.db.shape[0]
        batch_size = rows // n_queries

        last = 0
        answer = []
        for batch, query in enumerate(queries):
            if batch == n_queries - 1:
                batch_size = rows - last
            answer.append(
                lib.matrix_mul_vec_packed(
                    a=self.db[last : last + batch_size],
                    b=query,
                    plaintext_modulus=self.parameters.plaintext_modulus,
                    basis=self.parameters.compression_basis,
                    compression=self.parameters.compression_squishing,
                ).astype(np.uint32)
            )
            last += batch_size

        return answer


@dataclass
class QueryState:
    index: int
    effective_index: int
    query: np.ndarray
    secret: np.ndarray


class SimplePirClient:
    def __init__(self, parameters, offline_data):
        self.parameters = parameters

        self.A = lib.shake_rand_mat(
            rows=self.parameters.db_cols,
            cols=self.parameters.lwe_secret_dimension,
            logmod=self.parameters.logq,
            key=offline_data.A_key,
        )
        self.hint = offline_data.hint

    def query(self, index: int):
        secret = lib.rand_mat(
            self.parameters.lwe_secret_dimension, 1, self.parameters.logq
        )
        err = lib.gauss_mat(self.parameters.db_cols, 1)
        ax = self.A @ secret
        query = np.empty_like(ax, dtype=np.uint32)
        np.add(ax, err, out=query, casting="unsafe")

        if self.parameters.db_entries_per_zp_element > 0:
            effective_index = index // self.parameters.db_entries_per_zp_element
        else:
            effective_index = index

        query[effective_index % self.parameters.db_cols] += self.parameters.delta

        if self.parameters.db_cols % self.parameters.compression_squishing != 0:
            padding_size = self.parameters.compression_squishing - (
                self.parameters.db_cols % self.parameters.compression_squishing
            )
            query = np.vstack(
                [
                    query,
                    np.zeros((padding_size, 1), dtype=np.uint32),
                ]
            )

        return QueryState(
            index=index, effective_index=effective_index, query=query, secret=secret
        ), query

    def recover(self, query_state: QueryState, answer: np.ndarray):
        ratio = np.uint64(self.parameters.plaintext_modulus // 2)
        off = np.uint64(0)
        for i in range(query_state.query.shape[0]):
            query_at_i = query_state.query[i][0].astype(np.uint64)
            off += ratio * query_at_i
        q = np.uint64(1 << self.parameters.logq)
        off %= q

        # offset = q - offset
        offset = np.empty_like(off, dtype=np.uint32)
        np.subtract(q, off, out=offset, casting="unsafe")

        row = query_state.effective_index // self.parameters.db_cols

        tmp = (self.hint @ query_state.secret).astype(np.uint32)

        # decryption_base = answer - tmp
        decryption_base = np.empty_like(answer, dtype=np.uint32)
        np.subtract(answer, tmp, out=decryption_base, casting="unsafe")

        pd2 = np.uint64(self.parameters.plaintext_modulus // 2)

        vals = []
        for i in range(
            row * self.parameters.zp_elements_per_db_entry,
            (row + 1) * self.parameters.zp_elements_per_db_entry,
        ):
            noised = decryption_base[i].astype(np.uint64) + offset
            denoised = lib.round(
                noised,
                delta=self.parameters.delta,
                plaintext_modulus=self.parameters.plaintext_modulus,
            )
            tmp = ((denoised + pd2) % q) % self.parameters.plaintext_modulus
            vals.append(tmp)

        elem = lib.reconstruct_elem(
            vals,
            query_state.index,
            self.parameters,
        )

        return elem[0]

    def recover_large_record(self, query_state: QueryState, answer: np.ndarray):
        ratio = np.uint64(self.parameters.plaintext_modulus // 2)
        off = np.uint64(0)
        for i in range(query_state.query.shape[0]):
            query_at_i = query_state.query[i][0].astype(np.uint64)
            off += ratio * query_at_i
        q = np.uint64(1 << self.parameters.logq)
        off %= q

        # offset = q - offset
        offset = np.empty_like(off, dtype=np.uint32)
        np.subtract(q, off, out=offset, casting="unsafe")

        logical_row = query_state.effective_index // self.parameters.db_cols

        tmp = (self.hint @ query_state.secret).astype(np.uint32)

        # decryption_base = answer - tmp
        decryption_base = np.empty_like(answer, dtype=np.uint32)
        np.subtract(answer, tmp, out=decryption_base, casting="unsafe")

        pd2 = np.uint64(self.parameters.plaintext_modulus // 2)

        db_entries = []
        for i in range(self.parameters.db_entries_per_logical_entry):
            vals = []
            for j in range(self.parameters.zp_elements_per_db_entry):
                index = (
                    logical_row
                    * self.parameters.zp_elements_per_db_entry
                    * self.parameters.db_entries_per_logical_entry
                    + i * self.parameters.zp_elements_per_db_entry
                    + j
                )
                noised = decryption_base[index].astype(np.uint64) + offset
                denoised = lib.round(
                    noised,
                    delta=self.parameters.delta,
                    plaintext_modulus=self.parameters.plaintext_modulus,
                )
                tmp = ((denoised + pd2) % q) % self.parameters.plaintext_modulus
                vals.append(tmp)

            elem = lib.reconstruct_elem(
                vals,
                query_state.index,
                self.parameters,
            )
            db_entries.append(elem[0])
        return db_entries
