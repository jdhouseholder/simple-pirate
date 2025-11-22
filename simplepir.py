from dataclasses import dataclass
from typing import List
import numpy as np

from parameters import Parameters
import configuration_solver
import lib


@dataclass
class OfflineData:
    A_key: bytes
    hint: np.ndarray


def process_database(parameters: Parameters, og_db: np.ndarray):
    db = np.empty(
        (
            parameters.db_rows,
            parameters.db_cols,
        ),
        dtype=np.uint32,
    )

    if parameters.num_db_entries_per_zp_element > 0:
        # Multiple db entries per zp element
        at = np.uint64(0)
        cur = np.uint64(0)
        coeff = np.uint64(1)
        for i, v in enumerate(og_db):
            cur += v * coeff
            coeff *= np.uint64(1 << parameters.bits_per_entry)
            if ((i + 1) % parameters.num_db_entries_per_zp_element == 0) or (
                i == len(og_db) - 1
            ):
                i = at // parameters.db_cols
                j = at % parameters.db_cols

                db[i, j] = cur

                at += 1
                cur = 0
                coeff = 1
    else:
        # Multiple zp elements per db entry
        for i, v in enumerate(og_db):
            for j in range(parameters.num_zp_elements_per_db_entry):
                k = (
                    np.uint64(i) // parameters.db_cols
                ) * parameters.num_zp_elements_per_db_entry + j
                l = np.uint64(i) % parameters.db_cols
                db[k, l] = lib.base_p(
                    p=parameters.plaintext_modulus,
                    m=v,
                    i=j,
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

        self.hint = self.db @ self.A

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
                )
            )
            last += batch_size

        return answer


@dataclass
class QueryState:
    index: int
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
        t = self.A @ secret
        query = np.empty_like(t, dtype=np.uint32)
        np.add(t, err, out=query, casting="unsafe")

        query[index % self.parameters.db_cols] += self.parameters.delta

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

        return QueryState(index=index, query=query, secret=secret), query

    def recover(self, query_state: QueryState, answer: np.ndarray):
        ratio = np.uint64(self.parameters.plaintext_modulus // 2)
        offset = np.uint64(0)
        for i in range(self.parameters.db_cols):
            q = query_state.query[i][0].astype(np.uint64)
            offset += ratio * q
        q = np.uint64(1 << self.parameters.logq)
        offset %= q
        offset = q - offset

        row = query_state.index // self.parameters.db_cols

        tmp = self.hint @ query_state.secret

        answer -= tmp

        vals = []
        for i in range(
            row * self.parameters.num_zp_elements_per_db_entry,
            (row + 1) * self.parameters.num_zp_elements_per_db_entry,
        ):
            noised = answer[i].astype(np.uint64) + offset
            denoised = lib.round(
                noised,
                delta=self.parameters.delta,
                plaintext_modulus=self.parameters.plaintext_modulus,
            )
            vals.append(denoised[0])

        answer += tmp

        return lib.reconstruct_elem(
            vals,
            query_state.index,
            self.parameters,
        )


if __name__ == "__main__":
    # TODO: There is a type error somewhere when we have very high number of
    # entries the recovery is off by 1.
    num_entries = 100_000
    db = np.random.randint(0, 256, size=(num_entries,), dtype=np.uint32)

    parameters = configuration_solver.solve_system_parameters(
        num_entries=num_entries,
        bits_per_entry=8,
        lwe_secret_dimension=1024,
        logq=32,
    )

    server = SimplePirServer(parameters, db)

    offline_data = server.get_offline_data()

    client = SimplePirClient(
        parameters,
        offline_data,
    )

    for index in range(num_entries):
        want = int(db[index] % parameters.plaintext_modulus)

        state, query = client.query(index)
        answer = server.answer([query])
        got = client.recover(state, answer[0])

        assert got == want, f"{index}: {got} != {want}"
        if index % 1000 == 0:
            print(f"Completed up to {index}")
    print("Done")
