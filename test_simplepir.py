import pytest
import numpy as np

from parameters import solve_system_parameters
import simplepir

NUM_ENTRIES = 1000


def rand_db(num_entries, bits_per_entry):
    num_db_entries_per_logical_entry = 1
    if bits_per_entry > 64:
        assert bits_per_entry % 64 == 0
        num_db_entries_per_logical_entry = bits_per_entry // 64
        high = 1 << 64
    else:
        high = 1 << bits_per_entry
    return np.random.randint(
        0, high, size=(num_entries * num_db_entries_per_logical_entry,), dtype=np.uint64
    )


def setup_server_and_client(num_entries, bits_per_entry):
    db = rand_db(num_entries=num_entries, bits_per_entry=bits_per_entry)

    parameters = solve_system_parameters(
        num_entries=num_entries,
        bits_per_entry=bits_per_entry,
    )

    server = simplepir.SimplePirServer(parameters, db)

    offline_data = server.get_offline_data()

    client = simplepir.SimplePirClient(
        parameters,
        offline_data,
    )

    return db, parameters, server, client


def _test_small_records(num_entries, bits_per_entry):
    db, parameters, server, client = setup_server_and_client(
        num_entries, bits_per_entry
    )

    for index in range(num_entries):
        want = db[index]

        state, query = client.query(index)
        answer = server.answer([query])
        got = client.recover(state, answer[0])

        assert got == want, f"{index}: {got} != {want}"


def _test_large_records(num_entries, bits_per_entry):
    db, parameters, server, client = setup_server_and_client(
        num_entries,
        bits_per_entry,
    )

    num_db_entries_per_logical_entry = bits_per_entry // 64

    for index in range(num_entries):
        state, query = client.query(index)
        answer = server.answer([query])
        got = client.recover_large_record(state, answer[0])

        want = db[
            index * num_db_entries_per_logical_entry : (index + 1)
            * num_db_entries_per_logical_entry
        ]
        assert np.all(got == want), f"{index}: {got} != {want}"


def test_1_bit_entries():
    _test_small_records(num_entries=NUM_ENTRIES, bits_per_entry=1)


def test_8_bit_entries():
    _test_small_records(num_entries=NUM_ENTRIES, bits_per_entry=8)


def test_32_bit_entries():
    _test_small_records(num_entries=NUM_ENTRIES, bits_per_entry=32)


def test_64_bit_entries():
    _test_small_records(num_entries=NUM_ENTRIES, bits_per_entry=64)


def test_512_bit_entries():
    _test_large_records(num_entries=NUM_ENTRIES, bits_per_entry=512)
