import pprint

import numpy as np

from simplepir import SimplePirServer, SimplePirClient
from parameters import solve_system_parameters


def random_db(entries, bits_per_entry):
    # Duplicate this logic for the demo
    db_entries_per_logical_entry = 1
    if bits_per_entry > 64:
        assert bits_per_entry % 64 == 0
        db_entries_per_logical_entry = bits_per_entry // 64
        high = 1 << 64
    else:
        high = 1 << bits_per_entry
    return np.random.randint(
        0,
        high,
        size=(entries * db_entries_per_logical_entry,),
        dtype=np.uint64,
    )


def main():
    entries = 1024
    bits_per_entry = 512

    db = random_db(entries, bits_per_entry)

    print(f"Solving for parameters for database with {entries} entries")
    parameters = solve_system_parameters(
        entries=entries,
        bits_per_entry=bits_per_entry,
    )
    print("Found parameters")
    pprint.pp(parameters)

    print("Setting up server")
    server = SimplePirServer(parameters, db)
    print("Setup server")

    offline_data = server.get_offline_data()

    client = SimplePirClient(
        parameters,
        offline_data,
    )

    print("Starting test")

    for index in range(entries):
        want = db[index * 8 : (index + 1) * 8]

        state, query = client.query(index)
        answer = server.answer([query])
        got = client.recover_large_record(state, answer[0])

        assert np.all(got == want), f"{index}: {got} != {want}"

        if index > 0 and index % 100 == 0:
            print(f"Completed up to {index}")

    print("Done")


if __name__ == "__main__":
    main()
