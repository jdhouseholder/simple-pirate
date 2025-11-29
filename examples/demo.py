import pprint

import numpy as np

from simple_pirate import simplepir
from simple_pirate.parameters import solve_system_parameters


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
    entries = 10_000_000  # 1024
    bits_per_entry = 64 * 34
    n = bits_per_entry // 64

    db = random_db(entries, bits_per_entry)

    print(f"Solving for parameters for database with {entries} entries")
    parameters = solve_system_parameters(
        entries=entries,
        bits_per_entry=bits_per_entry,
    )
    print("Found parameters")
    pprint.pp(parameters)

    print("Setting up server")
    server = simplepir.SimplePirServer(parameters, db)
    print("Setup server")

    offline_data = server.get_offline_data()

    client = simplepir.SimplePirClient(
        parameters,
        offline_data,
    )

    print("Starting test")

    for index in range(entries):
        want = db[index * n : (index + 1) * n]

        state, query = client.query(index)
        answer = server.answer([query])
        got = client.recover_large_record(state, answer[0])

        assert np.all(got == want), f"{index}: {got} != {want}"

        if index > 0 and index % 10 == 0:
            print(f"Completed up to {index}")

    print("Done")


if __name__ == "__main__":
    main()
