import pprint

import numpy as np

from simple_pirate import simplepir
from simple_pirate.parameters import solve_system_parameters
from simple_pirate import serde

strings = [
    "wow",
    "hey",
    "cryptography",
    "yhpargotpyrc",
    "private information retreival",
    "is cool",
    "is lame",
    "does it work?",
    "im not really sure",
    "can you really learn with errors, idk",
]


def main():
    entries = len(strings)
    bits_per_entry = 512

    db = []
    for s in strings:
        db.extend(serde.str_to_uint64_list(s, N=bits_per_entry // 8))
    db = np.asarray(db)

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

    for i, want in enumerate(strings):
        state, query = client.query(i)
        answer = server.answer([query])
        got = client.recover_large_record(state, answer[0])
        got = serde.uint64_list_to_str(got)
        got = serde.unpad_zeros(got)
        assert got == want

    print("Done")


if __name__ == "__main__":
    main()
