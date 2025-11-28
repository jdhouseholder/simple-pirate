import pprint

import numpy as np

from simplepir import SimplePirServer, SimplePirClient
from parameters import solve_system_parameters
from serde import str_to_uint64_list, uint64_list_to_str

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
        db.extend(str_to_uint64_list(s, N=bits_per_entry // 8))
    db = np.asarray(db)

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

    for i, want in enumerate(strings):
        state, query = client.query(i)
        answer = server.answer([query])
        got = client.recover_large_record(state, answer[0])
        got = uint64_list_to_str(got)
        assert got == want

    print("Done")


if __name__ == "__main__":
    main()
