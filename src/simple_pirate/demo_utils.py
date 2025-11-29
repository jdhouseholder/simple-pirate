import numpy as np


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


