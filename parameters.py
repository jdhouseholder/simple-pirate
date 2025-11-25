from dataclasses import dataclass
import numpy as np


@dataclass
class Parameters:
    lwe_secret_dimension: int  # N

    num_entries: int
    bits_per_entry: int

    db_rows: int  # L
    db_cols: int  # M

    logq: int
    plaintext_modulus: int

    delta: int

    num_db_entries_per_zp_element: int  # if log(p) > bits_per_entry
    num_zp_elements_per_db_entry: int  # if bits_per_entry > log(p)
    num_db_entries_per_logical_entry: int

    communication_x: int

    compression_basis: int
    compression_squishing: int
    compression_columns: int
