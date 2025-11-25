import math
from dataclasses import dataclass
import numpy as np

from config_table import _config_table


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


@dataclass
class ElementConfig:
    num_zp_elements: int
    num_zp_elements_per_db_entry: int
    num_zp_elements_per_logical_entry: int
    num_db_entries_per_zp_element: int


def compute_required_zp_elements(
    num_entries, bits_per_entry, num_db_entries_per_logical_entry, modulus
):
    logp = math.log2(modulus)
    if bits_per_entry <= logp:
        assert num_db_entries_per_logical_entry == 1
        # Pack multipe db entries into one Zp element.
        entries_per_element = math.floor(logp / bits_per_entry)
        num_zp_elements = np.uint64(math.ceil(num_entries / entries_per_element))
        return ElementConfig(
            num_zp_elements=num_zp_elements,
            num_zp_elements_per_db_entry=np.uint64(1),
            num_zp_elements_per_logical_entry=np.uint64(1),
            num_db_entries_per_zp_element=np.uint64(entries_per_element),
        )
    else:
        # Split one db entry across multiple Zp elements.
        num_zp_elements_per_db_entry = int(math.ceil(bits_per_entry / logp))
        num_zp_elements = (
            num_entries
            * num_zp_elements_per_db_entry
            * num_db_entries_per_logical_entry
        )
        num_zp_elements_per_logical_entry = (
            num_zp_elements_per_db_entry * num_db_entries_per_logical_entry
        )
        return ElementConfig(
            num_zp_elements=num_zp_elements,
            num_zp_elements_per_db_entry=num_zp_elements_per_db_entry,
            num_zp_elements_per_logical_entry=num_zp_elements_per_logical_entry,
            num_db_entries_per_zp_element=np.uint64(0),
        )


def compute_database_shape(num_entries, bits_per_entry, element_config):
    rows = np.uint64(math.floor(math.sqrt(element_config.num_zp_elements)))

    rem = rows % element_config.num_zp_elements_per_logical_entry
    if rem != 0:
        rows += element_config.num_zp_elements_per_logical_entry - rem

    cols = np.uint64(math.ceil(float(element_config.num_zp_elements) / float(rows)))

    return rows, cols


def pick_parameters(lwe_secret_dimension, logq, num_samples):
    # We only have table values for lwe_secret_dimension=1024 & logq=32
    assert lwe_secret_dimension == 1024
    assert logq == 32

    for row in _config_table:
        if (
            lwe_secret_dimension == (1 << row["logn"])  # always 1024 for now
            and num_samples <= (1 << row["logm"])
            and logq == row["logq"]  # always 32 for now
        ):
            sigma = row["sigma"]
            plaintext_modulus = row["p_simple"]
            return sigma, plaintext_modulus
    raise ValueError("Unable to find match in table")


def solve_system_parameters(
    num_entries,
    bits_per_entry,
    lwe_secret_dimension=1024,  # We always use 1024. Good standard choice, 128-bit security
    logq=32,  # We always use 32.
) -> Parameters:
    num_db_entries_per_logical_entry = 1
    if bits_per_entry > 64:
        if bits_per_entry % 64 != 0:
            raise ValueError(
                f"When bits_per_entry > 64 this lib only supports entries that are multiples of 64"
            )
        num_db_entries_per_logical_entry = bits_per_entry // 64
        bits_per_entry = 64

    mod_p = 2
    while True:
        element_config = compute_required_zp_elements(
            num_entries, bits_per_entry, num_db_entries_per_logical_entry, mod_p
        )
        rows, cols = compute_database_shape(num_entries, bits_per_entry, element_config)
        sigma, plaintext_modulus = pick_parameters(lwe_secret_dimension, logq, mod_p)
        if plaintext_modulus < mod_p:
            return Parameters(
                lwe_secret_dimension=np.uint64(lwe_secret_dimension),
                num_entries=np.uint64(num_entries),
                bits_per_entry=np.uint64(bits_per_entry),
                num_db_entries_per_logical_entry=np.uint64(
                    num_db_entries_per_logical_entry
                ),
                db_rows=np.uint64(rows),
                db_cols=np.uint64(cols),
                logq=np.uint64(logq),
                plaintext_modulus=np.uint64(plaintext_modulus),
                delta=(np.uint64(1) << logq) // plaintext_modulus,
                num_db_entries_per_zp_element=np.uint64(
                    element_config.num_db_entries_per_zp_element
                ),
                num_zp_elements_per_db_entry=np.uint64(
                    element_config.num_zp_elements_per_db_entry
                ),
                communication_x=np.uint64(element_config.num_zp_elements_per_db_entry),
                compression_basis=np.uint64(10),
                compression_squishing=np.uint64(3),
                compression_columns=np.uint64(0),
            )

        mod_p += 1
