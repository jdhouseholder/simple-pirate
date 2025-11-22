import math
import numpy as np
from dataclasses import dataclass

from parameters import Parameters

_config_table = [
    {
        "logn": 10,
        "logm": 13,
        "logq": 32,
        "sigma": 6.400000,
        "log_p_simple": 9,
        "p_simple": 991,
        "p_double": 929,
    },
    {
        "logn": 10,
        "logm": 14,
        "logq": 32,
        "sigma": 6.400000,
        "log_p_simple": 9,
        "p_simple": 833,
        "p_double": 781,
    },
    {
        "logn": 10,
        "logm": 15,
        "logq": 32,
        "sigma": 6.400000,
        "log_p_simple": 9,
        "p_simple": 701,
        "p_double": 657,
    },
    {
        "logn": 10,
        "logm": 16,
        "logq": 32,
        "sigma": 6.400000,
        "log_p_simple": 9,
        "p_simple": 589,
        "p_double": 552,
    },
    {
        "logn": 10,
        "logm": 17,
        "logq": 32,
        "sigma": 6.400000,
        "log_p_simple": 8,
        "p_simple": 495,
        "p_double": 464,
    },
    {
        "logn": 10,
        "logm": 18,
        "logq": 32,
        "sigma": 6.400000,
        "log_p_simple": 8,
        "p_simple": 416,
        "p_double": 390,
    },
    {
        "logn": 10,
        "logm": 19,
        "logq": 32,
        "sigma": 6.400000,
        "log_p_simple": 8,
        "p_simple": 350,
        "p_double": 328,
    },
    {
        "logn": 10,
        "logm": 20,
        "logq": 32,
        "sigma": 6.400000,
        "log_p_simple": 8,
        "p_simple": 294,
        "p_double": 276,
    },
    {
        "logn": 10,
        "logm": 21,
        "logq": 32,
        "sigma": 6.400000,
        "log_p_simple": 7,
        "p_simple": 247,
        "p_double": 231,
    },
]


@dataclass
class ElementConfig:
    num_entries: int
    num_zp_elements_per_db_entry: int
    num_db_entries_per_zp_element: int


def compute_required_zp_elements(num_entries, bits_per_entry, modulus):
    logp = math.log(modulus)
    if bits_per_entry <= logp:
        logp = np.uint64(logp)
        entries_per_element = logp / bits_per_entry
        db_entries = np.uint64(math.ceil(num_entries / entries_per_element))
        return ElementConfig(
            num_entries=db_entries,
            num_zp_elements_per_db_entry=1,
            num_db_entries_per_zp_element=entries_per_element,
        )
    else:
        num_zp_elements_per_db_entry = int(math.ceil(bits_per_entry / logp))
        return ElementConfig(
            num_entries=num_entries * num_zp_elements_per_db_entry,
            num_zp_elements_per_db_entry=num_zp_elements_per_db_entry,
            num_db_entries_per_zp_element=0,
        )


def compute_database_shape(num_entries, bits_per_entry, element_config):
    rows = np.uint64(math.floor(math.sqrt(num_entries)))

    rem = rows % element_config.num_zp_elements_per_db_entry
    if rem != 0:
        rows += element_config.num_zp_elements_per_db_entry - rem

    cols = np.uint64(math.ceil(num_entries / rows))

    return rows, cols


def pick_parameters(rows, cols, lwe_secret_dimension, logq, num_samples):
    for row in _config_table:
        if (
            lwe_secret_dimension == (1 << row["logn"])  # always 1024 for now
            and num_samples <= (1 << row["logm"])
            and logq == row["logq"]  # always 32 for now
        ):
            sigma = row["sigma"]
            plaintext_modulus = row["p_simple"]
            return sigma, plaintext_modulus


def solve_system_parameters(
    num_entries,
    bits_per_entry,
    lwe_secret_dimension,
    logq,
) -> Parameters:
    mod_p = 2
    while True:
        element_config = compute_required_zp_elements(
            num_entries, bits_per_entry, mod_p
        )
        rows, cols = compute_database_shape(num_entries, bits_per_entry, element_config)
        sigma, plaintext_modulus = pick_parameters(
            rows,
            cols,
            lwe_secret_dimension,
            logq,
            mod_p,
        )
        if plaintext_modulus < mod_p:
            return Parameters(
                lwe_secret_dimension=lwe_secret_dimension,
                lwe_error_stddev=sigma,
                num_entries=num_entries,
                bits_per_entry=bits_per_entry,
                db_rows=rows,
                db_cols=cols,
                logq=logq,
                plaintext_modulus=plaintext_modulus,
                delta=(np.uint64(1) << logq) // plaintext_modulus,
                num_db_entries_per_zp_element=1,  # element_config.num_db_entries_per_zp_element,
                num_zp_elements_per_db_entry=1,  # element_config.num_zp_elements_per_db_entry,
                communication_x=element_config.num_zp_elements_per_db_entry,
                compression_basis=10,
                compression_squishing=3,
                compression_columns=0,
            )

        mod_p += 1
