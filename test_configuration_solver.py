import pytest

import configuration_solver

NUM_ENTRIES = 1_048_576


def test_1_bit_entry():
    bits_per_entry = 1

    parameters = configuration_solver.solve_system_parameters(
        num_entries=NUM_ENTRIES,
        bits_per_entry=bits_per_entry,
    )

    assert parameters.num_db_entries_per_zp_element == 9
    assert parameters.num_zp_elements_per_db_entry == 1


def test_u8_entry():
    bits_per_entry = 8

    parameters = configuration_solver.solve_system_parameters(
        num_entries=NUM_ENTRIES,
        bits_per_entry=bits_per_entry,
    )

    assert parameters.num_db_entries_per_zp_element == 1
    assert parameters.num_zp_elements_per_db_entry == 1


def test_large_entry():
    bits_per_entry = 16

    parameters = configuration_solver.solve_system_parameters(
        num_entries=NUM_ENTRIES,
        bits_per_entry=bits_per_entry,
    )

    assert parameters.num_db_entries_per_zp_element == 0
    assert parameters.num_zp_elements_per_db_entry == 2


def test_extra_large_entry():
    bits_per_entry = 512

    parameters = configuration_solver.solve_system_parameters(
        num_entries=NUM_ENTRIES,
        bits_per_entry=bits_per_entry,
    )

    assert parameters.num_db_entries_per_zp_element == 0
    assert parameters.num_zp_elements_per_db_entry == 52
