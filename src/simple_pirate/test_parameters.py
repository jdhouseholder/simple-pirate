import pytest

from parameters import solve_system_parameters

NUM_ENTRIES = 1_048_576


def test_1_bit_entry():
    bits_per_entry = 1

    parameters = solve_system_parameters(
        entries=NUM_ENTRIES,
        bits_per_entry=bits_per_entry,
    )

    assert parameters.db_entries_per_zp_element == 9
    assert parameters.zp_elements_per_db_entry == 1


def test_u8_entry():
    bits_per_entry = 8

    parameters = solve_system_parameters(
        entries=NUM_ENTRIES,
        bits_per_entry=bits_per_entry,
    )

    assert parameters.db_entries_per_zp_element == 1
    assert parameters.zp_elements_per_db_entry == 1


def test_large_entry():
    bits_per_entry = 16

    parameters = solve_system_parameters(
        entries=NUM_ENTRIES,
        bits_per_entry=bits_per_entry,
    )

    assert parameters.db_entries_per_zp_element == 0
    assert parameters.zp_elements_per_db_entry == 2


def test_extra_large_entry():
    bits_per_entry = 512

    parameters = solve_system_parameters(
        entries=NUM_ENTRIES,
        bits_per_entry=bits_per_entry,
    )

    assert parameters.db_entries_per_zp_element == 0
    assert parameters.zp_elements_per_db_entry == 7
    assert parameters.db_entries_per_logical_entry == 8
