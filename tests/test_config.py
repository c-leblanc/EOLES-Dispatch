"""Tests for eoles_dispatch.config constants."""

from eoles_dispatch.config import (
    DEFAULT_AREAS,
    DEFAULT_EXO_AREAS,
    ETA_IN,
    ETA_OUT,
    TRLOSS,
)


def test_eta_in_values_between_0_and_1():
    assert (ETA_IN > 0).all()
    assert (ETA_IN <= 1).all()


def test_eta_out_values_between_0_and_1():
    assert (ETA_OUT > 0).all()
    assert (ETA_OUT <= 1).all()


def test_eta_in_out_same_index():
    assert list(ETA_IN.index) == list(ETA_OUT.index)


def test_trloss_positive_and_small():
    assert 0 < TRLOSS < 1


def test_default_areas_no_duplicates():
    assert len(DEFAULT_AREAS) == len(set(DEFAULT_AREAS))
    assert len(DEFAULT_EXO_AREAS) == len(set(DEFAULT_EXO_AREAS))
    # No overlap between modeled and exogenous areas
    assert set(DEFAULT_AREAS).isdisjoint(set(DEFAULT_EXO_AREAS))
