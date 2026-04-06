"""Tests for eoles_dispatch.run.scenario.extract_scenario."""

import pandas as pd
import pytest
from conftest import _make_scenario_dir

from eoles_dispatch.run.scenario import extract_scenario

AREAS = ["FR", "DE"]
EXO_AREAS = ["NL"]
THR = ["nuclear", "gas_ccgt1G"]
VRE = ["onshore", "solar"]
STO = ["lake_phs", "battery"]
TEC = ["nmd"] + VRE + THR + STO
FRR = THR + STO


@pytest.fixture
def hour_month():
    return pd.DataFrame({"hour": [0, 1, 2], "month": ["202003"] * 3})


@pytest.fixture
def scenario_csv_dir(tmp_path):
    p = tmp_path / "scenario"
    _make_scenario_dir(p, areas=AREAS, exo_areas=EXO_AREAS)
    return p


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


class TestExtractScenarioKeys:
    _EXPECTED_KEYS = {
        "thr_params",
        "rsv_req",
        "str_vOM",
        "thr",
        "vre",
        "str_tec",
        "tec",
        "frr",
        "no_frr",
        "capa",
        "maxAF",
        "yEAF",
        "capa_in",
        "stockMax",
        "links",
        "exo_EX",
        "exo_IM",
        "fuel_timeFactor",
        "fuel_areaFactor",
    }

    def test_returns_all_expected_keys(self, scenario_csv_dir, hour_month):
        result = extract_scenario(scenario_csv_dir, AREAS, EXO_AREAS, hour_month)
        assert set(result.keys()) == self._EXPECTED_KEYS

    def test_thr_list_matches_thr_specs(self, scenario_csv_dir, hour_month):
        result = extract_scenario(scenario_csv_dir, AREAS, EXO_AREAS, hour_month)
        assert result["thr"] == THR

    def test_vre_list_matches_rsv_req(self, scenario_csv_dir, hour_month):
        result = extract_scenario(scenario_csv_dir, AREAS, EXO_AREAS, hour_month)
        assert result["vre"] == VRE

    def test_tec_includes_nmd_and_all_types(self, scenario_csv_dir, hour_month):
        result = extract_scenario(scenario_csv_dir, AREAS, EXO_AREAS, hour_month)
        assert "nmd" in result["tec"]
        assert set(THR + VRE + STO).issubset(set(result["tec"]))

    def test_frr_includes_frr_thr_and_storage(self, scenario_csv_dir, hour_month):
        # Both thermal techs have frr=True in the fixture
        result = extract_scenario(scenario_csv_dir, AREAS, EXO_AREAS, hour_month)
        assert set(result["frr"]) == set(THR + STO)

    def test_no_frr_is_complement_of_frr(self, scenario_csv_dir, hour_month):
        result = extract_scenario(scenario_csv_dir, AREAS, EXO_AREAS, hour_month)
        tec_set = set(result["tec"])
        assert set(result["frr"]) | set(result["no_frr"]) == tec_set
        assert set(result["frr"]) & set(result["no_frr"]) == set()


# ---------------------------------------------------------------------------
# DataFrames
# ---------------------------------------------------------------------------


class TestExtractScenarioDataFrames:
    def test_capa_filtered_to_areas(self, scenario_csv_dir, hour_month):
        result = extract_scenario(scenario_csv_dir, AREAS, EXO_AREAS, hour_month)
        assert set(result["capa"]["area"].unique()) == set(AREAS)

    def test_capa_contains_all_technologies(self, scenario_csv_dir, hour_month):
        result = extract_scenario(scenario_csv_dir, AREAS, EXO_AREAS, hour_month)
        capa_tecs = set(result["capa"]["tec"].unique())
        expected_tecs = set(VRE + THR + STO)
        assert expected_tecs.issubset(capa_tecs)

    def test_links_excludes_self_loops(self, scenario_csv_dir, hour_month):
        result = extract_scenario(scenario_csv_dir, AREAS, EXO_AREAS, hour_month)
        links = result["links"]
        assert (links["importer"] == links["exporter"]).sum() == 0

    def test_links_filtered_to_areas(self, scenario_csv_dir, hour_month):
        result = extract_scenario(scenario_csv_dir, AREAS, EXO_AREAS, hour_month)
        links = result["links"]
        assert set(links["importer"].unique()).issubset(set(AREAS))
        assert set(links["exporter"].unique()).issubset(set(AREAS))

    def test_exo_ex_filtered_correctly(self, scenario_csv_dir, hour_month):
        result = extract_scenario(scenario_csv_dir, AREAS, EXO_AREAS, hour_month)
        exo_ex = result["exo_EX"]
        assert set(exo_ex["exporter"].unique()).issubset(set(AREAS))
        assert set(exo_ex["importer"].unique()).issubset(set(EXO_AREAS))

    def test_exo_im_filtered_correctly(self, scenario_csv_dir, hour_month):
        result = extract_scenario(scenario_csv_dir, AREAS, EXO_AREAS, hour_month)
        exo_im = result["exo_IM"]
        assert set(exo_im["importer"].unique()).issubset(set(AREAS))
        assert set(exo_im["exporter"].unique()).issubset(set(EXO_AREAS))

    def test_fuel_timefactor_month_strings_format(self, scenario_csv_dir, hour_month):
        result = extract_scenario(scenario_csv_dir, AREAS, EXO_AREAS, hour_month)
        months = result["fuel_timeFactor"]["month"].unique()
        assert all(str(m).isdigit() and len(str(m)) == 6 for m in months)

    def test_fuel_timefactor_covers_only_sim_months(self, scenario_csv_dir, hour_month):
        result = extract_scenario(scenario_csv_dir, AREAS, EXO_AREAS, hour_month)
        months = result["fuel_timeFactor"]["month"].unique().tolist()
        assert sorted(months) == ["202003"]

    def test_fuel_areafactor_filtered_to_areas(self, scenario_csv_dir, hour_month):
        result = extract_scenario(scenario_csv_dir, AREAS, EXO_AREAS, hour_month)
        assert set(result["fuel_areaFactor"]["area"].unique()).issubset(set(AREAS))

    def test_thr_params_keys_match_thr_specs_columns(self, scenario_csv_dir, hour_month):
        result = extract_scenario(scenario_csv_dir, AREAS, EXO_AREAS, hour_month)
        thr_params = result["thr_params"]
        # Must include at least the key params written in _make_scenario_dir
        for expected in ("efficiency", "frr", "thr_fuel", "fuel_price"):
            assert expected in thr_params, f"thr_params missing key: {expected}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestExtractScenarioEdgeCases:
    def test_missing_scenario_file_raises(self, tmp_path, hour_month):
        empty_dir = tmp_path / "empty_scenario"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            extract_scenario(empty_dir, AREAS, EXO_AREAS, hour_month)

    def test_no_frr_thermal(self, tmp_path, hour_month):
        """gas_ccgt1G with frr=False must be absent from frr list."""
        p = tmp_path / "scenario_nofrr"
        _make_scenario_dir(p, areas=AREAS, exo_areas=EXO_AREAS)
        # Overwrite thr_specs.csv with gas_ccgt1G frr=False
        df = pd.read_csv(p / "thr_specs.csv")
        df.loc[df["tec"] == "gas_ccgt1G", "frr"] = False
        df.to_csv(p / "thr_specs.csv", index=False)

        result = extract_scenario(p, AREAS, EXO_AREAS, hour_month)
        assert "gas_ccgt1G" not in result["frr"]
        assert "gas_ccgt1G" in result["no_frr"]

    def test_areas_subset_filters_capa(self, scenario_csv_dir, hour_month):
        """Only FR in areas → capa should contain only FR."""
        result = extract_scenario(scenario_csv_dir, ["FR"], EXO_AREAS, hour_month)
        assert list(result["capa"]["area"].unique()) == ["FR"]
