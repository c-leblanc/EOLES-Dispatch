"""Tests for eoles_dispatch.format_outputs."""

import numpy as np

from eoles_dispatch.format_outputs import _safe_gene_values, _safe_var_values

# ---------------------------------------------------------------------------
# _safe_gene_values
# ---------------------------------------------------------------------------


class TestSafeGeneValues:
    def test_safe_gene_values_existing_tec(self):
        """Keys matching the technology are extracted in order."""
        gene_vals = {
            ("FR", "nuclear", 0): 10.0,
            ("FR", "nuclear", 1): 20.0,
        }
        result = _safe_gene_values(gene_vals, "nuclear", 2)
        np.testing.assert_array_equal(result, [10.0, 20.0])

    def test_safe_gene_values_missing_tec(self):
        """Missing technology returns an array of zeros."""
        gene_vals = {("FR", "nuclear", 0): 10.0}
        result = _safe_gene_values(gene_vals, "solar", 3)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_safe_gene_values_none_replaced_by_zero(self):
        """None values in the dict are replaced by 0.0."""
        gene_vals = {
            ("FR", "wind", 0): None,
            ("FR", "wind", 1): 5.0,
        }
        result = _safe_gene_values(gene_vals, "wind", 2)
        np.testing.assert_array_equal(result, [0.0, 5.0])

    def test_safe_gene_values_sorted_order(self):
        """Keys are sorted, so ('BE',...) comes before ('DE',...)."""
        gene_vals = {
            ("DE", "nuc", 0): 30.0,
            ("BE", "nuc", 0): 10.0,
        }
        result = _safe_gene_values(gene_vals, "nuc", 2)
        np.testing.assert_array_equal(result, [10.0, 30.0])

    def test_safe_gene_values_length(self):
        """Output length matches n_rows when technology exists."""
        gene_vals = {("FR", "gas", i): float(i) for i in range(5)}
        result = _safe_gene_values(gene_vals, "gas", 5)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# _safe_var_values
# ---------------------------------------------------------------------------


class TestSafeVarValues:
    def test_safe_var_values_existing(self):
        """Keys matching the technology are extracted correctly."""
        var_vals = {
            ("FR", "battery", 0): 100.0,
            ("FR", "battery", 1): 200.0,
        }
        result = _safe_var_values(var_vals, "battery", 2)
        np.testing.assert_array_equal(result, [100.0, 200.0])

    def test_safe_var_values_missing(self):
        """Missing technology returns zeros."""
        var_vals = {("FR", "battery", 0): 100.0}
        result = _safe_var_values(var_vals, "phs", 4)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 0.0])

    def test_safe_var_values_none_to_zero(self):
        """None values are replaced by 0.0."""
        var_vals = {
            ("DE", "storage", 0): None,
            ("DE", "storage", 1): 50.0,
        }
        result = _safe_var_values(var_vals, "storage", 2)
        np.testing.assert_array_equal(result, [0.0, 50.0])
