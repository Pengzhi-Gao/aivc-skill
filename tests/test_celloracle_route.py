"""Fast CellOracle route guards; no GRN fit or model inference."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import anndata as ad


SCRIPT = Path(__file__).resolve().parents[1] / "examples/ko_oe/celloracle_run.py"
spec = importlib.util.spec_from_file_location("celloracle_run", SCRIPT)
route = importlib.util.module_from_spec(spec)
spec.loader.exec_module(route)


class CellOracleRouteContracts(unittest.TestCase):
    def test_oe_fixed_value_reports_cells_that_decrease(self):
        value, info = route.choose_oe_value([0, 1, 2, 3], None, 75)
        self.assertAlmostEqual(value, 2.25)
        self.assertEqual(info["cells_increased_at_setting"], 3)
        self.assertEqual(info["cells_decreased_at_setting"], 1)
        with self.assertRaisesRegex(ValueError, "inside"):
            route.choose_oe_value([0, 1, 2, 3], 4, None)

    def test_oe_setting_is_explicit_and_ko_cannot_take_it(self):
        required = ["--h5ad", "input.h5ad", "--base-grn", "base.parquet",
                    "--target-gene", "STAT3", "--output-dir", "run"]
        with self.assertRaisesRegex(ValueError, "explicit"):
            route.validate_arguments(route.parser().parse_args(required + ["--mode", "oe"]))
        with self.assertRaisesRegex(ValueError, "not applicable"):
            route.validate_arguments(route.parser().parse_args(
                required + ["--mode", "ko", "--oe-value", "2"]))

    def test_rejects_log_expression_as_raw_count(self):
        raw = ad.AnnData(X=np.array([[1, 0, 3, 1], [0, 2, 1, 1], [1, 1, 0, 1], [2, 1, 1, 0]], dtype=float))
        route.check_raw_counts(raw, None)
        raw.X = np.log1p(raw.X)
        with self.assertRaisesRegex(ValueError, "integer counts"):
            route.check_raw_counts(raw, None)

    def test_default_and_custom_grn_units_become_categorical(self):
        for column, labels in (("__aivc_grn_unit", ["all_cells"] * 4),
                               ("cell_type", ["A", "A", "B", "B"])):
            with self.subTest(column=column):
                data = ad.AnnData(X=np.ones((4, 4)))
                data.obs[column] = labels
                self.assertEqual(data.obs[column].dtype, object)
                route.prepare_oracle_grn_labels(data, column)
                self.assertEqual(data.obs[column].dtype.name, "category")
                self.assertEqual(set(data.obs[column].cat.categories), set(labels))


if __name__ == "__main__":
    unittest.main()
