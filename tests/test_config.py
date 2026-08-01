from __future__ import annotations

import unittest

from sparcof.config import normalize_config, validate_config


def _minimal_config():
    return {
        "project": {"n_splits_cv": 3, "test_size": 0.2},
        "datasets": [{"name": "sample", "path": "data.csv", "target_col": "label"}],
        "feature_selection": {},
        "model_evaluation": {},
        "scoring": {
            "scenarios": {
                "equal": {
                    "effectiveness": {"accuracy": 1.0},
                    "efficiency": {"inv_num_features": 1.0},
                }
            }
        },
    }


class ConfigTests(unittest.TestCase):
    def test_validate_minimal_config(self):
        validate_config(_minimal_config())


    def test_rejects_invalid_weight_sum(self):
        config = _minimal_config()
        config["scoring"]["scenarios"]["equal"]["effectiveness"] = {"accuracy": 0.8}
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            validate_config(config)


    def test_legacy_dataset_paths_are_normalized(self):
        config = _minimal_config()
        config["paths"] = {"unsw_dir": "data/raw/unsw"}
        config["datasets"] = [{"name": "unsw", "target_col": "label"}]
        normalized = normalize_config(config)
        self.assertEqual(normalized["datasets"][0]["path"], "data/raw/unsw")
        self.assertEqual(normalized["datasets"][0]["loader"], "unsw_nb15")


if __name__ == "__main__":
    unittest.main()
