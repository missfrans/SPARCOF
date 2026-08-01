from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sparcof.data import load_dataset


class DataTests(unittest.TestCase):
    def test_generic_loader_concatenates_and_maps_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data_dir = tmp_path / "raw"
            data_dir.mkdir()
            pd.DataFrame({"x": [1, 2], "label": ["Benign", "Bot"]}).to_csv(data_dir / "a.csv", index=False)
            pd.DataFrame({"x": [3, 4], "label": ["0", "Flood"]}).to_csv(data_dir / "b.csv", index=False)
            spec = {
                "name": "sample",
                "path": "raw",
                "target_col": "label",
                "task": "binary",
                "target_mapping": {
                    "case_insensitive": True,
                    "mapping": {"benign": "normal", "0": "normal"},
                    "default": "attack",
                },
                "benign_labels": ["normal"],
            }
            bundle = load_dataset(spec, tmp_path, seed=7)
            self.assertEqual(len(bundle.df), 4)
            self.assertEqual(bundle.df["label"].value_counts().to_dict(), {"normal": 2, "attack": 2})
            self.assertEqual(len(bundle.loader_metadata["dataset_fingerprint"]), 64)

    def test_unsw_loader_accepts_provider_column_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data_dir = tmp_path / "raw"
            data_dir.mkdir()
            for index in range(1, 5):
                pd.DataFrame([[index, index % 2]]).to_csv(
                    data_dir / f"UNSW-NB15_{index}.csv", index=False, header=False
                )
            pd.DataFrame({"Name": ["feature one", "label"]}).to_csv(
                data_dir / "UNSW-NB15_features.csv", index=False
            )
            spec = {
                "name": "unsw",
                "display_name": "UNSW-NB15",
                "path": "raw",
                "loader": "unsw_nb15",
                "target_col": "label",
                "task": "binary",
            }

            bundle = load_dataset(spec, tmp_path, seed=42)

            self.assertEqual(bundle.df.columns.tolist(), ["featureone", "label"])
            self.assertTrue(bundle.loader_metadata["column_names_file"].endswith("UNSW-NB15_features.csv"))


if __name__ == "__main__":
    unittest.main()
