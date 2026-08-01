from __future__ import annotations

import numpy as np
import pandas as pd
import unittest

from sparcof.preprocess import TabularPreprocessor


class PreprocessTests(unittest.TestCase):
    def test_transformed_feature_names_follow_matrix_order(self):
        # Raw types are intentionally interleaved: numeric, categorical, numeric.
        frame = pd.DataFrame(
            {
                "numeric_a": [1.0, 2.0, 3.0, 4.0],
                "category": ["x", "y", None, "x"],
                "numeric_b": [10.0, 11.0, np.nan, 13.0],
                "record_id": [100, 101, 102, 103],
                "label": ["normal", "attack", "normal", "attack"],
            }
        )
        preprocessor = TabularPreprocessor(
            "sample",
            "label",
            dataset_config={"identifier_columns": ["record_id"]},
        )
        matrix, labels, artifacts = preprocessor.fit_transform(frame)
        self.assertEqual(artifacts.feature_names, ["numeric_a", "numeric_b", "category"])
        self.assertEqual(matrix.shape, (4, 3))
        self.assertFalse(np.isnan(matrix).any())
        self.assertEqual(sorted(np.unique(labels).tolist()), [0, 1])


if __name__ == "__main__":
    unittest.main()
