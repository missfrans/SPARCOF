from __future__ import annotations

import unittest

import pandas as pd

from sparcof.scoring import apply_scores


class ScoringTests(unittest.TestCase):
    def test_cross_validation_can_drive_candidate_scoring(self):
        rows = pd.DataFrame(
            {
                "dataset": ["d", "d"],
                "classifier": ["dt_cpu", "dt_cpu"],
                "accuracy": [0.99, 0.80],
                "precision": [0.99, 0.80],
                "recall": [0.99, 0.80],
                "f1": [0.99, 0.80],
                "macro_FAR": [0.01, 0.20],
                "cv_accuracy_mean": [0.70, 0.90],
                "cv_precision_mean": [0.70, 0.90],
                "cv_recall_mean": [0.70, 0.90],
                "cv_f1_mean": [0.70, 0.90],
                "cv_macro_FAR_mean": [0.30, 0.10],
                "final_training_plus_prediction_time_s": [1.0, 1.0],
                "final_inference_time_ms_per_sample_mean": [1.0, 1.0],
                "num_features": [5, 5],
            }
        )
        scenarios = {
            "equal": {
                "effectiveness": {
                    "accuracy": 0.2,
                    "precision": 0.2,
                    "recall": 0.2,
                    "f1": 0.2,
                    "inv_far": 0.2,
                },
                "efficiency": {"inv_training_time": 0.4, "inv_inference_time": 0.4, "inv_num_features": 0.2},
            }
        }
        scored = apply_scores(rows, scenarios, metric_source="cross_validation")
        self.assertGreater(scored.loc[1, "effectiveness_score__equal"], scored.loc[0, "effectiveness_score__equal"])


if __name__ == "__main__":
    unittest.main()
