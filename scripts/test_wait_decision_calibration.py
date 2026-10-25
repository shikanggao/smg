import unittest

from wait_decision_calibration import fit


def samples(kind, target):
    return [
        dict(x=100.0, target=target, kind=kind, values=[i % 2, (i // 2) % 2, None])
        for i in range(32)
    ]


class CalibrationTests(unittest.TestCase):
    def test_exact_queue_has_closed_form_ridge_solution(self):
        value = fit(samples("exact", 200), early=False, censored=False)
        self.assertAlmostEqual(value["coefficients"][0], (2 + 0.1) / (1 + 0.1), places=10)

    def test_satisfied_lower_bound_does_not_pull_prediction_down(self):
        bound = fit(samples("lower_bound", 50), early=False, censored=True)
        exact = fit(samples("exact", 50), early=False, censored=True)
        self.assertEqual(bound["coefficients"], [1.0, 0.0, 0.0, 0.0])
        self.assertLess(exact["coefficients"][0], 1)

    def test_violated_lower_bound_pushes_up_and_preserves_299_fraction(self):
        lower = fit(samples("lower_bound", 299.4), early=False, censored=True)
        self.assertAlmostEqual(lower["coefficients"][0], (2.994 + 0.1) / 1.1, places=10)
        self.assertEqual(lower["lower_bound_count"], 32)

    def test_exact_arm_cannot_learn_from_censored_labels(self):
        self.assertFalse(fit(samples("lower_bound", 300), early=True, censored=False)["ready"])

    def test_missing_native_progress_does_not_become_zero_feature_support(self):
        value = fit(samples("exact", 200), early=True, censored=False)
        self.assertEqual(value["mask"], 3)
        self.assertIsNone(value["minimum"][2])

    def test_constant_features_fall_back_to_queue_control(self):
        data = samples("exact", 200)
        for row in data:
            row["values"] = [1.0, 1.0, None]
        self.assertEqual(
            fit(data, early=True, censored=False), fit(data, early=False, censored=False)
        )

    def test_effective_sample_and_count_gates(self):
        self.assertFalse(fit(samples("exact", 200)[:11], early=False, censored=False)["ready"])
        data = samples("exact", 200)
        data[0]["x"] = 1e6
        self.assertFalse(fit(data, early=False, censored=False)["ready"])

    def test_deterministic_and_coefficient_bounded(self):
        data = samples("exact", 1e6)
        value = fit(data, early=True, censored=False)
        self.assertEqual(value, fit(data, early=True, censored=False))
        self.assertTrue(0 <= value["coefficients"][0] <= 4)
        self.assertTrue(all(-2 <= v <= 2 for v in value["coefficients"][1:]))


if __name__ == "__main__":
    unittest.main()
