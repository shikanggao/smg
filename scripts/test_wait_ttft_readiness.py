"""Current-epoch readiness/isolation regressions, including busy-start controls."""
import copy
import unittest

try:
    from scripts.wait_ttft_readiness import METHOD, Readiness
except ModuleNotFoundError:
    from wait_ttft_readiness import METHOD, Readiness


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.r = Readiness("prod-test-")
        self.worker = "http://engine:8080"
        self.base = dict(worker=self.worker, calibration_method=METHOD, calibration_epoch=1,
                         calibration_state="capacity_ready")
        self.initial = dict(self.base, event="calibration_forecast", request_id="prod-test-0",
            calibration_observation_sequence=0, calibration_fit_sequence=0, calibration_history_revision=0,
            calibration_fitted_revision=0, calibration_baseline_seconds=.25,
            calibration_baseline_source="provisional", calibration_baseline_available=True,
            calibration_sample_count=0, native_ready=True)

    def learn(self):
        self.r.observe(self.initial)
        for i in range(4):
            self.r.observe(dict(self.base, event="calibration_feedback", request_id=f"prod-test-{i}",
                                current_calibration_epoch=1, reason="accepted"))
        self.r.observe(dict(self.base, event="calibration_update", sample_count_after=4,
                            fit_sequence_after=4, calibration_fitted_revision=4))
        self.r.observe(dict(self.initial, request_id="prod-test-4", calibration_observation_sequence=4,
            calibration_fit_sequence=4, calibration_history_revision=4, calibration_fitted_revision=4,
            calibration_sample_count=4))

    def test_provisional_baseline_current_fit_and_samples_can_pass(self):
        self.learn()
        proof = self.r.snapshot(True)
        self.assertTrue(proof["ready"])
        self.assertEqual(proof["workers"][0]["baseline_source"], "provisional")

    def test_busy_start_is_not_misrepresented_as_isolated(self):
        self.learn()
        self.assertFalse(self.r.snapshot(False)["ready"])
        self.assertFalse(self.r.snapshot(False)["initial_native_idle_verified"])

    def test_full_suite_busy_start_accepts_provisional_learning_without_quiet_drain(self):
        self.r = Readiness("prod-test-", scenario="busy_start")
        self.learn()
        self.assertFalse(self.r.snapshot(False, False)["ready"])
        proof = self.r.snapshot(False, True)
        self.assertTrue(proof["ready"])
        self.assertFalse(proof["initial_native_idle_verified"])
        self.assertTrue(proof["initial_native_busy_verified"])

    def test_historical_fit_cannot_make_a_new_epoch_ready(self):
        self.learn()
        self.r.observe(dict(self.base, event="calibration_reset", calibration_epoch=2))
        self.assertFalse(self.r.snapshot(True)["ready"])
        promoted = dict(self.initial, request_id="prod-test-5", calibration_epoch=2,
                        calibration_baseline_source="measured", calibration_sample_count=4,
                        calibration_fitted_revision=4)
        self.r.observe(promoted)
        checks = self.r.snapshot(True)["workers"][0]["checks"]
        self.assertFalse(checks["actual_current_epoch_fit"])
        self.assertFalse(checks["accepted_in_current_epoch"])

    def test_sample_count_without_a_real_poll_fit_fails(self):
        self.learn()
        self.r.fits.clear()
        self.assertFalse(self.r.snapshot(True)["ready"])

    def test_old_feedback_and_preexisting_history_fail_isolation(self):
        self.learn()
        self.r.observe(dict(self.base, event="calibration_feedback", request_id="old-run-9", reason="accepted"))
        self.assertFalse(self.r.snapshot(True)["ready"])
        other = Readiness("prod-test-")
        other.observe(dict(self.initial, calibration_observation_sequence=5))
        self.assertFalse(other.snapshot(True)["workers"][0]["checks"]["empty_initial_history"])

    def test_unknown_or_nonfinite_baseline_and_unready_capacity_fail(self):
        for update in [dict(calibration_baseline_available=False), dict(calibration_baseline_source="unknown"),
                       dict(calibration_baseline_seconds=float("nan")), dict(native_ready=False)]:
            self.setUp()
            self.learn()
            event = copy.deepcopy(self.r.forecasts[self.worker])
            event.update(update)
            self.r.observe(event)
            self.assertFalse(self.r.snapshot(True)["ready"])

    def test_unknown_method_fails_closed(self):
        self.r.observe(dict(self.initial, calibration_method="unreviewed"))
        self.assertTrue(self.r.snapshot(True)["errors"])


if __name__ == "__main__":
    unittest.main()
