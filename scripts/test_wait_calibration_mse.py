"""Causal replay of poll-fitted MSE calibration and historical trace compatibility."""

import copy
import unittest

from scripts.analyze_wait_calibration import audit, mse_scale
from scripts.test_wait_calibration_expiry import trace


def mse_trace():
    selected, callbacks = [], []
    for index, (x, y) in enumerate([(1.0, 1.0), (2.0, 4.0), (3.0, 6.0), (4.0, 40.0)]):
        forecast, feedback = trace(y + 0.2, accepted=True)
        forecast.update(
            request_id=str(index),
            calibration_method="mse_poll",
            calibration_fit_sequence=0,
            predicted_excess_seconds=x,
            predicted_ttft_seconds=x + 0.2,
            calibrated_excess_seconds=x,
            calibrated_ttft_seconds=x + 0.2,
        )
        feedback.update(
            request_id=str(index),
            calibration_method="mse_poll",
            raw_excess_seconds=x,
            admission_fit_sequence=0,
            fit_sequence_before=0,
            fit_sequence_after=0,
            sequence_before=index,
            sequence_after=index + 1,
            sample_count_before=index,
            sample_count_after=index + 1,
        )
        selected.append(forecast)
        callbacks.append(feedback)
    before_poll = dict(
        selected[-1],
        request_id="before-poll",
        calibration_time_seconds=40.5,
        calibration_observation_sequence=4,
        calibration_sample_count=4,
    )
    update = dict(
        event="calibration_update",
        calibration_method="mse_poll",
        worker="worker",
        calibration_epoch=0,
        current_calibration_epoch=0,
        calibration_state="capacity_ready",
        calibration_time_seconds=41.0,
        sequence_before=4,
        sequence_after=4,
        fit_sequence_before=0,
        fit_sequence_after=4,
        sample_count_before=4,
        sample_count_after=4,
        scale_before=1.0,
        scale_after=1.0,
        target_scale_before=1.0,
        target_scale_after=187.0 / 30.0,
    )
    after_poll = dict(
        before_poll,
        request_id="after-poll",
        calibration_time_seconds=42.0,
        calibration_fit_sequence=4,
        calibration_scale=1.1,
        calibration_target_scale=187.0 / 30.0,
        calibrated_excess_seconds=4.4,
        calibrated_ttft_seconds=4.6,
    )
    return selected + callbacks + [before_poll, update, after_poll]


class CalibrationMseAuditTests(unittest.TestCase):
    def test_feedback_collects_before_poll_fits_and_scale_slews(self):
        result = audit(mse_trace())
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual((result["accepted"], result["poll_fits"]), (4, 1))
        self.assertEqual(result["calibration_method"], "mse_poll")
        self.assertTrue(audit(mse_trace()[:-2])["ok"])

    def test_callback_cannot_change_target(self):
        events = mse_trace()
        events[7]["target_scale_after"] = 187.0 / 30.0
        self.assertFalse(audit(events)["ok"])

    def test_poll_must_use_mse_not_mean_ratio_or_median(self):
        for target in [3.75, 2.0]:
            events = mse_trace()
            events[-2]["target_scale_after"] = target
            self.assertFalse(audit(events)["ok"])

    def test_missing_duplicate_or_future_poll_fit_is_detected(self):
        original = mse_trace()
        missing = original[:-2] + [original[-1]]
        duplicate = original[:-1] + [original[-2], original[-1]]
        future = copy.deepcopy(original)
        future[-2]["calibration_time_seconds"] = 43.0
        for events in [missing, duplicate, future]:
            self.assertFalse(audit(events)["ok"])

    def test_ten_minute_skips_and_legacy_median_logs_remain_auditable(self):
        for old_epoch in [False, True]:
            events = trace(600.0, old_epoch=old_epoch)
            for event in events:
                event["calibration_method"] = "mse_poll"
            events[0]["calibration_fit_sequence"] = 0
            events[1].update(admission_fit_sequence=0, fit_sequence_before=0, fit_sequence_after=0)
            self.assertTrue(audit(events)["ok"])
        self.assertTrue(audit(trace(357.0, accepted=True))["ok"])

    def test_mse_formula_handles_large_and_small_inputs(self):
        for scale in [1.0, 1e200, 1e-200]:
            samples = [(ratio, x * scale) for ratio, x in [(1, 1), (2, 2), (2, 3), (10, 4)]]
            self.assertAlmostEqual(mse_scale(samples), 187.0 / 30.0)

    def test_mixed_fitting_methods_are_rejected_explicitly(self):
        old = trace(20.0, accepted=True)
        # Pre-limiter historical traces need not contain monotonic time fields.
        for event in old:
            event.pop("calibration_time_seconds")
        result = audit(old + mse_trace())
        self.assertFalse(result["ok"])
        self.assertEqual(result["errors"][0][0], "mixed_or_unknown_calibration_methods")

    def test_gate_only_trace_and_required_fit_metadata(self):
        gate = dict(mse_trace()[0], event="forecast", calibration_enabled=True)
        result = audit([gate])
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["calibration_method"], "mse_poll")
        del gate["calibration_fit_sequence"]
        self.assertFalse(audit([gate])["ok"])
        events = mse_trace()
        del events[4]["fit_sequence_after"]
        self.assertFalse(audit(events)["ok"])


if __name__ == "__main__":
    unittest.main()
