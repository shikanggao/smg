"""Causal audit compatibility for expired and pre-expiry diagnostic traces."""

import unittest

from scripts.analyze_wait_calibration import audit


def trace(age, *, timed=True, accepted=False, old_epoch=False):
    selected = dict(
        event="calibration_forecast",
        request_id="probe",
        worker="worker",
        calibration_state="capacity_ready",
        calibration_epoch=0,
        calibration_observation_sequence=0,
        calibration_time_seconds=0.0,
        calibration_scale=1.0,
        calibration_sample_count=0,
        calibration_target_scale=1.0,
        baseline_seconds=0.2,
        baseline_available=True,
        predicted_excess_seconds=10.0,
        predicted_ttft_seconds=10.2,
        calibrated_ttft_seconds=10.2,
        calibrated_excess_seconds=10.0,
        ready=True,
        threshold_seconds=300.0,
        calibrated_would_reject=False,
        native_waiting=2.0,
        native_queue_visible=True,
        native_ready=True,
    )
    feedback = dict(
        event="calibration_feedback",
        request_id="probe",
        worker="worker",
        calibration_state="capacity_ready",
        calibration_epoch=0,
        current_calibration_epoch=int(old_epoch),
        reason="calibration_generation_changed"
        if old_epoch
        else "accepted"
        if accepted
        else "feedback_expired",
        baseline_seconds=0.2,
        baseline_available=True,
        raw_excess_seconds=10.0,
        ttft_seconds=age,
        admission_scale=1.0,
        admission_observation_sequence=0,
        admission_calibration_time_seconds=0.0,
        calibration_time_seconds=age,
        admission_native_waiting=2.0,
        admission_native_queue_visible=True,
        admission_native_capacity_ready=True,
        sequence_before=0,
        sequence_after=int(accepted),
        sample_count_before=0,
        sample_count_after=int(accepted),
        scale_before=1.0,
        scale_after=1.0,
        target_scale_before=1.0,
        target_scale_after=1.0,
    )
    if timed:
        feedback.update(feedback_age_seconds=age, calibration_max_age_seconds=600.0)
    return [selected, feedback]


class CalibrationExpiryAuditTests(unittest.TestCase):
    def test_expired_feedback_is_skipped_and_cannot_silently_train(self):
        self.assertTrue(audit(trace(600.0))["ok"])
        invalid = audit(trace(600.0, accepted=True))
        self.assertFalse(invalid["ok"])
        self.assertTrue(any(e[0] == "incorrect_feedback_reason" for e in invalid["errors"]))

    def test_old_epoch_takes_precedence_over_age(self):
        self.assertTrue(audit(trace(700.0, old_epoch=True))["ok"])

    def test_fresh_and_legacy_long_feedback_remain_auditable(self):
        self.assertTrue(audit(trace(599.9, accepted=True))["ok"])
        self.assertTrue(audit(trace(700.0, timed=False, accepted=True))["ok"])


if __name__ == "__main__":
    unittest.main()
