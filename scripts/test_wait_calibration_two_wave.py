"""Chronological two-wave calibration fixtures and corruption checks."""

import copy
import importlib
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from scripts.analyze_wait_calibration import audit

METHOD = "mse_poll_two_wave_v1"


class TwoWaveTrace:
    """Small oracle using acceptance-time banks and explicit trajectory anchors."""

    def __init__(self):
        self.log = []
        self.samples = []
        self.first_accepted = None
        self.bank = self.sequence = self.fitted_sequence = 0
        self.revision = self.fitted_revision = 0
        self.trajectory = (1.0, 1.0, 0.0)
        self.last_time = float("-inf")
        self.serial = 0

    def gain(self, at):
        anchor, target, since = self.trajectory
        budget = 0.5 * max(0.0, at - since)
        return min(anchor + budget, max(anchor - budget, target))

    def emit(self, at, **fields):
        assert at >= self.last_time, (at, self.last_time)
        self.last_time = at
        event = dict(
            worker="worker",
            calibration_method=METHOD,
            calibration_state="capacity_ready",
            calibration_epoch=0,
            calibration_time_seconds=at,
            timestamp=(datetime(2026, 10, 18, tzinfo=UTC) + timedelta(seconds=at)).isoformat(),
        )
        event.update(fields)
        self.log.append(event)
        return event

    def rotate(self, at):
        if self.first_accepted is None:
            return
        bank = int((at - self.first_accepted) // 300)
        if bank == self.bank:
            return
        assert bank > self.bank
        before_count, before_gain = len(self.samples), self.gain(at)
        self.samples = [sample for sample in self.samples if sample[0] >= bank - 1]
        if len(self.samples) != before_count:
            self.revision += 1
            if len(self.samples) < 4:
                self.trajectory = (1.0, 1.0, at)
        self.bank = bank
        for ready in (False, True):
            self.emit(
                at,
                event="calibration_rotation",
                calibration_state="capacity_ready" if ready else "capacity_unready",
                calibration_bank_index=bank,
                calibration_bank_seconds=300,
                calibration_history_revision=self.revision if ready else 0,
                calibration_previous_sample_count=len(self.samples) if ready else 0,
                calibration_current_sample_count=0,
                sample_count_before=before_count if ready else 0,
                sample_count_after=len(self.samples) if ready else 0,
                scale_before=before_gain if ready else 1.0,
                scale_after=self.gain(at) if ready else 1.0,
                target_scale_after=self.trajectory[1] if ready else 1.0,
            )

    def snapshot(self, at, request_id, extra=1.0, *, event="calibration_forecast"):
        self.rotate(at)
        gain = self.gain(at)
        return self.emit(
            at,
            event=event,
            request_id=request_id,
            calibration_enabled=True,
            calibration_bank_index=self.bank,
            calibration_bank_seconds=300,
            calibration_bank_max_samples=32,
            calibration_history_revision=self.revision,
            calibration_fitted_revision=self.fitted_revision,
            calibration_max_scale_change_per_second=0.5,
            calibration_observation_sequence=self.sequence,
            calibration_fit_sequence=self.fitted_sequence,
            calibration_scale=gain,
            calibration_target_scale=self.trajectory[1],
            calibration_sample_count=len(self.samples),
            calibration_ready=len(self.samples) >= 4,
            baseline_seconds=0.2,
            baseline_available=True,
            predicted_excess_seconds=extra,
            predicted_ttft_seconds=0.2 + extra,
            calibrated_excess_seconds=gain * extra,
            calibrated_ttft_seconds=0.2 + gain * extra,
            ready=True,
            native_ready=True,
            capacity_learned_ready=True,
            native_waiting=2.0,
            native_queue_visible=True,
            threshold_seconds=300.0,
            would_reject=extra >= 300,
            calibrated_would_reject=gain * extra >= 300,
        )

    def forecast(self, at, request_id, extra=1.0, *, gate_extra=None):
        self.snapshot(at, request_id, extra if gate_extra is None else gate_extra, event="forecast")
        return self.snapshot(at, request_id, extra)

    def feedback(self, at, forecast, ttft):
        self.rotate(at)
        before_count, before_gain = len(self.samples), self.gain(at)
        sequence_before = self.sequence
        age = at - forecast["calibration_time_seconds"]
        reason = "feedback_expired" if age >= 600 else "accepted"
        if reason == "accepted":
            if self.first_accepted is None:
                self.first_accepted = at
            ratio = max(ttft - 0.2, 0.0) / forecast["predicted_excess_seconds"]
            self.samples.append((self.bank, ratio, forecast["predicted_excess_seconds"]))
            current = [sample for sample in self.samples if sample[0] == self.bank]
            if len(current) > 32:
                self.samples.remove(current[0])
            self.sequence += 1
            self.revision += 1
        return self.emit(
            at,
            event="calibration_feedback",
            request_id=forecast["request_id"],
            current_calibration_epoch=0,
            reason=reason,
            calibration_bank_index=self.bank,
            calibration_history_revision=self.revision,
            baseline_seconds=0.2,
            baseline_available=True,
            raw_excess_seconds=forecast["predicted_excess_seconds"],
            ttft_seconds=ttft,
            admission_scale=forecast["calibration_scale"],
            admission_sample_count=forecast["calibration_sample_count"],
            admission_observation_sequence=forecast["calibration_observation_sequence"],
            admission_fit_sequence=forecast["calibration_fit_sequence"],
            admission_bank_index=forecast["calibration_bank_index"],
            admission_history_revision=forecast["calibration_history_revision"],
            admission_fitted_revision=forecast["calibration_fitted_revision"],
            admission_calibration_time_seconds=forecast["calibration_time_seconds"],
            admission_native_waiting=2.0,
            admission_native_queue_visible=True,
            admission_native_capacity_ready=True,
            sequence_before=sequence_before,
            sequence_after=self.sequence,
            fit_sequence_before=self.fitted_sequence,
            fit_sequence_after=self.fitted_sequence,
            sample_count_before=before_count,
            sample_count_after=len(self.samples),
            scale_before=before_gain,
            scale_after=self.gain(at),
            target_scale_before=self.trajectory[1],
            target_scale_after=self.trajectory[1],
            feedback_age_seconds=age,
            calibration_max_age_seconds=600.0,
        )

    def accept(self, at, ratio=2.0, extra=1.0):
        self.serial += 1
        ttft = 0.2 + ratio * extra
        forecast = self.forecast(at - ttft, "sample-" + str(self.serial), extra)
        return self.feedback(at, forecast, ttft)

    def poll(self, at):
        self.rotate(at)
        assert self.revision != self.fitted_revision, "A clean bank does not emit a fit"
        before, target_before = self.gain(at), self.trajectory[1]
        target = (
            sum(ratio * extra * extra for _, ratio, extra in self.samples)
            / sum(extra * extra for _, _, extra in self.samples)
            if len(self.samples) >= 4
            else 1.0
        )
        fitted_before = self.fitted_sequence
        self.trajectory = (before, target, at)
        self.fitted_sequence, self.fitted_revision = self.sequence, self.revision
        return self.emit(
            at,
            event="calibration_update",
            current_calibration_epoch=0,
            calibration_bank_index=self.bank,
            calibration_history_revision=self.revision,
            calibration_fitted_revision=self.fitted_revision,
            sequence_before=self.sequence,
            sequence_after=self.sequence,
            fit_sequence_before=fitted_before,
            fit_sequence_after=self.fitted_sequence,
            sample_count_before=len(self.samples),
            sample_count_after=len(self.samples),
            scale_before=before,
            scale_after=before,
            target_scale_before=target_before,
            target_scale_after=target,
        )


def two_wave_trace():
    fixture = TwoWaveTrace()
    for at in (20, 24, 28, 32):
        fixture.accept(at)
    fixture.poll(33)
    fixture.forecast(34, "slewing", gate_extra=2.0)
    fixture.forecast(35, "settled")
    for at in (320, 326, 332, 338):
        fixture.accept(at, ratio=4.0)
    fixture.poll(339)
    fixture.forecast(341, "both-banks")
    fixture.forecast(620, "removal-before-poll")
    fixture.poll(621)
    fixture.forecast(622, "removal-slewing")
    fixture.forecast(623, "removal-fitted")
    fixture.forecast(920, "empty-immediate-fallback")
    fixture.poll(921)
    fixture.snapshot(922, "gate-only", 700.0, event="forecast")
    return fixture.log


class TwoWaveAuditTests(unittest.TestCase):
    def assert_valid(self, events):
        result = audit(events)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["calibration_method"], METHOD)
        return result

    def assert_corrupt(self, events, mutate):
        changed = copy.deepcopy(events)
        mutate(changed)
        self.assertFalse(audit(changed)["ok"])

    def test_chronological_rotation_and_removal_only_fit(self):
        events = two_wave_trace()
        result = self.assert_valid(events)
        self.assertEqual(result["accepted"], 8)
        self.assertEqual(result["poll_fits"], 4)
        removal = next(
            e
            for e in events
            if e["event"] == "calibration_update" and e["calibration_time_seconds"] == 621
        )
        self.assertEqual(
            (
                removal["sequence_before"],
                removal["sequence_after"],
                removal["fit_sequence_before"],
                removal["fit_sequence_after"],
            ),
            (8, 8, 8, 8),
        )
        self.assertEqual(
            (removal["calibration_history_revision"], removal["calibration_fitted_revision"]),
            (9, 9),
        )
        self.assertEqual(removal["target_scale_after"], 4.0)

    def test_acceptance_anchor_and_inflight_feedback_cross_rotation(self):
        events = two_wave_trace()
        first = next(e for e in events if e["event"] == "calibration_rotation")
        self.assertEqual(first["calibration_time_seconds"], 320)
        feedback = next(
            e
            for e in events
            if e["event"] == "calibration_feedback" and e["calibration_time_seconds"] == 320
        )
        self.assertEqual(
            (feedback["admission_bank_index"], feedback["calibration_bank_index"]), (0, 1)
        )
        self.assertEqual(feedback["reason"], "accepted")
        self.assert_valid(events)

    def test_both_banks_independently_cap_at_32(self):
        fixture = TwoWaveTrace()
        for index in range(32):
            fixture.accept(20 + 4 * index)
        fixture.poll(145)
        for index in range(33):
            fixture.accept(320 + 4 * index)
        fixture.poll(449)
        final = fixture.forecast(450, "full-banks")
        self.assertEqual(final["calibration_sample_count"], 64)
        self.assertEqual(final["calibration_observation_sequence"], 65)
        self.assert_valid(fixture.log)
        self.assert_corrupt(fixture.log, lambda log: log[-1].update(calibration_sample_count=65))

    def test_sparse_rotation_resets_gain_immediately_before_poll(self):
        fixture = TwoWaveTrace()
        for at in (20, 24, 28, 32):
            fixture.accept(at)
        fixture.poll(33)
        for at in (320, 326, 332):
            fixture.accept(at, ratio=4.0)
        fixture.poll(333)
        final = fixture.forecast(620, "sparse-fallback")
        self.assertEqual(
            (
                final["calibration_sample_count"],
                final["calibration_scale"],
                final["calibration_target_scale"],
            ),
            (3, 1.0, 1.0),
        )
        fixture.poll(621)
        self.assert_valid(fixture.log)
        self.assert_corrupt(
            fixture.log,
            lambda log: next(
                e
                for e in log
                if e.get("request_id") == "sparse-fallback" and e["event"] == "calibration_forecast"
            ).update(calibration_scale=2.0),
        )

    def test_skipped_banks_clear_in_one_rotation_without_new_epoch(self):
        fixture = TwoWaveTrace()
        for at in (20, 24, 28, 32):
            fixture.accept(at)
        fixture.poll(33)
        final = fixture.forecast(920, "bulk-idle")
        fixture.poll(921)
        self.assertEqual(
            (
                final["calibration_bank_index"],
                final["calibration_epoch"],
                final["calibration_sample_count"],
                final["calibration_scale"],
            ),
            (3, 0, 0, 1.0),
        )
        self.assertEqual(sum(e["event"] == "calibration_rotation" for e in fixture.log), 2)
        self.assert_valid(fixture.log)

    def test_scale_slew_is_half_per_second_and_poll_uses_mse(self):
        fixture = TwoWaveTrace()
        for at, extra, ratio in [(20, 1, 1), (30, 2, 2), (40, 3, 2), (90, 4, 10)]:
            fixture.accept(at, ratio, extra)
        update = fixture.poll(91)
        final = fixture.forecast(92, "mse-slew")
        self.assertAlmostEqual(update["target_scale_after"], 187.0 / 30.0)
        self.assertEqual(final["calibration_scale"], 1.5)
        self.assert_valid(fixture.log)
        self.assert_corrupt(
            fixture.log,
            lambda log: log[-1].update(
                calibration_scale=1.6, calibrated_excess_seconds=1.6, calibrated_ttft_seconds=1.8
            ),
        )
        self.assert_corrupt(
            fixture.log,
            lambda log: next(e for e in log if e["event"] == "calibration_update").update(
                target_scale_after=2.0
            ),
        )

    def test_individual_admission_snapshot_still_expires_at_600_seconds(self):
        fixture = TwoWaveTrace()
        forecast = fixture.forecast(20, "expired")
        fixture.feedback(620, forecast, 600.0)
        self.assert_valid(fixture.log)
        self.assert_corrupt(fixture.log, lambda log: log[-1].update(reason="accepted"))

    def test_corrupt_revision_rotation_and_saved_admission_metadata_fail(self):
        events = two_wave_trace()
        for field, value in [
            ("calibration_bank_index", 9),
            ("calibration_history_revision", 99),
            ("calibration_fitted_revision", 99),
            ("calibration_max_scale_change_per_second", 0.1),
        ]:
            with self.subTest(field=field):
                self.assert_corrupt(
                    events, lambda log, field=field, value=value: log[-1].update({field: value})
                )
        self.assert_corrupt(
            events,
            lambda log: next(
                e
                for e in log
                if e["event"] == "calibration_rotation"
                and e["calibration_state"] == "capacity_ready"
            ).update(calibration_history_revision=99),
        )
        self.assert_corrupt(
            events,
            lambda log: next(e for e in log if e["event"] == "calibration_feedback").update(
                admission_fitted_revision=99
            ),
        )
        self.assert_corrupt(
            events,
            lambda log: next(e for e in log if e["event"] == "calibration_feedback").update(
                admission_sample_count=999
            ),
        )

    def test_missing_duplicate_and_future_removal_fit_fail(self):
        events = two_wave_trace()
        removal = next(
            e
            for e in events
            if e["event"] == "calibration_update" and e["calibration_time_seconds"] == 621
        )
        missing = [
            e
            for e in copy.deepcopy(events)
            if not (e["event"] == "calibration_update" and e["calibration_time_seconds"] == 621)
        ]
        self.assertFalse(audit(missing)["ok"])
        duplicate = copy.deepcopy(events)
        position = next(i for i, e in enumerate(duplicate) if e == removal)
        duplicate.insert(position, copy.deepcopy(removal))
        self.assertFalse(audit(duplicate)["ok"])
        self.assert_corrupt(
            events,
            lambda log: next(
                e
                for e in log
                if e["event"] == "calibration_update" and e["calibration_time_seconds"] == 621
            ).update(calibration_time_seconds=624),
        )

    def test_missing_two_wave_metadata_is_not_silently_legacy(self):
        for field in (
            "calibration_bank_index",
            "calibration_history_revision",
            "calibration_fitted_revision",
            "calibration_bank_seconds",
            "calibration_bank_max_samples",
        ):
            with self.subTest(field=field):
                self.assert_corrupt(two_wave_trace(), lambda log, field=field: log[0].pop(field))

    def test_new_bank_metadata_without_method_fails_closed(self):
        events = two_wave_trace()
        for event in events:
            event.pop("calibration_method")
        self.assertFalse(audit(events)["ok"])

    def test_old_generation_feedback_uses_restarted_current_epoch_clock(self):
        fixture = TwoWaveTrace()
        for at in (20, 24, 28, 32):
            fixture.accept(at)
        fixture.poll(33)
        forecast = fixture.forecast(34, "old-selected")
        fixture.log.append(
            dict(
                event="calibration_reset",
                worker="worker",
                reason="queue_drained",
                calibration_epoch=1,
                timestamp="2026-10-18T00:01:40+00:00",
            )
        )
        stale = dict(
            event="calibration_feedback",
            worker="worker",
            calibration_method=METHOD,
            request_id="old-selected",
            calibration_state="capacity_ready",
            calibration_epoch=0,
            current_calibration_epoch=1,
            calibration_time_seconds=0.5,
            timestamp="2026-10-18T00:01:40.500000+00:00",
            reason="calibration_generation_changed",
            baseline_seconds=0.2,
            baseline_available=True,
            raw_excess_seconds=1.0,
            ttft_seconds=66.5,
            calibration_bank_index=0,
            calibration_history_revision=0,
            sequence_before=0,
            sequence_after=0,
            fit_sequence_before=0,
            fit_sequence_after=0,
            sample_count_before=0,
            sample_count_after=0,
            scale_before=1.0,
            scale_after=1.0,
            target_scale_before=1.0,
            target_scale_after=1.0,
            admission_scale=forecast["calibration_scale"],
            admission_sample_count=forecast["calibration_sample_count"],
            admission_observation_sequence=forecast["calibration_observation_sequence"],
            admission_fit_sequence=forecast["calibration_fit_sequence"],
            admission_bank_index=forecast["calibration_bank_index"],
            admission_history_revision=forecast["calibration_history_revision"],
            admission_fitted_revision=forecast["calibration_fitted_revision"],
            admission_calibration_time_seconds=34.0,
            admission_native_waiting=2.0,
            admission_native_queue_visible=True,
            admission_native_capacity_ready=True,
            feedback_age_seconds=66.5,
            calibration_max_age_seconds=600.0,
        )
        fixture.log.append(stale)
        result = self.assert_valid(fixture.log)
        self.assertEqual(result["accepted"], 4)
        self.assertEqual(result["feedback_reasons"]["calibration_generation_changed"], 1)
        for field in ("calibration_history_revision", "sample_count_after"):
            with self.subTest(field=field):
                self.assert_corrupt(
                    fixture.log, lambda log, field=field: log[-1].update({field: 1})
                )

    def test_feedback_cannot_change_saved_worker_state_or_same_epoch_age(self):
        events = two_wave_trace()
        for field, value in (
            ("worker", "another-worker"),
            ("calibration_state", "capacity_unready"),
            ("feedback_age_seconds", 3.2),
        ):
            with self.subTest(field=field):
                self.assert_corrupt(
                    events,
                    lambda log, field=field, value=value: next(
                        e for e in log if e["event"] == "calibration_feedback"
                    ).update({field: value}),
                )

    def test_rotations_require_both_states_at_the_same_time(self):
        events = two_wave_trace()
        missing = copy.deepcopy(events)
        missing.remove(next(e for e in missing if e["event"] == "calibration_rotation"))
        self.assertFalse(audit(missing)["ok"])
        self.assert_corrupt(
            events,
            lambda log: next(
                e
                for e in log
                if e["event"] == "calibration_rotation"
                and e["calibration_state"] == "capacity_ready"
            ).update(calibration_time_seconds=320.5),
        )

    def test_legacy_history_expiry_generation_reset_is_rejected(self):
        events = two_wave_trace()
        self.assert_valid(events)
        events.append(
            dict(
                event="calibration_reset",
                reason="history_expired",
                worker="worker",
                calibration_method=METHOD,
            )
        )
        self.assertFalse(audit(events)["ok"])


class TwoWaveReplayTests(unittest.TestCase):
    def test_selected_and_pre_admission_forecasts_remain_distinct(self):
        policy = importlib.import_module("scripts.analyze_wait_policy")
        events = two_wave_trace()
        rows = [
            dict(opc_request_id="slewing", started_elapsed_seconds=34.0),
            dict(opc_request_id="gate-only", started_elapsed_seconds=922.0),
        ]
        replayed, _, checked = policy.replay(events, rows)
        self.assertTrue(checked["ok"], checked["errors"])
        self.assertAlmostEqual(replayed[0]["raw"], 1.2)
        self.assertAlmostEqual(replayed[0]["gate_predictions"]["raw"], 2.2)
        self.assertAlmostEqual(replayed[0]["mse_poll"], 1.7)
        self.assertAlmostEqual(replayed[0]["gate_predictions"]["mse_poll"], 3.2)
        self.assertEqual(replayed[0][METHOD], replayed[0]["mse_poll"])
        self.assertEqual(
            replayed[0]["gate_predictions"][METHOD], replayed[0]["gate_predictions"]["mse_poll"]
        )
        self.assertIsNone(replayed[1]["raw"])
        self.assertIsNone(replayed[1].get("calibration_epoch"))
        self.assertAlmostEqual(replayed[1]["gate_predictions"]["raw"], 700.2)
        self.assertEqual(
            set(replayed[0]["gate_predictions"]),
            {"raw", "median_feedback", "median_poll", "mse_feedback", "mse_poll", METHOD},
        )

    def test_two_wave_artists_are_linear_and_keep_lifecycle_gaps(self):
        try:
            import matplotlib
        except ModuleNotFoundError:
            self.skipTest("matplotlib is needed for artist assertions")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        policy = importlib.import_module("scripts.analyze_wait_policy")
        diagnostic = []
        spans = [
            ("a", 0, "ready", 0, [1, 2]),
            ("a", 0, "ready", 1, [310, 311]),
            ("a", 1, "ready", 0, [620, 621]),
            ("b", 1, "ready", 0, [720, 721]),
            ("a", 1, "unready", 0, [820, 821]),
        ]
        for worker, epoch, state, bank, times in spans:
            for at in times:
                diagnostic.append(
                    dict(
                        worker=worker,
                        epoch=epoch,
                        state=state,
                        bank_index=bank,
                        elapsed_seconds=at,
                        event="calibration_update",
                        feedback_age=None,
                        history_admission_age_p50=100 + at,
                        previous_sample_count=4 if bank else 0,
                        current_sample_count=4,
                        history_revision=4 + bank,
                        fitted_revision=4 + bank,
                        scales={method: 1.5 for method in policy.METHODS},
                        targets={method: 2.0 for method in policy.METHODS},
                    )
                )
        rows = [
            dict(phase="pressure", started_elapsed_seconds=1, calibration_epoch=0),
            dict(
                phase="pressure",
                started_elapsed_seconds=2,
                controlled_rejection=True,
                gate_predictions={"raw": 400},
                gate_ready=True,
            ),
            dict(phase="pressure", started_elapsed_seconds=3, calibration_epoch=None),
            dict(phase="pressure", started_elapsed_seconds=620, calibration_epoch=1),
        ]
        captured = []
        subplots = plt.subplots

        def capture(*args, **kwargs):
            fig, axes = subplots(*args, **kwargs)
            captured.append((fig, axes))
            return fig, axes

        with tempfile.TemporaryDirectory() as folder:
            with patch.object(plt, "subplots", capture), patch.object(plt, "close"):
                policy.plot(rows, diagnostic, [], [], Path(folder) / "two-wave.png", two_wave=True)
        fig, axes = captured[0]
        try:
            self.assertEqual(len(axes), 6)
            self.assertEqual(len(fig.axes), 7)
            self.assertTrue(all(ax.get_yscale() == "linear" for ax in fig.axes))
            self.assertTrue(all(ax.get_xscale() == "linear" for ax in fig.axes))
            self.assertEqual(
                [list(line.get_xdata()) for line in axes[1].lines],
                [times for _, _, _, _, times in spans],
            )
            self.assertEqual(axes[3].collections[0].get_offsets().tolist(), [[1, 0], [620, 1]])
            gain_spans = {tuple(line.get_xdata()) for line in axes[0].lines}
            self.assertEqual(gain_spans, {tuple(times) for _, _, _, _, times in spans})
            epoch_spans = {(1, 2, 310, 311), (620, 621), (720, 721), (820, 821)}
            for ax in (axes[4], axes[5], fig.axes[6]):
                self.assertEqual({tuple(line.get_xdata()) for line in ax.lines}, epoch_spans)
                self.assertTrue(all(line.get_drawstyle() == "steps-post" for line in ax.lines))
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
