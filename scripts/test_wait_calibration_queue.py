"""Queue-only calibration contracts; historical fixtures remain unchanged."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.analyze_wait_calibration import audit, QUEUE_METHOD
from scripts.analyze_wait_policy import analyze, errors, replay, ttft_scorecard
from scripts.test_wait_calibration_two_wave import TwoWaveTrace, two_wave_trace


def queue_trace(old=None, *, source="measured", cal_baseline=.2, raw_baseline=.2, raw_available=True):
    """Convert the independent bank oracle's input x into N, with a larger raw F.

    Its fitted ratios/weights still use x=N; raw F is deliberately changed only
    in saved raw fields, so a floor-contaminated audit/replay cannot pass.
    """
    log = copy.deepcopy(two_wave_trace() if old is None else old)
    for event in log:
        event["calibration_method"] = QUEUE_METHOD
        kind = event["event"]
        if kind in ("forecast", "calibration_forecast", "calibration_feedback"):
            event.update(calibration_baseline_seconds=cal_baseline if source != "unknown" else -1.,
                calibration_baseline_source=source, calibration_baseline_available=source != "unknown",
                baseline_seconds=raw_baseline if raw_available else 0., baseline_available=raw_available)
            key = "raw_excess_seconds" if kind == "calibration_feedback" else "predicted_excess_seconds"
            event["queue_excess_seconds"] = event["calibration_input_excess_seconds"] = event[key]
            event[key] = 80.
            if kind != "calibration_feedback":
                event["predicted_ttft_seconds"] = event["baseline_seconds"] + 80.
                event["would_reject"] = False
                if source == "unknown":
                    event["calibrated_ttft_seconds"] = -1.
                    event["calibrated_would_reject"] = False
    if source != "unknown":
        first = log[0]
        log.insert(0, dict(event="calibration_baseline", worker="worker", calibration_method=QUEUE_METHOD,
            calibration_epoch=0, timestamp=first["timestamp"], baseline_previous_source="unknown",
            baseline_previous_seconds=-1., calibration_baseline_source=source,
            calibration_baseline_seconds=cal_baseline, calibration_baseline_available=True,
            calibration_baseline_material_promotion=False))
    return log


def row(request_id="settled", **values):
    return dict(opc_request_id=request_id, phase="pressure", cohort="fixture", started_elapsed_seconds=35.,
                client_ttft_seconds=2.2, smg_ttft_seconds=2.2, first_text_late=False,
                complete=False, completion_late=None, intentional_stream_abort=True,
                measurement_target="ttft", observation_horizon_seconds=420, status=200, **values)


class QueueCalibrationTests(unittest.TestCase):
    def assert_valid(self, log):
        checked = audit(log)
        self.assertTrue(checked["ok"], checked["errors"])
        self.assertEqual(checked["calibration_method"], QUEUE_METHOD)
        return checked

    def test_queue_fit_preserves_banks_and_ignores_raw_floor(self):
        log = queue_trace()
        checked = self.assert_valid(log)
        self.assertEqual((checked["accepted"], checked["poll_fits"]), (8, 4))
        first_fit = next(e for e in log if e["event"] == "calibration_update")
        self.assertEqual(first_fit["target_scale_after"], 2.)
        self.assertEqual(checked["calibration_input"], "saved queue_excess_seconds N, excluding empirical floor and visibility")

    def test_mse_weights_use_queue_work_instead_of_raw_floor(self):
        fixture = TwoWaveTrace()
        for at, ratio, extra in ((20, 2, 1), (30, 4, 2), (40, 2, 1), (50, 4, 2)):
            fixture.accept(at, ratio, extra)
        fixture.poll(51)
        log = queue_trace(fixture.log)
        self.assert_valid(log)
        self.assertAlmostEqual(log[-1]["target_scale_after"], 3.6)

    def test_raw_queue_and_calibrated_predictions_remain_distinct(self):
        rows, _, _ = replay(queue_trace(), [row()])
        r = rows[0]
        self.assertEqual(r["raw"], 80.2)
        self.assertEqual(r["queue_only_raw"], 1.2)
        self.assertEqual(r[QUEUE_METHOD], 2.2)
        self.assertEqual(r[QUEUE_METHOD], r["mse_poll"])
        self.assertEqual(r["calibration_baseline_source"], "measured")

    def test_provisional_calibration_does_not_invent_raw_baseline(self):
        rows, _, _ = replay(queue_trace(source="provisional", raw_available=False), [row()])
        self.assertIsNone(rows[0]["raw"])
        self.assertEqual(rows[0]["queue_only_raw"], 1.2)
        self.assertEqual(rows[0][QUEUE_METHOD], 2.2)

    def test_unknown_calibration_baseline_stays_unknown(self):
        fixture = TwoWaveTrace()
        fixture.forecast(1, "unknown")
        log = queue_trace(fixture.log, source="unknown", raw_available=False)
        self.assert_valid(log)
        rows, _, _ = replay(log, [row("unknown")])
        self.assertIsNone(rows[0]["raw"])
        self.assertIsNone(rows[0]["queue_only_raw"])
        self.assertIsNone(rows[0][QUEUE_METHOD])

    def test_corrupt_queue_saved_fields_and_provenance_fail_closed(self):
        for field, bad in (("queue_excess_seconds", 2.), ("calibration_input_excess_seconds", 2.),
                           ("calibration_baseline_source", "invented"), ("calibration_baseline_seconds", 0.)):
            log = queue_trace()
            event = next(e for e in log if e["event"] == "calibration_forecast")
            event[field] = bad
            self.assertFalse(audit(log)["ok"], field)
        log = queue_trace()
        next(e for e in log if e["event"] == "calibration_feedback").pop("calibration_input_excess_seconds")
        self.assertFalse(audit(log)["ok"])

    def test_material_measured_promotion_requires_reset(self):
        fixture = TwoWaveTrace()
        fixture.forecast(1, "before")
        log = queue_trace(fixture.log, source="provisional", raw_available=False)
        reset = dict(event="calibration_reset", worker="worker", calibration_epoch=1,
                     reason="baseline_promoted", timestamp=log[-1]["timestamp"])
        promotion = dict(log[0], calibration_epoch=1, baseline_previous_source="provisional",
            baseline_previous_seconds=.2, calibration_baseline_source="measured",
            calibration_baseline_seconds=1., calibration_baseline_material_promotion=True)
        log += [reset, promotion]
        self.assert_valid(log)
        self.assertFalse(audit(log[:-2] + [promotion])["ok"])
        changed = copy.deepcopy(log)
        changed[-1]["calibration_baseline_material_promotion"] = False
        self.assertFalse(audit(changed)["ok"])
        changed = copy.deepcopy(log)
        changed[-2]["calibration_epoch"] = changed[-1]["calibration_epoch"] = 0
        self.assertFalse(audit(changed)["ok"])
        changed = copy.deepcopy(log)
        changed[-1]["calibration_baseline_seconds"] = .25
        changed[-1]["calibration_baseline_material_promotion"] = False
        self.assertFalse(audit(changed)["ok"])

    def test_small_measured_promotion_keeps_epoch(self):
        fixture = TwoWaveTrace()
        fixture.forecast(1, "before")
        log = queue_trace(fixture.log, source="provisional", raw_available=False)
        log.append(dict(log[0], baseline_previous_source="provisional", baseline_previous_seconds=.2,
            calibration_baseline_source="measured", calibration_baseline_seconds=.25,
            calibration_baseline_material_promotion=False))
        checked = self.assert_valid(log)
        self.assertEqual(checked["material_baseline_promotions"], 0)

    def test_ordinary_resets_clear_baseline_provenance(self):
        fixture = TwoWaveTrace()
        fixture.forecast(1, "before")
        for reason in ("queue_drained", "native_profile_changed", "worker_invalidated"):
            log = queue_trace(fixture.log)
            log.append(dict(event="calibration_reset", worker="worker", calibration_epoch=1,
                            reason=reason, timestamp=log[-1]["timestamp"]))
            after = queue_trace(fixture.log)
            for event in after:
                event["calibration_epoch"] = 1
                if "request_id" in event:
                    event["request_id"] = "after"
            self.assert_valid(log + after)
            after[0]["baseline_previous_source"] = "measured"
            after[0]["baseline_previous_seconds"] = .2
            self.assertFalse(audit(log + after)["ok"], reason)

    def test_baseline_fallback_reset_logs_captured_previous_value(self):
        fixture = TwoWaveTrace()
        fixture.forecast(1, "before")
        log = queue_trace(fixture.log)
        log += [dict(event="calibration_reset", worker="worker", calibration_epoch=1,
                     reason="baseline_fallback_changed", timestamp=log[-1]["timestamp"]),
                dict(log[0], calibration_epoch=1, baseline_previous_source="measured",
                     baseline_previous_seconds=.2, calibration_baseline_source="provisional",
                     calibration_baseline_seconds=.1)]
        self.assert_valid(log)

    def test_feedback_saved_before_promotion_uses_current_epoch_clock(self):
        fixture = TwoWaveTrace()
        fixture.accept(20)
        saved = fixture.forecast(21, "inflight")
        fixture.feedback(22, saved, 2.2)
        log = queue_trace(fixture.log, source="provisional", raw_available=False)
        delayed = log.pop()
        reset = dict(event="calibration_reset", worker="worker", calibration_epoch=1,
                     reason="baseline_promoted", timestamp=delayed["timestamp"])
        promotion = dict(log[0], calibration_epoch=1, baseline_previous_source="provisional",
            baseline_previous_seconds=.2, calibration_baseline_source="measured",
            calibration_baseline_seconds=1., calibration_baseline_material_promotion=True)
        delayed.update(current_calibration_epoch=1, calibration_time_seconds=.1,
            reason="calibration_generation_changed", sequence_before=0, sequence_after=0,
            fit_sequence_before=0, fit_sequence_after=0, sample_count_before=0, sample_count_after=0,
            scale_before=1., scale_after=1., target_scale_before=1., target_scale_after=1.,
            calibration_history_revision=0, calibration_bank_index=0)
        checked = self.assert_valid(log + [reset, promotion, delayed])
        self.assertEqual(checked["feedback_reasons"]["calibration_generation_changed"], 1)

    def test_calibrated_extra_gate_requires_native_capacity_and_freshness(self):
        fixture = TwoWaveTrace()
        fixture.forecast(1, "unready", extra=400.)
        log = queue_trace(fixture.log)
        for event in log[1:]:
            event.update(ready=True, native_ready=False, capacity_learned_ready=False,
                         calibration_state="capacity_unready", calibrated_would_reject=False)
        self.assert_valid(log)
        broken = copy.deepcopy(log)
        broken[-1]["calibrated_would_reject"] = True
        self.assertFalse(audit(broken)["ok"])
        for event in log[1:]:
            event.update(ready=False, native_ready=event["event"] == "calibration_forecast",
                         capacity_learned_ready=True, calibration_state="capacity_ready")
        self.assert_valid(log)

    def test_measured_baseline_always_wins_over_provisional(self):
        self.assertFalse(audit(queue_trace(source="provisional"))["ok"])
        self.assertFalse(audit(queue_trace(source="measured", raw_available=False))["ok"])

    def test_old_and_unknown_methods_are_not_reinterpreted(self):
        self.assertTrue(audit(two_wave_trace())["ok"])
        log = queue_trace()
        log[-1]["calibration_method"] = "future_method"
        self.assertFalse(audit(log)["ok"])
        log[-1]["calibration_method"] = "mse_poll_two_wave_v1"
        self.assertFalse(audit(log)["ok"])

    def test_primary_scorecard_uses_exact_gate_total_and_readiness_separately(self):
        r = row()
        r.update(client_ttft_seconds=300., first_text_late=True, gate_ready=False,
                 gate_native_ready=False, gate_calibration_ready=True,
                 gate_predictions={QUEUE_METHOD: 320.}, **{QUEUE_METHOD: 200.})
        pre = ttft_scorecard([r], QUEUE_METHOD, 300)
        selected = ttft_scorecard([r], QUEUE_METHOD, 300, pre_admission=False)
        self.assertEqual((pre["tp"], pre["gate_unready"], selected["fn"]), (1, 1, 1))
        self.assertEqual((pre["native_gate_unready"], pre["calibration_history_unready"]), (1, 0))

    def test_censored_late_is_classified_without_numeric_imputation(self):
        r = row()
        r.update(client_ttft_seconds=None, smg_ttft_seconds=None, timed_out=True,
                 first_text_late=True, e2e_seconds=420., intentional_stream_abort=False,
                 gate_predictions={QUEUE_METHOD: 300.})
        self.assertEqual(ttft_scorecard([r], QUEUE_METHOD, 300)["censored_known_late"], 1)
        self.assertIsNone(r["client_ttft_seconds"])
        self.assertEqual(errors([r | {QUEUE_METHOD: 300.}], QUEUE_METHOD)["n"], 0)
        r["observation_horizon_seconds"] = 10
        with self.assertRaises(ValueError):
            ttft_scorecard([r], QUEUE_METHOD, 300)

    def test_error_or_short_elapsed_cannot_certify_censored_late(self):
        r = row()
        r.update(client_ttft_seconds=None, timed_out=True, first_text_late=True,
                 e2e_seconds=420., intentional_stream_abort=False,
                 gate_predictions={QUEUE_METHOD: 300.})
        for values in (dict(error="TimeoutError"), dict(e2e_seconds=10.),
                       dict(cancelled=True), dict(intentional_stream_abort=True)):
            with self.assertRaises(ValueError):
                ttft_scorecard([r | values], QUEUE_METHOD, 300)

    def test_short_or_error_outcomes_remain_unknown(self):
        r = row()
        r.update(client_ttft_seconds=None, first_text_late=None, observation_horizon_seconds=10,
                 gate_predictions={QUEUE_METHOD: 10.}, error="connection closed")
        self.assertEqual(ttft_scorecard([r], QUEUE_METHOD, 300)["unknown"], 1)
        r.update(client_ttft_seconds=400., first_text_late=False)
        with self.assertRaises(ValueError):
            ttft_scorecard([r], QUEUE_METHOD, 300)

    def test_analysis_records_new_method_and_separate_primary_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            (source / "summary.json").write_text(json.dumps(dict(execution_status="PASS")))
            (source / "profile.json").write_text(json.dumps(dict(deadline_seconds=300)))
            (source / "paired-production.jsonl").write_text(json.dumps(row()) + "\n")
            (source / "router-events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in queue_trace()))
            result = analyze(source, Path(temporary) / "analysis")
            self.assertIn("queue_only_raw", result["errors"]["all"])
            self.assertIn(QUEUE_METHOD, result["ttft_scorecards"]["all"])
            self.assertEqual(result["scorecards"]["all"]["completed_before_deadline"], 0)
            self.assertEqual(result["ttft_scorecards"]["all"][QUEUE_METHOD]["tn"], 1)

    def test_queue_plot_has_linear_distinct_banks_and_known_selected_epochs(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scripts import analyze_wait_policy as policy
        log = queue_trace()
        rows = []
        for event in log:
            if event["event"] == "calibration_forecast":
                r = row(event["request_id"])
                r["started_elapsed_seconds"] = 100 + event["calibration_time_seconds"]
                rows.append(r)
        rows.append(row("no-selected-snapshot", calibration_epoch=999))
        replayed, diagnostic, _ = replay(log, rows)
        captured = []
        original = plt.subplots
        def capture(*args, **kwargs):
            figure, axes = original(*args, **kwargs)
            captured.append((figure, axes))
            return figure, axes
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(plt, "subplots", capture), patch.object(plt, "close"):
                policy.plot(replayed, diagnostic, [], [], Path(temporary) / "queue.png",
                            two_wave=True, live_method=QUEUE_METHOD)
            figure, axes = captured[0]
            self.assertTrue(all(ax.get_xscale() == ax.get_yscale() == "linear" for ax in figure.axes))
            previous = next(line for line in axes[4].lines if line.get_label() == "previous_sample_count")
            current = next(line for line in axes[4].lines if line.get_label() == "current_sample_count")
            total = next(line for line in axes[4].lines if line.get_label() == "Total retained (max64)")
            self.assertEqual((previous.get_linestyle(), current.get_linestyle()), ("--", "-"))
            self.assertLessEqual(max(total.get_ydata()), 64)
            plotted = [value for collection in axes[3].collections for _, value in collection.get_offsets()]
            self.assertNotIn(999, plotted)
            self.assertIn("queue-only N", figure._suptitle.get_text())
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
