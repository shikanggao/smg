import json
import tempfile
import unittest
from pathlib import Path

try:
    from scripts.wait_recovery_candidate import CURRENT_METHOD, METHOD, TRAINING, pair, scores
except ModuleNotFoundError:
    from wait_recovery_candidate import CURRENT_METHOD, METHOD, TRAINING, pair, scores


class CandidateTests(unittest.TestCase):
    def event(self, **changed):
        event = dict(
            event="recovery_forecast",
            request_id="owned",
            worker="http://engine:8080",
            diagnostic_time_seconds=100.0,
            recovery_method=METHOD,
            available=True,
            reason="eligible",
            queue_excess_seconds=100.0,
            original_queue_excess_seconds=402.0,
            native_population=196.0,
            native_waiting=190.0,
            legacy_calibration_scale=2.0,
            predicted_ttft_seconds=101.0,
            legacy_gain_ablation_ttft_seconds=201.0,
            net_population_depletion_per_second=2.0,
            window_seconds=30.0,
            window_poll_count=31,
            dispatch_debt_seconds=2.0,
            threshold_seconds=300.0,
            baseline_seconds=1.0,
            gain_training=TRAINING,
            candidate_is_calibrated=False,
            would_reject=False,
            legacy_gain_ablation_would_reject=False,
            total_would_reject=False,
            legacy_gain_ablation_total_would_reject=False,
            native_cancellation_acknowledgement="unavailable",
        )
        if changed.get("available") is False:
            event.update(
                queue_excess_seconds=402.0,
                predicted_ttft_seconds=403.0,
                legacy_gain_ablation_ttft_seconds=805.0,
                would_reject=None,
                legacy_gain_ablation_would_reject=None,
                total_would_reject=None,
                legacy_gain_ablation_total_would_reject=None,
            )
        return event | changed

    def paired(self, **changed):
        return pair(self.event(**changed), {"queue_only_raw": 400.0, "calibrated": 500.0}, 300.0)

    def test_eligible_candidate_keeps_untrained_gain_identity(self):
        result = self.paired()
        self.assertEqual(result["eligible_total_ttft_predictions"]["legacy_gain_ablation"], 201.0)
        self.assertFalse(result["candidate_is_calibrated"])
        self.assertEqual(result["gain_training"], TRAINING)

    def v2(self, **changed):
        return (
            self.event(
                recovery_method=CURRENT_METHOD,
                poll_request_watermark=100,
                total_dispatched_requests=103,
                requests_since_previous_poll=2,
                rebound_allowance_requests=1,
                rebound_retained=True,
            )
            | changed
        )

    def test_v2_preserves_gain_and_actual_net_rate_with_later_inflight_requests(self):
        event = self.v2()
        result = pair(event, {}, 300.0)
        self.assertEqual(result["method"], CURRENT_METHOD)
        self.assertEqual(result["components"]["poll_request_watermark"], 100)
        self.assertEqual(result["components"]["total_dispatched_requests"], 103)
        self.assertEqual(result["eligible_total_ttft_predictions"]["legacy_gain_ablation"], 201.0)
        self.assertEqual(result["components"]["net_population_depletion_per_second"], 2.0)
        self.assertEqual(self.paired()["method"], METHOD)
        self.assertNotIn("poll_request_watermark", self.paired()["components"])

    def test_v2_missing_history_and_first_interval_are_explicit(self):
        event = self.event(
            available=False,
            recovery_method=CURRENT_METHOD,
            reason="native_empty",
            window_poll_count=0,
            window_seconds=0.0,
            total_dispatched_requests=103,
            rebound_allowance_requests=0,
            rebound_retained=False,
        )
        result = pair(event, {}, 300.0)
        self.assertIsNone(result["components"]["poll_request_watermark"])
        event.update(
            poll_request_watermark=103,
            window_poll_count=1,
            requests_since_previous_poll=2,
            rebound_allowance_requests=1,
        )
        self.assertFalse(pair(event, {}, 300.0)["available"])

    def test_v2_arrival_diagnostics_fail_closed(self):
        changes = [
            {"total_dispatched_requests": None},
            {"total_dispatched_requests": 99},
            {"poll_request_watermark": None},
            {"poll_request_watermark": 100.0},
            {"total_dispatched_requests": True},
            {"total_dispatched_requests": 1 << 64},
            {"requests_since_previous_poll": -1},
            {"requests_since_previous_poll": 101},
            {"requests_since_previous_poll": None},
            {"rebound_allowance_requests": 2},
            {"rebound_retained": 1},
            {"rebound_allowance_requests": 0},
            {"requests_since_previous_poll": 0, "rebound_allowance_requests": 0},
        ]
        for changed in changes:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                pair(self.v2(**changed), {}, 300.0)
        event = self.v2(
            requests_since_previous_poll=0, rebound_allowance_requests=0, rebound_retained=False
        )
        self.assertEqual(pair(event, {}, 300.0)["components"]["rebound_allowance_requests"], 0)

    def test_zero_legacy_gain_is_a_valid_ablation_not_missing_or_negative(self):
        result = self.paired(
            legacy_calibration_scale=0.0,
            legacy_gain_ablation_ttft_seconds=1.0,
            legacy_gain_ablation_would_reject=False,
            legacy_gain_ablation_total_would_reject=False,
        )
        self.assertEqual(result["eligible_total_ttft_predictions"]["legacy_gain_ablation"], 1.0)
        self.assertIs(result["eligible_recorded_extra_flags"]["legacy_gain_ablation"], False)

    def test_ineligible_values_are_unknown_and_fallback_is_explicit_original_snapshot(self):
        result = self.paired(
            available=False,
            reason="rebound",
            window_seconds=0,
            window_poll_count=0,
            net_population_depletion_per_second=None,
        )
        self.assertTrue(all(v is None for v in result["eligible_total_ttft_predictions"].values()))
        self.assertEqual(
            result["explicit_fallback_total_ttft_predictions"],
            {"queue_only_raw": 400.0, "legacy_gain_ablation": 500.0},
        )

    def test_unknown_baseline_never_invents_zero_total(self):
        event = self.event(
            baseline_seconds=None,
            predicted_ttft_seconds=None,
            legacy_gain_ablation_ttft_seconds=None,
            total_would_reject=None,
            legacy_gain_ablation_total_would_reject=None,
        )
        result = pair(event, {}, 300.0)
        self.assertTrue(result["available"])
        self.assertIsNone(result["eligible_total_ttft_predictions"]["queue_only_raw"])
        event["predicted_ttft_seconds"] = 100.0
        with self.assertRaisesRegex(ValueError, "Unknown recovery baseline"):
            pair(event, {}, 300.0)

    def test_plain_queue_extra_fallback_does_not_substitute_floor_inclusive_raw(self):
        event = self.event(available=False, reason="rebound")
        result = pair(event, {}, 300.0, {"raw": False, "calibrated": False}, True)
        self.assertTrue(result["explicit_fallback_recorded_extra_flags"]["queue_only_raw"])
        result = pair(event, {}, 300.0, {"raw": True}, False)
        self.assertIsNone(result["explicit_fallback_recorded_extra_flags"]["queue_only_raw"])

    def test_unknown_method_training_ack_and_window_fail_closed(self):
        for change in [
            {"recovery_method": "future"},
            {"gain_training": "retrained"},
            {"candidate_is_calibrated": True},
            {"native_cancellation_acknowledgement": "done"},
            {"available": 1},
            {"threshold_seconds": 301},
            {"window_seconds": 29.9},
            {"window_seconds": 121},
            {"window_poll_count": 4},
            {"window_poll_count": 122},
            {"net_population_depletion_per_second": 0},
            {"queue_excess_seconds": float("nan")},
            {"predicted_ttft_seconds": 99.0},
            {"native_population": 200.0},
            {"total_would_reject": True},
            {"would_reject": True},
            {"legacy_gain_ablation_ttft_seconds": 202.0},
            {"legacy_calibration_scale": 1.0},
        ]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.paired(**change)

    def test_eligible_coverage_and_fallback_differ_and_rejects_are_unknown(self):
        rows = [
            dict(first_text_late=False, client_ttft_seconds=10.0, recovery_candidate=self.paired()),
            dict(
                first_text_late=True,
                client_ttft_seconds=None,
                recovery_candidate=self.paired(available=False, reason="rebound"),
            ),
            dict(
                first_text_late=False, controlled_rejection=True, recovery_candidate=self.paired()
            ),
            dict(first_text_late=None),
        ]
        result = scores(rows, 300.0)
        self.assertEqual((result["offered"], result["captured"], result["eligible"]), (4, 3, 2))
        candidate = result["eligible_total_ttft_predictions"]["queue_only_raw"]
        fallback = result["explicit_fallback_total_ttft_predictions"]["queue_only_raw"]
        self.assertEqual(
            (candidate["tn"], candidate["unknown"], candidate["censored_known_late"]), (1, 3, 1)
        )
        self.assertEqual((fallback["tp"], fallback["tn"], fallback["unknown"]), (1, 1, 2))
        self.assertIsNone(candidate["fnr"])
        self.assertEqual(fallback["fnr"], 0.0)


class CandidateJoinTests(unittest.TestCase):
    def fixture(self, root):
        try:
            from scripts import run_wait_production_mix as driver
        except ModuleNotFoundError:
            import run_wait_production_mix as driver
        profile = json.loads((driver.HERE / "wait-ttft-focused.json").read_text())
        row = dict(
            opc_request_id="prod-live-0",
            index=0,
            phase="pressure",
            cohort="short-chat",
            control=False,
            input_words=16384,
            max_output_tokens=32,
            complete=True,
            client_ttft_seconds=320.0,
            first_text_late=True,
            completion_late=True,
            e2e_seconds=321.0,
            first_answer_seconds=None,
            dispatch_lag_seconds=0.0,
            error=None,
            usage={},
            controlled_rejection=False,
            local_drop=False,
        )
        for name, value in {
            "profile.json": profile,
            "run-config.json": dict(run_id="live", smoke=False),
            "execution.json": dict(error=None, planned_requests=1, native_drained=True),
            "readiness.json": dict(ready=True),
        }.items():
            (root / name).write_text(json.dumps(value))
        (root / "requests.jsonl").write_text(json.dumps(row) + "\n")
        (root / "metrics.jsonl").write_text("")
        candidate = CandidateTests().event(request_id=row["opc_request_id"])
        gate = dict(
            event="forecast",
            request_id=row["opc_request_id"],
            worker=candidate["worker"],
            threshold_seconds=300.0,
            predicted_ttft_seconds=403.0,
            calibrated_ttft_seconds=805.0,
            would_reject=True,
            calibrated_would_reject=True,
            native_ready=True,
            ready=True,
            baseline_available=True,
            baseline_seconds=1.0,
            calibration_baseline_available=True,
            calibration_baseline_seconds=1.0,
            calibration_input_excess_seconds=402.0,
            calibration_scale=2.0,
            calibration_method="mse_poll_two_wave_queue_v2",
        )
        return driver, [
            gate,
            dict(gate, event="calibration_forecast"),
            candidate,
            dict(event="first_text", request_id=row["opc_request_id"], ttft_seconds=320.0),
        ]

    def test_exact_candidate_join_keeps_eligible_scores_separate_from_legacy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            driver, events = self.fixture(root)
            log = root / "capture.jsonl"
            log.write_text("".join(json.dumps(e) + "\n" for e in events))
            result = driver.analyze(root, log)
            self.assertEqual(result["execution_status"], "PASS")
            live = result["recovery_candidate"]["pressure"]
            self.assertEqual(live["eligible"], 1)
            self.assertEqual(live["eligible_total_ttft_predictions"]["queue_only_raw"]["fn"], 1)
            self.assertEqual(
                result["scores"]["pressure"]["queue_only_raw"][
                    "pre_admission_total_ttft_prediction"
                ]["tp"],
                1,
            )
            self.assertTrue(any("complete normally" in s for s in result["notes"]))

    def test_missing_candidate_or_actual_probe_invalidates_shadow_main(self):
        for changed in ("missing", "actual_probe"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                driver, events = self.fixture(root)
                if changed == "missing":
                    events = [e for e in events if e["event"] != "recovery_forecast"]
                else:
                    events.append(
                        dict(
                            event="recovery_probe_reserved",
                            request_id="prod-live-0",
                            worker="http://engine:8080",
                        )
                    )
                log = root / "capture.jsonl"
                log.write_text("".join(json.dumps(e) + "\n" for e in events))
                result = driver.analyze(root, log)
                self.assertEqual(result["execution_status"], "INVALID")
                key = (
                    "exact_recovery_forecasts"
                    if changed == "missing"
                    else "no_actual_recovery_probe_in_shadow_round"
                )
                self.assertFalse(result["execution_checks"][key])

    def test_profile_pins_fresh_v2_without_breaking_historical_v1(self):
        for use_v2 in (False, True):
            with self.subTest(use_v2=use_v2), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                driver, events = self.fixture(root)
                profile = json.loads((root / "profile.json").read_text())
                profile["expected_recovery_method"] = CURRENT_METHOD
                (root / "profile.json").write_text(json.dumps(profile))
                if use_v2:
                    events[2].update(
                        recovery_method=CURRENT_METHOD,
                        poll_request_watermark=100,
                        total_dispatched_requests=103,
                        requests_since_previous_poll=2,
                        rebound_allowance_requests=1,
                        rebound_retained=True,
                    )
                log = root / "capture.jsonl"
                log.write_text("".join(json.dumps(e) + "\n" for e in events))
                result = driver.analyze(root, log)
                self.assertEqual(result["execution_checks"]["consistent_recovery_method"], use_v2)
                self.assertEqual(result["execution_status"], "PASS" if use_v2 else "INVALID")


if __name__ == "__main__":
    unittest.main()
