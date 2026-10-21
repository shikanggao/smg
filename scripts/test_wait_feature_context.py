"""Check the telemetry boundary and keep rejection mechanics separate from truth."""
import copy
import json
import unittest

try:
    from . import wait_feature_context as f
except ImportError:
    import wait_feature_context as f


def fixture():
    value = dict(method=f.METHOD,
        context=dict(values=[.5, 0, 0], bucket_second=10, activity_origin_time_seconds=-.001,
            counts=dict(first_text=16, upstream_timeout=0,
            pretext_downstream_disconnect=0, unknown_pretext_end=0, admitted=16, actual_rejected=0,
            untracked_admitted=0)),
        fit=dict(beta=[-.2, 0, 0], mean=[.5, 0, 0], minimum=[0, 0, 0], maximum=[1, 0, 0],
                 support=[True, False, False], count=16, ess=12, revision=16),
        prediction=dict(gain=2, target_gain=2, contextual=True, reason="context_supported",
                        active=[True, False, False], trajectory_anchor_gain=1,
                        trajectory_anchor_time_seconds=10),
        original_queue_excess_seconds=100, baseline_seconds=1, legacy_gain=2,
        legacy_target_gain=2, total_ttft_seconds=201, extra_seconds=200,
        extra_would_reject=False, total_would_reject=False, native_ready=True, fresh=True,
        supported_path=True, prediction_available=True, threshold_seconds=300,
        ridge=.1, max_delta=.5, max_abs_coefficient=2)
    gate = dict(request_id="owned-1", worker="http://worker", calibration_input_excess_seconds=100,
                calibration_scale=2, calibration_baseline_seconds=1,
                calibration_baseline_available=True, threshold_seconds=300)
    return value, gate


def event(value, gate=None):
    gate = gate or fixture()[1]
    return dict(event="feature_context_forecast", feature_method=f.METHOD,
                request_id=gate["request_id"], worker=gate["worker"],
                diagnostic_time_seconds=12, calibration_epoch=0, feature_json=json.dumps(value))


class FeatureForecastTests(unittest.TestCase):
    def test_exact_original_queue_snapshot_and_negative_slope_are_retained(self):
        value, gate = fixture()
        result = f.forecast(event(value), gate=gate)
        self.assertTrue(result["contextual"])
        self.assertEqual(result["snapshot"]["fit"]["beta"], [-.2, 0, 0])
        self.assertEqual(result["total_ttft_seconds"], 201)

    def test_low_mechanics_extra_threshold_is_not_the_300s_truth_threshold(self):
        value, gate = fixture()
        value.update(threshold_seconds=5, extra_would_reject=True, total_would_reject=True)
        gate["threshold_seconds"] = 5
        result = f.forecast(event(value), gate=gate)
        self.assertTrue(result["recorded_extra_would_reject"])
        self.assertFalse(result["total_ttft_over_deadline"])

    def test_zero_gain_and_unknown_baseline_remain_valid_without_invented_total(self):
        value, gate = fixture()
        value["prediction"].update(gain=0, target_gain=0)
        value.update(extra_seconds=0, total_ttft_seconds=None, baseline_seconds=None,
                     extra_would_reject=None, total_would_reject=None, prediction_available=False)
        gate["calibration_baseline_available"] = False
        result = f.forecast(event(value), gate=gate)
        self.assertIsNone(result["total_ttft_seconds"])
        self.assertIsNone(result["total_ttft_over_deadline"])

    def test_missing_features_are_not_zero_imputed(self):
        value, _ = fixture()
        value["context"]["counts"].update(first_text=15, admitted=15)
        value["context"]["values"][1:] = [None, None]
        result = f.forecast(event(value))
        self.assertEqual(result["feature_values"], [.5, None, None])

    def test_known_rates_cannot_be_silently_missing(self):
        for dimension in (1, 2):
            value, _ = fixture()
            value["context"]["values"][dimension] = None
            with self.assertRaisesRegex(ValueError, "availability mismatch"):
                f.forecast(event(value))

    def test_gain_must_follow_the_recorded_half_unit_trajectory(self):
        value, _ = fixture()
        value["prediction"]["gain"] = 1.25
        value.update(extra_seconds=125, total_ttft_seconds=126)
        captured = event(value); captured["diagnostic_time_seconds"] = 10.5
        self.assertEqual(f.forecast(captured)["extra_seconds"], 125)
        value["prediction"]["gain"] = 99
        value.update(extra_seconds=9900, total_ttft_seconds=9901,
                     extra_would_reject=True, total_would_reject=True)
        captured = event(value); captured["diagnostic_time_seconds"] = 10
        with self.assertRaisesRegex(ValueError, "trajectory gain"):
            f.forecast(captured)

    def test_signed_anchor_time_is_valid_but_a_future_anchor_is_not(self):
        value, _ = fixture()
        value["prediction"]["trajectory_anchor_time_seconds"] = -.001
        captured = event(value); captured["diagnostic_time_seconds"] = 2
        self.assertEqual(f.forecast(captured)["total_ttft_seconds"], 201)
        value["prediction"]["trajectory_anchor_time_seconds"] = 3
        with self.assertRaisesRegex(ValueError, "follows admission"):
            f.forecast(event(value, None) | {"diagnostic_time_seconds": 2})

    def test_actual_reject_rate_uses_admissions_plus_actual_rejections(self):
        value, _ = fixture()
        value["context"]["counts"].update(admitted=12, actual_rejected=4)
        value["context"]["values"][2] = .25
        self.assertEqual(f.forecast(event(value))["feature_values"][2], .25)
        value["context"]["values"][2] = 0
        with self.assertRaisesRegex(ValueError, "rejection rate"):
            f.forecast(event(value))

    def test_unknown_terminal_fraction_blocks_cutoff_rate(self):
        value, _ = fixture()
        value["context"]["counts"]["unknown_pretext_end"] = 2
        with self.assertRaisesRegex(ValueError, "outcome coverage"):
            f.forecast(event(value))

    def test_no_context_fallback_must_equal_original_scalar(self):
        value, _ = fixture()
        value["prediction"].update(contextual=False, reason="legacy_fallback_no_current_support",
                                   trajectory_anchor_gain=None, trajectory_anchor_time_seconds=None)
        self.assertFalse(f.forecast(event(value))["contextual"])
        value["prediction"]["gain"] = 1
        with self.assertRaisesRegex(ValueError, "fallback gain"):
            f.forecast(event(value))

    def test_stale_snapshot_flags_are_unknown_not_false(self):
        value, _ = fixture()
        value.update(fresh=False, extra_would_reject=None, total_would_reject=None,
                     prediction_available=False)
        value["prediction"].update(contextual=False, reason="snapshot_unavailable",
                                   trajectory_anchor_gain=None, trajectory_anchor_time_seconds=None)
        self.assertIsNone(f.forecast(event(value))["recorded_extra_would_reject"])
        value["extra_would_reject"] = False
        with self.assertRaisesRegex(ValueError, "decision mismatch"):
            f.forecast(event(value))

    def test_finite_components_with_overflow_have_unknown_predictions(self):
        value, _ = fixture()
        value["prediction"].update(gain=1e200, target_gain=1e200, trajectory_anchor_gain=1e200)
        value.update(original_queue_excess_seconds=1e200, extra_seconds=None,
                     total_ttft_seconds=None, extra_would_reject=None,
                     total_would_reject=None, prediction_available=False)
        result = f.forecast(event(value))
        self.assertIsNone(result["total_ttft_seconds"])
        self.assertFalse(result["prediction_available"])

    def test_reject_changed_original_input_or_worker(self):
        value, gate = fixture()
        for key, replacement in [("calibration_input_excess_seconds", 101), ("calibration_scale", 3),
                                 ("calibration_baseline_seconds", 2), ("worker", "other")]:
            changed = dict(gate, **{key: replacement})
            with self.assertRaises(ValueError):
                f.forecast(event(value), gate=changed)

    def test_unknown_method_nonfinite_and_bool_numbers_fail_closed(self):
        value, _ = fixture()
        wrong = event(value); wrong["feature_method"] = "future_method"
        with self.assertRaises(ValueError):
            f.forecast(wrong)
        for replacement in [True, float("nan"), 2.01]:
            changed = copy.deepcopy(value); changed["fit"]["beta"][0] = replacement
            with self.assertRaises(ValueError):
                f.forecast(event(changed))


class DeadlineContextTests(unittest.TestCase):
    def value(self):
        value, gate = fixture()
        value["method"] = f.CURRENT_METHOD
        value["context"].update(deadline_untracked_pending=0)
        value["context"]["counts"].update(deadline_timely_first_text=12, deadline_breach=4,
            deadline_or_cutoff_adverse=4, deadline_or_cutoff_unknown=0)
        value["context"]["values"][1] = .25
        return value, gate

    def test_deadline_union_feature_is_distinct_from_actual_terminal_rate(self):
        value, gate = self.value()
        captured = event(value, gate); captured["feature_method"] = f.CURRENT_METHOD
        result = f.forecast(captured, gate=gate)
        self.assertEqual(result["feature_values"][1], .25)
        self.assertEqual(value["context"]["counts"]["upstream_timeout"], 0)
        self.assertEqual(value["context"]["counts"]["pretext_downstream_disconnect"], 0)

    def test_pending_untracked_or_unknown_outcomes_suppress_union_rate(self):
        for field in ("pending", "unknown", "untracked"):
            value, _ = self.value()
            if field == "pending": value["context"]["deadline_untracked_pending"] = 1
            elif field == "unknown": value["context"]["counts"]["deadline_or_cutoff_unknown"] = 2
            else: value["context"]["counts"]["untracked_admitted"] = 1
            with self.assertRaisesRegex(ValueError, "availability mismatch"):
                f.validate_context(value["context"], f.CURRENT_METHOD)
            value["context"]["values"][1] = None
            self.assertIsNone(f.validate_context(value["context"], f.CURRENT_METHOD)[0][1])

    def test_new_method_requires_exact_counter_schema(self):
        value, _ = self.value()
        for name in f.DEADLINE_COUNTS:
            changed = copy.deepcopy(value["context"]); del changed["counts"][name]
            with self.assertRaisesRegex(ValueError, "counter schema"):
                f.validate_context(changed, f.CURRENT_METHOD)
        with self.assertRaisesRegex(ValueError, "counter schema"):
            f.validate_context(value["context"], f.METHOD)

    def test_fresh_capture_cannot_mix_historical_methods(self):
        current = dict(event="feature_deadline_started", feature_method=f.CURRENT_METHOD)
        self.assertTrue(f.audit_method([current], f.CURRENT_METHOD)["ok"])
        legacy = dict(event="feature_context_feedback", feature_method=f.METHOD)
        result = f.audit_method([current, legacy], f.CURRENT_METHOD)
        self.assertFalse(result["ok"])
        self.assertEqual(result["errors"][0]["captured_method"], f.METHOD)
        self.assertFalse(f.audit_method([], f.CURRENT_METHOD)["ok"])


def activity_event(rid, category, counts, second=0, worker="http://worker", tracking=True):
    context = dict(values=[None, None, None], bucket_second=second,
                   activity_origin_time_seconds=-.001, counts={key: counts.get(key, 0) for key in f.COUNTS})
    decision = category in {"admitted", "actual_rejected"}
    return dict(event="feature_admission_decision" if decision else "feature_outcome",
                feature_method=f.METHOD, request_id=rid, worker=worker,
                diagnostic_time_seconds=second + .249, calibration_epoch=0,
                **({"decision": category, "outcome_tracking_supported": tracking if category == "admitted" else False}
                   if decision else {"outcome": category}),
                feature_json=json.dumps(dict(activity=context, counts=context["counts"])))


class ActivityAndScoreTests(unittest.TestCase):
    def test_actual_rejection_never_waives_the_raw_forecast_capture(self):
        try:
            from .wait_ttft_readiness import audit_isolation
        except ImportError:
            from wait_ttft_readiness import audit_isolation
        events = [dict(event="forecast", request_id="a", worker="http://worker"),
                  activity_event("b", "actual_rejected", {"admitted": 1, "actual_rejected": 1})]
        proof = audit_isolation(events, {"a", "b"})
        self.assertFalse(proof["ok"])
        self.assertEqual(proof["missing_owned_forecasts"], ["b"])
        events.append(dict(event="forecast", request_id="b", worker="http://worker"))
        self.assertTrue(audit_isolation(events, {"a", "b"})["ok"])

    def test_foreign_feature_event_and_mismatched_worker_fail_whole_capture(self):
        try:
            from .wait_ttft_readiness import audit_isolation
        except ImportError:
            from wait_ttft_readiness import audit_isolation
        base = [dict(event="forecast", request_id="a", worker="http://worker")]
        foreign = dict(event="feature_outcome_ignored", request_id="foreign", worker="http://worker")
        self.assertFalse(audit_isolation(base + [foreign], {"a"})["ok"])
        wrong = activity_event("a", "admitted", {"admitted": 1}, worker="other")
        self.assertFalse(audit_isolation(base + [wrong], {"a"})["ok"])

    def test_real_rejection_and_terminal_accounting_have_distinct_denominators(self):
        rows = [dict(opc_request_id="a", phase="hard", controlled_rejection=False, first_text_late=False),
                dict(opc_request_id="b", phase="hard", controlled_rejection=True, first_text_late=None,
                     status=503, response_error_code="worker_overload_protection_shed")]
        events = [activity_event("a", "admitted", {"admitted": 1}),
                  activity_event("a", "first_text", {"admitted": 1, "first_text": 1}),
                  activity_event("b", "actual_rejected", {"admitted": 1, "first_text": 1, "actual_rejected": 1})]
        result = f.audit_activity(events, rows)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["actual_rejected_per_offered"], .5)
        self.assertEqual(result["known_terminal_outcomes"], 1)
        self.assertTrue(f.audit_windows(events)["ok"])

    def test_duplicate_terminal_and_unverified_http_rejection_fail(self):
        rows = [dict(opc_request_id="a", phase="hard", controlled_rejection=False, first_text_late=None)]
        events = [activity_event("a", "admitted", {"admitted": 1}),
                  activity_event("a", "first_text", {"admitted": 1, "first_text": 1})]
        self.assertFalse(f.audit_activity(events + events[-1:], rows)["ok"])
        events = [activity_event("a", "actual_rejected", {"actual_rejected": 1})]
        self.assertFalse(f.audit_activity(events, rows)["ok"])

    def test_client_horizon_does_not_invent_router_timeout_category(self):
        row = dict(opc_request_id="a", phase="main", controlled_rejection=False,
                   timed_out=True, cancellation_reason="observation_horizon", first_text_late=True)
        result = f.audit_activity([activity_event("a", "admitted", {"admitted": 1})], [row])
        self.assertFalse(result["ok"])
        self.assertEqual(result["terminal_outcomes"], {})

    def test_explicit_untracked_admission_is_not_an_invented_terminal_or_cutoff(self):
        row = dict(opc_request_id="a", phase="main", controlled_rejection=False, first_text_late=False)
        events = [activity_event("a", "admitted", {"admitted": 1, "untracked_admitted": 1}, tracking=False)]
        result = f.audit_activity(events, [row])
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["unsupported_tracking_admissions"], 1)
        self.assertEqual(result["known_terminal_outcomes"], 0)
        self.assertTrue(f.audit_windows(events)["ok"])

    def test_old_generation_ignored_terminal_preserves_exclusion_without_counting(self):
        row = dict(opc_request_id="a", phase="main", controlled_rejection=False, first_text_late=True)
        events = [activity_event("a", "admitted", {"admitted": 1}),
                  dict(event="feature_outcome_ignored", feature_method=f.METHOD, request_id="a", worker="http://worker",
                       outcome="pretext_downstream_disconnect", reason="generation_changed", diagnostic_time_seconds=5)]
        result = f.audit_activity(events, [row])
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["ignored_terminal_outcomes"], {"generation_changed": 1})
        self.assertEqual(result["terminal_outcomes"], {})

    def test_exact_bucket_expiry_and_future_count_injection(self):
        events = [activity_event("a", "admitted", {"admitted": 1}, second=0),
                  activity_event("b", "admitted", {"admitted": 2}, second=59),
                  activity_event("c", "admitted", {"admitted": 2}, second=60)]
        self.assertTrue(f.audit_windows(events)["ok"])
        changed = copy.deepcopy(events)
        obj = json.loads(changed[1]["feature_json"])
        obj["counts"]["actual_rejected"] = obj["activity"]["counts"]["actual_rejected"] = 1
        changed[1]["feature_json"] = json.dumps(obj)
        self.assertFalse(f.audit_windows(changed)["ok"])

    def test_new_origin_or_epoch_cannot_silently_clear_activity(self):
        first = activity_event("a", "admitted", {"admitted": 1})
        changed = activity_event("b", "admitted", {"admitted": 1})
        value = json.loads(changed["feature_json"])
        value["activity"]["activity_origin_time_seconds"] = 4.999
        changed["feature_json"] = json.dumps(value)
        changed["diagnostic_time_seconds"] = 5.249
        result = f.audit_windows([first, changed])
        self.assertFalse(result["ok"])
        self.assertTrue(any("origin changed" in error["reason"] for error in result["errors"]))
        changed["calibration_epoch"] = 1
        self.assertFalse(f.audit_windows([first, changed])["ok"])

    def test_captured_reset_starts_new_counts_and_ignored_old_outcome_does_not(self):
        first = activity_event("a", "admitted", {"admitted": 1})
        reset = dict(event="calibration_reset", worker="http://worker", calibration_epoch=1,
                     reason="queue_drained")
        second = activity_event("b", "admitted", {"admitted": 1})
        second["calibration_epoch"] = 1
        second["diagnostic_time_seconds"] = 5.249
        value = json.loads(second["feature_json"])
        value["activity"]["activity_origin_time_seconds"] = 4.999
        second["feature_json"] = json.dumps(value)
        ignored = dict(event="feature_outcome_ignored", request_id="a", worker="http://worker",
                       calibration_epoch=1, admission_epoch=0, reason="generation_changed",
                       outcome="pretext_downstream_disconnect")
        result = f.audit_windows([first, reset, ignored, second])
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["activity_events"], 2)
        self.assertEqual(result["worker_generations"], 2)

    def test_whole_cohort_fallback_keeps_unknown_and_fail_open_flags(self):
        rows = [dict(first_text_late=False, client_ttft_seconds=10,
                     pre_admission_total_ttft_predictions={"calibrated": 400},
                     pre_admission_recorded_extra_flags={"calibrated": False},
                     feature_context_candidate={"total_ttft_seconds": 200, "contextual": True,
                                                "recorded_extra_would_reject": None}),
                dict(first_text_late=True, client_ttft_seconds=None,
                     pre_admission_total_ttft_predictions={"calibrated": 350},
                     pre_admission_recorded_extra_flags={"calibrated": True},
                     feature_context_candidate={"total_ttft_seconds": 200, "contextual": True,
                                                "recorded_extra_would_reject": False}),
                dict(first_text_late=None, controlled_rejection=True)]
        cards = f.scores(rows)
        self.assertEqual(cards["feature_reported_extra_flag"]["unknown"], 2)
        self.assertEqual(cards["feature_legacy_fallback_extra_flag"]["unknown"], 1)
        self.assertEqual(cards["feature_legacy_fallback_extra_flag"]["tn"], 1)
        self.assertEqual(cards["feature_legacy_fallback_total"]["fn"], 1)
        self.assertEqual(cards["feature_legacy_fallback_total"]["censored_known_late"], 1)
        self.assertEqual(cards["feature_legacy_fallback_total"]["offered"], 3)


if __name__ == "__main__":
    unittest.main()
