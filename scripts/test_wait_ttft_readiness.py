"""Current-epoch readiness/isolation regressions, including busy-start controls."""
import copy
import json
import unittest

try:
    from scripts.wait_ttft_readiness import METHOD, Readiness, audit_isolation, native_completion_stats_proof
except ModuleNotFoundError:
    from wait_ttft_readiness import METHOD, Readiness, audit_isolation, native_completion_stats_proof


class NativeCompletionStatsTests(unittest.TestCase):
    VLLM = ["vllm:request_prompt_tokens_count", "vllm:request_prompt_tokens_sum",
            "vllm:request_prefill_kv_computed_tokens_count", "vllm:request_prefill_kv_computed_tokens_sum",
            "vllm:e2e_request_latency_seconds_count"]
    SGLANG = ["sglang:prompt_tokens_total", "sglang:e2e_request_latency_seconds_count"]

    def scrape(self, names, value, labels='model_name="test-model",engine="0"'):
        return "\n".join(f"{name}{{{labels}}} {value}" for name in names)

    def test_positive_delta_for_each_backend_and_native_sglang_alias(self):
        for backend, names in [("vllm", self.VLLM), ("sglang", self.SGLANG),
                               ("sglang", [name.replace(":", "_") for name in self.SGLANG])]:
            with self.subTest(backend=backend, names=names):
                proof = native_completion_stats_proof(self.scrape(names, 10), self.scrape(names, 12), backend)
                self.assertTrue(proof["ok"], proof)
                self.assertEqual(len(proof["series"]), len(names))
                self.assertTrue(all(row["before"] == 10 and row["after"] == 12 and row["delta"] == 2
                                    for row in proof["series"]))

    def test_unchanged_vllm_completion_stats_fail(self):
        initial = self.scrape(self.VLLM, 123131)
        proof = native_completion_stats_proof(initial, initial, "vllm")
        self.assertFalse(proof["ok"])
        self.assertTrue(all(row["delta"] == 0 and "Counter did not advance" in row["reasons"]
                            for row in proof["series"]))

    def test_prompt_and_e2e_advance_cannot_hide_frozen_consumed_prefill_histogram(self):
        current = "\n".join(self.scrape([name], 10 if "prefill_kv" in name else 12) for name in self.VLLM)
        proof = native_completion_stats_proof(self.scrape(self.VLLM, 10), current, "vllm")
        self.assertFalse(proof["ok"])
        frozen = [row["metric"] for row in proof["series"] if row["reasons"]]
        self.assertEqual(set(frozen), {name for name in self.VLLM if "prefill_kv" in name})

    def test_reset_is_not_hidden_by_other_series_growth(self):
        initial = self.scrape(self.SGLANG, 10) + "\n" + self.scrape(self.SGLANG, 10, 'engine="1"')
        current = self.scrape(self.SGLANG, 9) + "\n" + self.scrape(self.SGLANG, 100, 'engine="1"')
        proof = native_completion_stats_proof(initial, current, "sglang")
        self.assertFalse(proof["ok"])
        self.assertEqual(sum("Counter reset or decreased" in row["reasons"] for row in proof["series"]), 2)

    def test_missing_or_new_unmatched_series_fail_closed(self):
        initial = self.scrape(self.VLLM, 10)
        for current in [self.scrape(self.VLLM[:-1], 12),
                        self.scrape(self.VLLM, 12, 'model_name="other-model",engine="0"'),
                        self.scrape(self.VLLM, 12) + "\n" + self.scrape(self.VLLM, 1, 'engine="new"')]:
            with self.subTest(current=current):
                proof = native_completion_stats_proof(initial, current, "vllm")
                self.assertFalse(proof["ok"])
                self.assertTrue(any("Series is not present in both scrapes" in row["reasons"]
                                    for row in proof["series"]))
        self.assertFalse(native_completion_stats_proof("", "", "vllm")["ok"])

    def test_nonfinite_negative_or_malformed_required_samples_fail(self):
        for value in ["NaN", "+Inf", "-1", "invalid"]:
            with self.subTest(value=value):
                proof = native_completion_stats_proof(self.scrape(self.SGLANG, 10),
                                                       self.scrape(self.SGLANG, value), "sglang")
                self.assertFalse(proof["ok"])
                json.dumps(proof, allow_nan=False)
        duplicate = self.scrape(self.SGLANG, 12) + "\n" + self.scrape(self.SGLANG, 13)
        self.assertFalse(native_completion_stats_proof(self.scrape(self.SGLANG, 10), duplicate, "sglang")["ok"])

    def test_label_order_is_canonical_and_receipt_excludes_raw_labels_and_scrape(self):
        initial = self.scrape(self.SGLANG, 10, 'model_name="private-prompt",token="private-credential"')
        current = self.scrape(self.SGLANG, 12, 'token="private-credential",model_name="private-prompt"')
        current += '\n# private-comment\nunrelated_metric{body="private-body"} 1'
        proof = native_completion_stats_proof(initial, current, "sglang")
        self.assertTrue(proof["ok"], proof)
        receipt = json.dumps(proof)
        for secret in ["private-prompt", "private-credential", "private-comment", "private-body"]:
            self.assertNotIn(secret, receipt)

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            native_completion_stats_proof("", "", "other")


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

    def test_foreign_forecast_after_readiness_without_feedback_fails_whole_capture(self):
        self.learn()
        self.assertTrue(self.r.snapshot(True)["ready"])
        capture = [dict(event="forecast", request_id="prod-test-0", timestamp="2026-10-19 08:18:31"),
                   dict(event="readiness_verified", timestamp="2026-10-19 08:19:31"),
                   dict(event="forecast", request_id="foreign-nonstream", timestamp="2026-10-19 08:23:55",
                        calibration_enabled=False, stream=False)]
        result = audit_isolation(capture, {"prod-test-0"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["foreign_event_count"], 1)
        self.assertEqual(result["foreign_requests"][0]["request_id"], "foreign-nonstream")
        self.assertEqual(result["foreign_requests"][0]["events"], [dict(capture_event_index=3,
                         event_kind="forecast", timestamp="2026-10-19 08:23:55")])


class WholeCaptureIsolationTests(unittest.TestCase):
    def test_exact_owned_forecasts_and_feedback_pass(self):
        events = [dict(event=kind, request_id=rid) for rid in ["owned-0", "owned-1"]
                  for kind in ["forecast", "calibration_forecast", "first_text", "calibration_feedback"]]
        result = audit_isolation(iter(events), {"owned-0", "owned-1"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["checked_event_count"], 8)
        self.assertEqual(result["owned_forecast_request_count"], 2)
        self.assertEqual(result["missing_owned_forecasts"], [])

    def test_same_prefix_unplanned_id_is_foreign(self):
        result = audit_isolation([dict(event="forecast", request_id="owned-0"),
                                  dict(event="forecast", request_id="owned-999")], {"owned-0"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["foreign_requests"][0]["request_id"], "owned-999")

    def test_each_request_event_kind_detects_foreign_without_worker_or_method(self):
        for kind in ["forecast", "calibration_forecast", "calibration_feedback", "first_text", "request_transport"]:
            with self.subTest(kind=kind):
                result = audit_isolation([dict(event="forecast", request_id="owned"),
                                          dict(event=kind, request_id="foreign")], {"owned"})
                self.assertFalse(result["ok"])
                self.assertEqual(result["foreign_requests"][0]["event_kind_counts"], {kind: 1})

    def test_unknown_request_kind_and_lifecycle_span_are_not_false_positives(self):
        result = audit_isolation([dict(event="forecast", request_id="owned"),
                                  dict(event="unrelated_request_log", request_id="foreign"),
                                  dict(event="calibration_update", span=dict(request_id="foreign"))], {"owned"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["ignored_event_count"], 2)

    def test_tracing_fields_are_normalized_and_foreign_events_are_grouped(self):
        events = [dict(timestamp="t0", fields=dict(event="forecast", request_id="owned")),
                  dict(timestamp="t1", fields=dict(event="forecast", request_id="foreign")),
                  dict(timestamp="t2", event="first_text", request_id="foreign")]
        result = audit_isolation(events, {"owned"})
        self.assertFalse(result["ok"])
        self.assertEqual(result["foreign_request_count"], 1)
        self.assertEqual(result["foreign_event_count"], 2)
        self.assertEqual([x["timestamp"] for x in result["foreign_requests"][0]["events"]], ["t1", "t2"])

    def test_missing_request_id_fails_closed(self):
        for missing in [None, "", " "]:
            with self.subTest(missing=missing):
                result = audit_isolation([dict(event="forecast", request_id="owned"),
                                          dict(event="first_text", request_id=missing)], {"owned"})
                self.assertFalse(result["ok"])
                self.assertEqual(result["unattributed_event_count"], 1)

    def test_empty_or_partial_capture_cannot_prove_isolation(self):
        for events in [[], [dict(event="first_text", request_id="owned-0")],
                       [dict(event="forecast", request_id="owned-0")]]:
            with self.subTest(events=events):
                result = audit_isolation(events, {"owned-0", "owned-1"})
                self.assertFalse(result["ok"])
                self.assertIn("owned-1", result["missing_owned_forecasts"])

    def test_receipt_never_includes_request_body_or_log_message(self):
        result = audit_isolation([dict(event="forecast", request_id="owned"),
                                  dict(event="forecast", request_id="foreign", body="private-body",
                                       message="private-message", fields=dict(prompt="private-prompt"))], {"owned"})
        receipt = json.dumps(result)
        for text in ["private-body", "private-message", "private-prompt"]:
            self.assertNotIn(text, receipt)

    def test_owned_set_must_be_explicit_and_nonempty(self):
        for invalid in ["owned-prefix-", set(), {""}, {None}]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                audit_isolation([], invalid)


if __name__ == "__main__":
    unittest.main()
