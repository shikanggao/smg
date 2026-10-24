#!/usr/bin/env python3
"""Planner, deadline, SSE and real overlapping HTTP checks; no engine required."""
import argparse
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("production_mix", HERE / "run_wait_production_mix.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
DEFAULT = json.loads((HERE / "wait-production-mix.json").read_text())


def smoke_profile():
    value = copy.deepcopy(DEFAULT)
    value.update(deadline_seconds=.07, router_extra_ttft_budget_seconds=.07, max_inflight=32, sample_interval_seconds=.01,
                 drain_timeout_seconds=1, maximum_p99_dispatch_lag_seconds=.2)
    value["phases"] = [dict(name="pressure", seconds=.3, rps=80, arrivals="periodic")]
    return value


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p.validate(DEFAULT)
        cls.rows = p.make_plan(DEFAULT)

    def test_reproducible_and_bounded_arrivals(self):
        self.assertEqual(self.rows, p.make_plan(DEFAULT))
        self.assertLessEqual(len(self.rows), DEFAULT["max_requests"])
        for phase in DEFAULT["phases"]:
            arrivals = [r["scheduled_phase_seconds"] for r in self.rows if r["phase"] == phase["name"]]
            self.assertTrue(arrivals)
            self.assertEqual(arrivals, sorted(set(arrivals)))
            self.assertLess(max(arrivals), phase["seconds"])

    def test_all_length_quadrants_and_natural_output_budgets(self):
        for small_input, long_output in [(True, False), (True, True), (False, False), (False, True)]:
            self.assertTrue(any((r["input_words"] < 1024) == small_input and (r["max_output_tokens"] >= 1024) == long_output for r in self.rows))
        self.assertEqual({r["cohort"] for r in self.rows}, set(p.TASKS))
        self.assertTrue(any(r["shared_prefix"] for r in self.rows))
        self.assertTrue(any(not r["shared_prefix"] for r in self.rows))

    def test_actual_prompt_words_and_saved_hash(self):
        for row in self.rows[::97]:
            messages = p.messages(row, DEFAULT)
            self.assertEqual(sum(len(m["content"].split()) for m in messages), row["input_words"])
            self.assertEqual(p.digest(messages), row["prompt_sha256"])

    def test_shared_prefix_survives_unique_request_suffix(self):
        row = dict(self.rows[0], shared_prefix=True, input_words=1024, session=3)
        other = dict(row, index=row["index"] + 9999)
        self.assertEqual(p.messages(row, DEFAULT)[0], p.messages(other, DEFAULT)[0])
        self.assertNotEqual(p.messages(row, DEFAULT)[1], p.messages(other, DEFAULT)[1])
        row["shared_prefix"] = other["shared_prefix"] = False
        self.assertNotEqual(p.messages(row, DEFAULT)[0], p.messages(other, DEFAULT)[0])

    def test_cache_namespace_changes_prefix_without_changing_arrivals_or_words(self):
        profile = dict(DEFAULT, cache_namespace="repeat_a")
        for row in self.rows[::97]:
            changed = p.messages(row, profile)
            self.assertEqual(sum(len(m["content"].split()) for m in changed), row["input_words"])
            self.assertNotEqual(p.digest(changed), row["prompt_sha256"])
            self.assertTrue(changed[0]["content"].startswith("Experiment repeat_a."))
        other = p.make_plan(profile)
        self.assertEqual([{k: v for k, v in r.items() if k != "prompt_sha256"} for r in self.rows],
                         [{k: v for k, v in r.items() if k != "prompt_sha256"} for r in other])

    def test_balanced_sparse_controls_and_pressure_longer_than_deadline(self):
        for phase in DEFAULT["phases"]:
            if phase.get("balanced_controls"):
                counts = p.Counter(r["cohort"] for r in self.rows if r["phase"] == phase["name"])
                self.assertGreaterEqual(min(counts.values()), DEFAULT["coverage"]["minimum_healthy_controls_per_cohort"])
            if phase["name"].startswith("pressure"):
                self.assertGreater(phase["seconds"], DEFAULT["deadline_seconds"])

    def test_raw_hard_continuation_keeps_the_approved_fixed_bound(self):
        profile = json.loads((HERE / "wait-ttft-raw-hard.json").read_text())
        p.validate(profile)
        rows = p.make_plan(profile)
        self.assertEqual(p.Counter(r["phase"] for r in rows),
                         {"readiness-learn": 63, "readiness-refresh": 6, "hard-observe": 51})
        self.assertEqual(len(rows), profile["bounded_continuation"]["maximum_offered_requests"])
        last_arrival = max(r["scheduled_phase_seconds"] for r in rows if r["phase"] == "hard-observe")
        maximum_to_last_offer = (sum(phase["seconds"] for phase in profile["phases"][:-1])
                                + profile["readiness"]["timeout_seconds"] + last_arrival)
        self.assertLessEqual(maximum_to_last_offer,
                             profile["bounded_continuation"]["maximum_arrival_wall_seconds"])
        self.assertEqual(profile["raw_policy_mode"], "hard")
        self.assertEqual(profile["router_extra_ttft_budget_seconds"], 5)
        self.assertEqual(profile["deadline_seconds"], 300)
        self.assertEqual(profile["observation_horizon_seconds"], 420)
        self.assertEqual(profile["ttft_stream_policy"], "complete_bounded_response")
        self.assertEqual(profile["bounded_continuation"]["rejected_outcome"], "unknown")

    def test_deadline_pair_has_matched_continuous_pressure_and_separate_response_limit(self):
        plans = []
        for name, policy, numeric_late_min in [("deadline300", "cancel_if_no_first_text", 0),
                                                ("observe420", "observe", 1)]:
            profile = json.loads((HERE / ("wait-ttft-" + name + ".json")).read_text())
            p.validate(profile)
            self.assertEqual(profile["ttft_deadline_policy"], policy)
            self.assertEqual(profile["deadline_seconds"], 300)
            self.assertEqual(profile["observation_horizon_seconds"], 420)
            self.assertEqual(profile["coverage"]["minimum_observed_late_ttft"], numeric_late_min)
            self.assertEqual(profile["readiness"]["before_phase"], "pressure")
            self.assertTrue(profile["readiness"]["require_entry_epoch_continuity"])
            self.assertEqual([(x["name"], x["seconds"], x["rps"]) for x in profile["phases"]],
                             [("readiness-learn", 60, 4), ("pressure", 420, 2),
                              ("pressure-high", 180, 4), ("recovery", 720, .1)])
            rows = p.make_plan(profile)
            self.assertEqual(len(rows), 1868)
            plans.append([{k: v for k, v in r.items() if k != "prompt_sha256"} for r in rows])
        self.assertEqual(plans[0], plans[1])

    def test_smoke_never_claims_300_seconds(self):
        with self.assertRaisesRegex(AssertionError, "300-second"):
            p.validate(smoke_profile())
        p.validate(smoke_profile(), smoke=True)

    def test_invalid_or_unbounded_profiles_rejected(self):
        for key, value in [("max_inflight", 0), ("shared_prefix_fraction", 1.1), ("sample_interval_seconds", 0)]:
            profile = dict(DEFAULT, **{key: value})
            with self.assertRaises(AssertionError):
                p.validate(profile)
        with self.assertRaisesRegex(AssertionError, "safety bound"):
            p.make_plan(dict(smoke_profile(), max_requests=1))

    def test_frames_split_utf8_crlf_comments_and_multiline(self):
        decoder = p.Frames()
        raw = ': heartbeat\r\ndata: {"text":"世"}\r\n\r\ndata: one\r\ndata: two\r\n\r\ndata: [DONE]\r\n\r\n'.encode()
        actual = []
        for byte in raw:
            actual += decoder.feed(bytes([byte]))
        actual += decoder.feed(b"", final=True)
        self.assertEqual(actual, ['{"text":"世"}', 'one\ntwo', '[DONE]'])

    def test_native_metric_profiles_and_missing_not_zero(self):
        value = p.native_metrics('vllm:num_requests_waiting{model="a"} 2\nvllm:num_requests_running 3\nvllm:gpu_cache_usage_perc .95\n', "vllm")
        self.assertEqual({k: value[k] for k in ["native_waiting", "native_running", "native_kv"]}, dict(native_waiting=2, native_running=3, native_kv=.95))
        self.assertIsNone(value["native_completed"])
        sg = p.native_metrics('sglang_num_queue_reqs 4\nsglang_num_running_reqs 5\nsglang_token_usage .9\n', "sglang")
        self.assertEqual({k: sg[k] for k in ["native_waiting", "native_running", "native_kv"]}, dict(native_waiting=4, native_running=5, native_kv=.9))
        self.assertIsNone(p.native_metrics('vllm:num_requests_running NaN\n', "vllm")["native_running"])

    def test_native_progress_counters_keep_missing_unknown_and_avoid_alias_double_count(self):
        text = ('vllm:request_success_total{finished_reason="stop"} 7\n'
                'vllm:request_success_total{finished_reason="length"} 3\n'
                'vllm:e2e_request_latency_seconds_count 10\n'
                'vllm:generation_tokens_total 48\n'
                'vllm:prompt_tokens_total 100\n'
                'vllm:time_to_first_token_seconds_count 12\n')
        value = p.native_metrics(text, "vllm")
        self.assertEqual(value["native_completed"], 10)
        self.assertEqual(value["native_generation_tokens"], 48)
        self.assertEqual(value["native_first_tokens"], 12)
        self.assertIsNone(value["native_preemptions"])
        sg = p.native_metrics('sglang:e2e_request_latency_seconds_count 8\nsglang:prompt_tokens_total 123\n', "sglang")
        self.assertEqual(sg["native_completed"], 8)
        self.assertEqual(sg["native_prompt_tokens"], 123)
        self.assertIsNone(sg["native_generation_tokens"])

    def test_live_progress_and_abort_counts_are_aggregate_and_missing_unknown(self):
        vllm = p.native_metrics('vllm:iteration_tokens_total_sum 321\n', "vllm")
        self.assertEqual(vllm["native_iteration_tokens"], 321)
        self.assertIsNone(vllm["native_aborted"])
        sg = p.native_metrics('sglang:num_aborted_requests_total 4\n'
                              'sglang_num_aborted_requests_total 40\n'
                              'sglang:gen_throughput 12\n', "sglang")
        self.assertEqual(sg["native_aborted"], 4)
        self.assertEqual(sg["native_generation_throughput"], 12)
        self.assertIsNone(sg["native_iteration_tokens"])
        missing = p.native_metrics('sglang:gen_throughput NaN\n', "sglang")
        self.assertIsNone(missing["native_generation_throughput"])

    def test_unknown_outcomes_do_not_become_false_positives(self):
        rows = [dict(pred=True, truth=None), dict(pred=True, truth=False, controlled_rejection=True),
                dict(pred=False, truth=True), dict(pred=True, truth=False)]
        self.assertEqual(p.classify(rows, "pred", "truth"), dict(tp=0, fn=1, fp=1, tn=0, unknown=2))

    def test_post_run_foreign_forecast_invalidates_otherwise_valid_ttft_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = dict(DEFAULT, measurement_target="ttft", observation_horizon_seconds=420)
            profile["coverage"] = dict(DEFAULT["coverage"], minimum_pressure_ttft_misses=1, minimum_observed_late_ttft=1)
            row = dict(self.rows[0], opc_request_id="prod-isolated-0", complete=True,
                client_ttft_seconds=1, first_text_late=False, completion_late=False,
                e2e_seconds=2, first_answer_seconds=1, dispatch_lag_seconds=0,
                timed_out=False, controlled_rejection=False, cancelled=False, error=None, usage={})
            for name, value in {
                "profile.json": profile, "run-config.json": dict(run_id="isolated", smoke=False),
                "execution.json": dict(error=None, planned_requests=1, native_drained=True),
                "readiness.json": dict(ready=True, elapsed_seconds=.5),
            }.items():
                (root / name).write_text(json.dumps(value))
            (root / "requests.jsonl").write_text(json.dumps(row) + "\n")
            (root / "metrics.jsonl").write_text("")
            gate = dict(event="forecast", request_id=row["opc_request_id"], timestamp="2026-10-19 00:00:00",
                threshold_seconds=300, predicted_ttft_seconds=1, calibrated_ttft_seconds=1,
                would_reject=False, calibrated_would_reject=False, baseline_available=True,
                calibration_baseline_available=True, calibration_baseline_seconds=1,
                calibration_input_excess_seconds=0, calibration_method="mse_poll_two_wave_queue_v2")
            events = [gate, dict(gate, event="calibration_forecast"),
                      dict(event="first_text", request_id=row["opc_request_id"], ttft_seconds=1)]
            log = root / "captured.jsonl"
            log.write_text("".join(json.dumps(event) + "\n" for event in events))
            before = p.analyze(root, log)
            self.assertEqual(before["execution_status"], "PASS")
            self.assertTrue(before["execution_checks"]["exclusive_workload_isolation"])
            events.append(dict(gate, request_id="foreign-nonstream", timestamp="2026-10-19 00:01:00"))
            log.write_text("".join(json.dumps(event) + "\n" for event in events))
            after = p.analyze(root, log)
            self.assertEqual(after["execution_status"], "INVALID")
            self.assertFalse(after["execution_checks"]["exclusive_workload_isolation"])
            self.assertEqual(after["runtime_isolation"]["foreign_request_count"], 1)
            self.assertEqual(after["runtime_isolation"]["foreign_requests"][0]["request_id"], "foreign-nonstream")
            missing = p.analyze(root)
            self.assertFalse(missing["execution_checks"]["exclusive_workload_isolation"])

    def test_completion_seed_keeps_feature_activity_and_whole_capture_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = dict(DEFAULT, feature_context_required=True)
            rid = "prod-seed-0"
            row = dict(self.rows[0], opc_request_id=rid, complete=True,
                client_ttft_seconds=1, first_text_late=False, completion_late=False,
                e2e_seconds=2, first_answer_seconds=1, dispatch_lag_seconds=0,
                timed_out=False, controlled_rejection=False, cancelled=False, error=None, usage={})
            for name, value in {"profile.json": profile,
                "run-config.json": dict(run_id="seed", smoke=False),
                "execution.json": dict(error=None, planned_requests=1, native_drained=True)}.items():
                (root / name).write_text(json.dumps(value))
            (root / "requests.jsonl").write_text(json.dumps(row) + "\n")
            (root / "metrics.jsonl").write_text("")
            gate = dict(event="forecast", request_id=rid, worker="http://worker",
                threshold_seconds=300, predicted_ttft_seconds=1, calibrated_ttft_seconds=1,
                would_reject=False, calibrated_would_reject=False, baseline_available=True,
                calibration_baseline_available=True, calibration_baseline_seconds=1,
                calibration_input_excess_seconds=0, calibration_method="mse_poll_two_wave_queue_v2")
            counts = dict(first_text=0, upstream_timeout=0, pretext_downstream_disconnect=0,
                          unknown_pretext_end=0, admitted=1, actual_rejected=0, untracked_admitted=0)
            def activity(kind, time):
                context = dict(values=[None, None, None], bucket_second=0,
                               activity_origin_time_seconds=0, counts=dict(counts))
                return dict(event=kind, feature_method="feature_context_ridge_v1", request_id=rid,
                    worker="http://worker", calibration_epoch=0, diagnostic_time_seconds=time,
                    feature_json=json.dumps(dict(activity=context, counts=context["counts"])))
            admitted = activity("feature_admission_decision", .1)
            admitted.update(decision="admitted", outcome_tracking_supported=True)
            counts["first_text"] = 1
            first = activity("feature_outcome", .2);first["outcome"] = "first_text"
            events = [gate, dict(gate, event="calibration_forecast"), admitted, first,
                      dict(event="first_text", request_id=rid, ttft_seconds=1),
                      dict(gate, request_id="foreign-nonstream")]
            log = root / "captured.jsonl";log.write_text("".join(json.dumps(e) + "\n" for e in events))
            summary = p.analyze(root, log)
            self.assertEqual(summary["feature_context"]["activity"]["selected_admissions"], 1)
            self.assertEqual(summary["feature_context"]["activity"]["known_terminal_outcomes"], 1)
            self.assertTrue(summary["feature_context"]["causal_windows"]["ok"])
            self.assertEqual(summary["runtime_isolation"]["foreign_request_count"], 1)
            self.assertFalse(summary["execution_checks"]["exclusive_workload_isolation"])

    def test_versioned_ttft_profiles_validate_and_busy_start_is_focused_skip(self):
        for name in ["wait-ttft-focused.json", "wait-ttft-busy-start.json", "wait-ttft-seed.json"]:
            profile = json.loads((HERE / name).read_text())
            p.validate(profile)
            rows = p.make_plan(profile)
            self.assertLessEqual(len(rows), profile["max_requests"])
        suite = json.loads((HERE / "wait-ttft-suite.json").read_text())
        busy = next(c for c in suite["full_suite_cases"] if c["name"] == "busy-start-provisional")
        self.assertFalse(busy["focused_enabled"])
        self.assertEqual(json.loads((HERE / busy["profile"]).read_text())["startup_scenario"], "busy_start")

    def test_ttft_profile_requires_longer_bounded_horizon_and_generation_bound(self):
        profile = json.loads((HERE / "wait-ttft-focused.json").read_text())
        for horizon in [300, 601, float("inf")]:
            with self.assertRaises(AssertionError):
                p.validate(dict(profile, observation_horizon_seconds=horizon))
        profile["cohorts"][0]["max_output_tokens"] = [32, 16384]
        with self.assertRaisesRegex(AssertionError, "unnecessary generation"):
            p.validate(profile)


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from aiohttp import web
        self.web = web
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.mode = "mixed"
        self.active = self.peak = self.posts = 0
        self.router_budget = .07
        self.payloads = []
        self.authorizations = []
        self.request_ids = []
        app = web.Application()
        app.router.add_get("/metrics", self.metrics)
        app.router.add_post("/v1/chat/completions", self.completion)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"

    async def asyncTearDown(self):
        await self.runner.cleanup()
        self.temp.cleanup()

    async def metrics(self, request):
        return self.web.Response(text='vllm:num_requests_waiting 0\nvllm:num_requests_running 0\nvllm:kv_cache_usage_perc .95\n' + f'smg_estimated_wait_threshold_seconds {self.router_budget}\n')

    async def completion(self, request):
        self.authorizations.append(request.headers.get("Authorization"))
        self.request_ids.append((request.headers.get("opc-request-id"), request.headers.get("x-request-id")))
        body = await request.json()
        self.payloads.append(body)
        self.posts += 1
        if self.mode == "stalled-headers":
            await asyncio.sleep(.2)
        if self.mode == "stalled-http-error":
            response = self.web.StreamResponse(status=500)
            await response.prepare(request)
            await asyncio.sleep(.2)
            return response
        index = int(request.headers["opc-request-id"].rsplit("-", 1)[1])
        mode = index % 4 if self.mode == "mixed" else 1 if self.mode in {"slow", "late-first-text"} else 2 if self.mode == "early-text-stalled-generation" else 0
        if mode == 3:
            return self.web.json_response({"error": {"code": "worker_overload_protection_shed",
                "message": "estimated wait budgets exceeded"}}, status=503)
        self.active += 1
        self.peak = max(self.peak, self.active)
        response = self.web.StreamResponse(headers={"Content-Type": "text/event-stream", "x-request-id": "native-header-id"})
        async def event(value):
            value["id"] = "native-sse-id"
            await response.write(("data: " + json.dumps(value) + "\n\n").encode())
        try:
            await response.prepare(request)
            await event({"choices": [{"delta": {"role": "assistant"}}]})
            if mode == 1:
                await asyncio.sleep(.2)
                if self.mode == "late-first-text":
                    await event({"choices": [{"delta": {"content": "late"}}]})
                    await asyncio.sleep(.1)
            elif mode == 2:
                await event({"choices": [{"delta": {"content": "first"}}]})
                await asyncio.sleep(.2)
            else:
                await asyncio.sleep(.005)
                await event({"choices": [{"delta": {"reasoning_content": "reason"}}]})
                await asyncio.sleep(.005)
                await event({"choices": [{"delta": {"content": "answer"}}]})
            if self.mode != "incomplete":
                await event({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "completion_tokens": 8}})
                await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
        except ConnectionResetError:
            pass
        finally:
            self.active -= 1
        return response

    async def exercise(self, profile=None):
        profile = profile or smoke_profile()
        p.write_plan(self.root / "plan", profile, smoke=True)
        args = argparse.Namespace(plan=self.root / "plan", output=self.root / "run", url=self.url,
                                  model="mock", engine="vllm", engine_metrics_url=self.url + "/metrics",
                                  router_metrics_url=self.url + "/metrics", router_events=None, run_id="smoke")
        return await p.run(args)

    def rows(self):
        return [json.loads(s) for s in (self.root / "run/requests.jsonl").read_text().splitlines()]

    async def test_overlapping_arrivals_natural_eos_and_deadline_censoring(self):
        summary = await self.exercise()
        rows = self.rows()
        self.assertGreater(self.peak, 1, "Slow responses must not serialize offered arrivals")
        self.assertEqual(summary["execution_status"], "PASS")
        self.assertEqual(summary["production_coverage"], "INSUFFICIENT")
        self.assertFalse(summary["coverage_checks"]["actual_300_second_deadline"])
        self.assertTrue(all("ignore_eos" not in b and "min_tokens" not in b for b in self.payloads))
        self.assertTrue(any(r["timed_out"] and r["client_ttft_seconds"] is None and r["first_text_late"] is True for r in rows))
        self.assertTrue(any(r["timed_out"] and r["client_ttft_seconds"] is not None and r["first_text_late"] is False for r in rows))
        for row in rows:
            if row["complete"]:
                self.assertGreater(row["client_ttft_seconds"], row["first_sse_seconds"])
                self.assertGreater(row["first_answer_seconds"], row["client_ttft_seconds"])
                self.assertEqual(row["usage"]["completion_tokens"], 8)
                self.assertLess(8, row["max_output_tokens"])
            if row["controlled_rejection"]:
                self.assertIsNone(row["completion_late"])
        self.assertTrue(all(v["unknown"] == len(rows) for v in [summary["scores"]["pressure"]["raw"]["total_ttft_prediction"]]))

    async def test_serving_credential_is_header_only_and_never_persisted(self):
        self.mode = "normal"
        token = "unit-test-ephemeral-not-a-real-credential"
        with mock.patch.dict(p.os.environ, {"SMG_EXPERIMENT_INGRESS_KEY": token}):
            result = await self.exercise(dict(smoke_profile(), serving_auth_env="SMG_EXPERIMENT_INGRESS_KEY"))
        self.assertEqual(result["execution_status"], "PASS")
        self.assertTrue(self.authorizations)
        self.assertTrue(all(value == "Bearer " + token for value in self.authorizations))
        for file in (self.root / "run").rglob("*"):
            if file.is_file():
                self.assertNotIn(token, file.read_text())

    async def test_missing_serving_credential_fails_before_requests(self):
        with mock.patch.dict(p.os.environ, {"SMG_EXPERIMENT_INGRESS_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "credential is unavailable"):
                await self.exercise(dict(smoke_profile(), serving_auth_env="SMG_EXPERIMENT_INGRESS_KEY"))
        self.assertEqual(self.posts, 0)

    async def test_incomplete_stream_is_not_success(self):
        self.mode = "incomplete"
        # Exercise protocol EOF, independently of the short deadline/load test.
        # One immediate response has ample time to reach EOF without racing a
        # valid timeout (whose completion_late=True has different semantics).
        self.router_budget = 10
        profile = dict(smoke_profile(), deadline_seconds=10,
                       router_extra_ttft_budget_seconds=10, max_inflight=1,
                       phases=[dict(name="pressure", seconds=.1, rps=20, arrivals="periodic")])
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            await self.exercise(profile)
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["error"], "Incomplete SSE response")
        self.assertFalse(row["timed_out"])
        self.assertFalse(row["complete"])
        self.assertFalse(row["done_seen"])
        self.assertIsNone(row["completion_late"])

    async def test_opt_in_content_preview_is_bounded_and_excludes_reasoning(self):
        self.mode = "normal"
        profile = dict(smoke_profile(), capture_content_chars=3)
        await self.exercise(profile)
        for row in self.rows():
            self.assertEqual(row["content_preview"], "ans")
            self.assertEqual(row["content_characters"], len("answer"))

    async def ttft_request(self, horizon=.3, stream_policy="abort_after_first_text", deadline_policy="observe"):
        import aiohttp
        profile = dict(smoke_profile(), measurement_target="ttft", observation_horizon_seconds=horizon,
                       ttft_stream_policy=stream_policy, ttft_deadline_policy=deadline_policy,
                       readiness=dict(before_phase="pressure", timeout_seconds=.05))
        output = self.root / "ttft-request"
        output.mkdir()
        events = self.root / "router-events.jsonl"
        events.touch()
        args = argparse.Namespace(output=output, model="mock", url=self.url, run_id="ttft", router_events=events)
        workload = p.Run(args, profile)
        item = p.make_plan(profile)[0]
        async with aiohttp.ClientSession() as session:
            await workload.request(session, item, p.time.monotonic())
        workload.stream.close()
        workload.metrics_stream.close()
        workload.client_events.close()
        return workload.rows[0]

    async def test_ttft_mode_stops_after_first_text_without_claiming_completion(self):
        self.mode = "normal"
        row = await self.ttft_request()
        self.assertTrue(row["intentional_stream_abort"])
        self.assertIsNone(row["error"])
        self.assertFalse(row["complete"])
        self.assertIsNone(row["completion_late"])
        self.assertFalse(row["first_text_late"])
        self.assertIsNotNone(row["client_ttft_seconds"])

    async def test_ttft_normal_completion_keeps_usage_done_and_first_text(self):
        self.mode = "normal"
        row = await self.ttft_request(stream_policy="complete_bounded_response")
        self.assertTrue(row["complete"])
        self.assertTrue(row["done_seen"])
        self.assertEqual(row["finish_reason"], "stop")
        self.assertEqual(row["usage"], dict(prompt_tokens=100, completion_tokens=8))
        self.assertFalse(row["intentional_stream_abort"])
        self.assertFalse(row["first_text_late"])
        self.assertLess(row["client_ttft_seconds"], row["e2e_seconds"])
        self.assertIsNone(row["error"])
        self.assertEqual(self.request_ids, [(row["opc_request_id"], row["opc_request_id"])])
        self.assertEqual(row["upstream_response_request_id"], "native-header-id")
        self.assertEqual(row["upstream_sse_response_id"], "native-sse-id")
        self.assertIsNone(row["cancellation_delivered_elapsed_seconds"])
        self.assertIsNone(row["engine_cancellation_acknowledged"])
        events = [json.loads(line) for line in (self.root / "ttft-request/client-events.jsonl").read_text().splitlines()]
        self.assertEqual([event["phase"] for event in events], ["request_started", "response_headers_received", "first_text", "request_context_exited"])
        self.assertTrue(all(event["opc_request_id"] == row["opc_request_id"] for event in events))

    async def test_ttft_normal_completion_retains_numeric_late_first_text(self):
        self.mode = "late-first-text"
        row = await self.ttft_request(horizon=.4, stream_policy="complete_bounded_response")
        self.assertTrue(row["complete"])
        self.assertTrue(row["first_text_late"])
        self.assertGreater(row["client_ttft_seconds"], .07)
        self.assertFalse(row["intentional_stream_abort"])

    async def test_ttft_generation_timeout_after_first_text_is_numeric_on_time(self):
        self.mode = "early-text-stalled-generation"
        row = await self.ttft_request(horizon=.1, stream_policy="complete_bounded_response")
        self.assertTrue(row["timed_out"])
        self.assertIsNotNone(row["client_ttft_seconds"])
        self.assertFalse(row["first_text_late"])
        self.assertFalse(row["intentional_stream_abort"])
        self.assertFalse(row["complete"])
        self.assertIsNone(row["completion_late"])

    async def test_pending_first_text_deadline_censors_without_inventing_numeric_ttft(self):
        self.mode = "late-first-text"
        row = await self.ttft_request(horizon=.4, stream_policy="complete_bounded_response",
                                      deadline_policy="cancel_if_no_first_text")
        self.assertTrue(row["timed_out"])
        self.assertTrue(row["first_text_late"])
        self.assertIsNone(row["client_ttft_seconds"])
        self.assertEqual(row["cancellation_reason"], "ttft_deadline")
        self.assertEqual(row["first_text_observation_horizon_seconds"], .07)
        self.assertEqual(row["response_completion_horizon_seconds"], .4)
        self.assertGreaterEqual(row["e2e_seconds"], .07)
        self.assertLess(row["e2e_seconds"], .4)
        events = [json.loads(line) for line in (self.root / "ttft-request/client-events.jsonl").read_text().splitlines()]
        cancelled = [e for e in events if e["phase"] == "cancellation_delivered"]
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0]["cancellation_reason"], "ttft_deadline")
        self.assertLessEqual(row["cancellation_delivered_elapsed_seconds"], row["response_context_exited_elapsed_seconds"])

    async def test_pending_text_timer_is_disarmed_and_response_can_finish_after_deadline(self):
        self.mode = "early-text-stalled-generation"
        row = await self.ttft_request(horizon=.4, stream_policy="complete_bounded_response",
                                      deadline_policy="cancel_if_no_first_text")
        self.assertTrue(row["complete"])
        self.assertFalse(row["timed_out"])
        self.assertFalse(row["first_text_late"])
        self.assertLess(row["client_ttft_seconds"], .07)
        self.assertGreater(row["e2e_seconds"], .07)
        self.assertIsNone(row["cancellation_reason"])

    async def test_response_horizon_after_timely_text_is_not_a_ttft_deadline_cancel(self):
        self.mode = "early-text-stalled-generation"
        row = await self.ttft_request(horizon=.1, stream_policy="complete_bounded_response",
                                      deadline_policy="cancel_if_no_first_text")
        self.assertTrue(row["timed_out"])
        self.assertFalse(row["first_text_late"])
        self.assertEqual(row["cancellation_reason"], "response_completion_horizon")
        self.assertIsNotNone(row["client_ttft_seconds"])
        self.assertGreaterEqual(row["e2e_seconds"], .1)

    async def test_pending_text_deadline_includes_time_waiting_for_headers(self):
        self.mode = "stalled-headers"
        row = await self.ttft_request(horizon=.4, stream_policy="complete_bounded_response",
                                      deadline_policy="cancel_if_no_first_text")
        self.assertEqual(row["status"], 0)
        self.assertTrue(row["timed_out"])
        self.assertIsNone(row["client_ttft_seconds"])
        self.assertEqual(row["cancellation_reason"], "ttft_deadline")

    async def test_delayed_timer_callback_never_turns_observed_late_text_into_timely_text(self):
        timer = mock.Mock()
        timer.expired.return_value = False
        row = {}
        p.observe_first_text(row, 300.25, 300, timer)
        self.assertEqual(row["client_ttft_seconds"], 300.25)
        self.assertGreaterEqual(row["client_ttft_seconds"], 300)
        self.assertEqual(row["ttft_deadline_observation_lag_seconds"], .25)
        self.assertEqual(row["ttft_deadline_policy_deviation"], "first_text_observed_after_planned_deadline")
        timer.reschedule.assert_called_once_with(None)

    async def test_ttft_horizon_observes_numeric_late_first_text(self):
        self.mode = "late-first-text"
        row = await self.ttft_request()
        self.assertGreater(row["client_ttft_seconds"], .07)
        self.assertTrue(row["first_text_late"])
        self.assertTrue(row["intentional_stream_abort"])
        self.assertFalse(row["timed_out"])
        self.assertIsNone(row["completion_late"])

    async def test_ttft_horizon_censoring_has_binary_late_but_no_numeric_ttft(self):
        self.mode = "slow"
        row = await self.ttft_request(horizon=.1)
        self.assertTrue(row["timed_out"])
        self.assertTrue(row["first_text_late"])
        self.assertIsNone(row["client_ttft_seconds"])
        self.assertIsNone(row["completion_late"])
        self.assertFalse(row["intentional_stream_abort"])
        self.assertEqual(row["cancellation_reason"], "observation_horizon")
        self.assertGreaterEqual(row["cancellation_delivered_elapsed_seconds"], row["observation_deadline_elapsed_seconds"])
        self.assertLessEqual(row["cancellation_delivered_elapsed_seconds"], row["response_context_exited_elapsed_seconds"])
        self.assertIsNone(row["engine_cancellation_acknowledged"])
        events = [json.loads(line) for line in (self.root / "ttft-request/client-events.jsonl").read_text().splitlines()]
        self.assertEqual(sum(event["phase"] == "cancellation_delivered" for event in events), 1)
        self.assertNotIn("first_text", [event["phase"] for event in events])

    async def test_short_observation_error_does_not_become_ttft_late(self):
        self.mode = "stalled-http-error"
        row = await self.ttft_request(horizon=.1)
        self.assertEqual(row["status"], 500)
        self.assertIsNone(row["first_text_late"])
        self.assertIsNone(row["client_ttft_seconds"])

    async def test_readiness_timeout_prevents_main_pressure(self):
        profile = dict(smoke_profile(), measurement_target="ttft", observation_horizon_seconds=.3,
                       readiness=dict(before_phase="pressure", timeout_seconds=.03))
        output = self.root / "readiness-timeout"
        output.mkdir()
        events = self.root / "router-events.jsonl"
        events.touch()
        args = argparse.Namespace(output=output, model="mock", url=self.url, run_id="readiness", router_events=events)
        workload = p.Run(args, profile)
        with self.assertRaisesRegex(RuntimeError, "Readiness failed"):
            await workload.exercise(None, p.make_plan(profile))
        self.assertEqual(self.posts, 0)
        self.assertFalse(json.loads((output / "readiness.json").read_text())["ready"])
        workload.stream.close()
        workload.metrics_stream.close()
        workload.client_events.close()

    async def test_sequential_seed_proves_idle_and_never_overlaps_requests(self):
        self.mode = "normal"
        profile = smoke_profile()
        profile["phases"] = [dict(name="seed", seconds=.1, rps=40, arrivals="periodic", sequential_seed=True)]
        result = await self.exercise(profile)
        self.assertEqual(result["execution_status"], "PASS")
        self.assertEqual(self.peak, 1)
        stages = json.loads((self.root / "run/stages.json").read_text())
        self.assertEqual(sum(s["name"] == "sequential_seed_idle_verified" for s in stages), len(self.rows()))

    async def test_busy_start_verifies_actual_busy_metrics_without_drain(self):
        profile = dict(smoke_profile(), measurement_target="ttft", observation_horizon_seconds=.3,
                       readiness=dict(before_phase="pressure", timeout_seconds=.05), busy_start_timeout_seconds=.1,
                       startup_scenario="busy_start")
        output = self.root / "busy-start"
        output.mkdir()
        events = self.root / "router-events.jsonl"
        events.touch()
        args = argparse.Namespace(output=output, run_id="busy", router_events=events)
        workload = p.Run(args, profile)
        async def samples():
            for _ in range(30):
                now = p.time.monotonic() - workload.started
                workload.latest = dict(elapsed_seconds=now, sample_completed_elapsed_seconds=now,
                    native_running=2, native_waiting=3, calibration_baseline_provisional=1,
                    calibration_baseline_measured=0, calibration_sample_count=0, calibration_fitted_revision=0)
                await asyncio.sleep(.002)
        sampler = asyncio.create_task(samples())
        await workload.verify_busy_start()
        self.assertTrue(workload.initial_native_busy)
        self.assertFalse(workload.initial_native_idle)
        self.assertEqual(self.posts, 0)
        await sampler
        workload.stream.close()
        workload.metrics_stream.close()
        workload.client_events.close()

    async def test_fresh_http_with_stale_internal_snapshot_cannot_pass_readiness(self):
        profile = dict(smoke_profile(), measurement_target="ttft", observation_horizon_seconds=.3,
                       readiness=dict(before_phase="pressure", timeout_seconds=.03))
        output = self.root / "stale-snapshot"
        output.mkdir()
        events = self.root / "stale-events.jsonl"
        events.touch()
        args = argparse.Namespace(output=output, run_id="stale", router_events=events)
        workload = p.Run(args, profile)
        workload.initial_native_idle = True
        workload.readiness.snapshot = lambda *unused: dict(ready=True, workers=[dict(epoch=1)], errors=[])
        async def samples():
            for _ in range(30):
                now = p.time.monotonic() - workload.started
                workload.latest = dict(elapsed_seconds=now, sample_completed_elapsed_seconds=now,
                    calibration_native_capacity_ready=1, calibration_sample_count=32, calibration_epoch=1,
                    data_usable=0)
                await asyncio.sleep(.002)
        sampler = asyncio.create_task(samples())
        with self.assertRaisesRegex(RuntimeError, "Readiness failed"):
            await workload.require_readiness()
        await sampler
        proof = json.loads((output / "readiness.json").read_text())
        self.assertFalse(proof["ready"])
        self.assertFalse(proof["fresh_native_metrics"])
        self.assertEqual(self.posts, 0)
        workload.stream.close()
        workload.metrics_stream.close()
        workload.client_events.close()

    async def test_native_stats_gate_blocks_unchanged_counters_then_accepts_completions(self):
        profile = dict(smoke_profile(), measurement_target="ttft", observation_horizon_seconds=.3,
            readiness=dict(before_phase="pressure", timeout_seconds=.02, require_native_completion_stats=True))
        output = self.root / "native-stats-gate"
        output.mkdir()
        events = self.root / "native-stats-events.jsonl"
        events.touch()
        workload = p.Run(argparse.Namespace(output=output, run_id="stats", router_events=events, engine="vllm"), profile)
        workload.initial_native_idle = True
        workload.readiness.snapshot = lambda *unused: dict(ready=True, workers=[dict(epoch=1)], errors=[])
        counters = ["request_prompt_tokens_count", "request_prompt_tokens_sum", "request_prefill_kv_computed_tokens_count",
                    "request_prefill_kv_computed_tokens_sum", "e2e_request_latency_seconds_count"]
        workload.initial_engine_prometheus = "".join("vllm:" + n + " 100\n" for n in counters)
        workload.rows = [dict(complete=True) for _ in range(4)]
        workload.latest = dict(sample_completed_elapsed_seconds=0, data_usable=1,
            calibration_native_capacity_ready=1, calibration_sample_count=4, calibration_epoch=1,
            engine_prometheus=workload.initial_engine_prometheus)
        try:
            with self.assertRaisesRegex(RuntimeError, "Readiness failed"):
                await workload.require_readiness()
            proof = json.loads((output / "readiness.json").read_text())
            self.assertFalse(proof["native_completion_stats"]["ok"])
            self.assertEqual(self.posts, 0)
            workload.latest.update(engine_prometheus="".join("vllm:" + n + " 200\n" for n in counters),
                sample_completed_elapsed_seconds=p.time.monotonic()-workload.started)
            await workload.require_readiness()
            self.assertTrue(json.loads((output / "readiness.json").read_text())["ready"])
        finally:
            workload.stream.close()
            workload.metrics_stream.close()
            workload.client_events.close()

    async def test_live_event_readiness_passes_only_after_current_fit_and_fresh_metrics(self):
        profile = dict(smoke_profile(), measurement_target="ttft", observation_horizon_seconds=.3,
                       readiness=dict(before_phase="pressure", timeout_seconds=.1))
        output = self.root / "ready"
        output.mkdir()
        events = self.root / "ready-events.jsonl"
        base = dict(worker="http://engine", calibration_method="mse_poll_two_wave_queue_v2", calibration_epoch=1,
                    calibration_state="capacity_ready")
        initial = dict(base, event="calibration_forecast", request_id="prod-ready-0", native_ready=True,
            calibration_observation_sequence=0, calibration_fit_sequence=0, calibration_history_revision=0,
            calibration_fitted_revision=0, calibration_sample_count=0, calibration_baseline_available=True,
            calibration_baseline_source="provisional", calibration_baseline_seconds=.2)
        log = [initial] + [dict(base, event="calibration_feedback", request_id=f"prod-ready-{i}", reason="accepted") for i in range(4)]
        log += [dict(base, event="calibration_update", sample_count_after=4, fit_sequence_after=4, calibration_fitted_revision=4),
                dict(initial, request_id="prod-ready-4", calibration_observation_sequence=4, calibration_fit_sequence=4,
                     calibration_history_revision=4, calibration_fitted_revision=4, calibration_sample_count=4)]
        events.write_text("".join(json.dumps(e) + "\n" for e in log))
        args = argparse.Namespace(output=output, run_id="ready", router_events=events)
        workload = p.Run(args, profile)
        workload.initial_native_idle = True
        workload.latest = dict(elapsed_seconds=0, sample_completed_elapsed_seconds=0, calibration_epoch=1,
                              calibration_native_capacity_ready=1, calibration_sample_count=4, data_usable=1)
        await workload.require_readiness()
        self.assertTrue(json.loads((output / "readiness.json").read_text())["ready"])
        self.assertEqual(workload.stages[-1]["name"], "readiness_verified")
        workload.stream.close()
        workload.metrics_stream.close()
        workload.client_events.close()

    async def test_stalled_http_error_is_not_model_timeout_truth(self):
        self.mode = "stalled-http-error"
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            await self.exercise()
        rows = self.rows()
        self.assertTrue(rows)
        self.assertTrue(all(r["status"] == 500 and r["timed_out"] for r in rows))
        self.assertTrue(all(r["error"] == "HTTP 500" for r in rows))
        self.assertTrue(all(r["completion_late"] is None and r["first_text_late"] is None for r in rows))
        summary = json.loads((self.root / "run/summary.json").read_text())
        self.assertEqual(summary["execution_status"], "INVALID")
        self.assertFalse(summary["coverage_checks"]["pressure_deadline_misses"])

    async def test_exact_forecasts_distinguish_ttft_from_full_completion(self):
        await self.exercise()
        rows = self.rows()
        events = []
        selected_changed = False
        for row in rows:
            events.append(dict(event="forecast", request_id=row["opc_request_id"], threshold_seconds=.07,
                               ready=True, would_reject=row["controlled_rejection"]))
            if row["controlled_rejection"]:
                continue
            changed = row["complete"] and not selected_changed
            selected_changed |= changed
            events.append(dict(event="calibration_forecast", request_id=row["opc_request_id"],
                               baseline_seconds=.001, predicted_ttft_seconds=.1 if changed else .005,
                               calibrated_ttft_seconds=.11 if changed else .1, threshold_seconds=.07,
                               would_reject=changed, calibrated_would_reject=True))
            if row["client_ttft_seconds"] is not None:
                events.append(dict(event="first_text", request_id=row["opc_request_id"], ttft_seconds=.005))
        log = self.root / "events.jsonl"
        log.write_text("\n".join(json.dumps(event) for event in events) + "\n")
        result = p.analyze(self.root / "run", log)
        scores = result["scores"]["pressure"]
        after_text = sum(r["timed_out"] and r["client_ttft_seconds"] is not None for r in rows)
        self.assertGreater(after_text, 0)
        self.assertEqual(scores["raw"]["gate_vs_completion_deadline"]["fn"] - scores["raw"]["total_ttft_prediction"]["fn"], after_text)
        self.assertEqual(scores["calibrated"]["gate_vs_completion_deadline"]["fn"], 0)
        self.assertEqual(scores["raw"]["gate_vs_completion_deadline"]["fp"], 0)
        self.assertEqual(scores["raw"]["total_ttft_prediction"]["fp"], 1)
        self.assertTrue(result["coverage_checks"]["exact_request_forecasts"])
        paired = [json.loads(line) for line in (self.root / "run/paired-production.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["smg_ttft_seconds"] is None for r in paired if r["client_ttft_seconds"] is None))
        log.write_text(log.read_text() + json.dumps(events[0]) + "\n")
        with self.assertRaisesRegex(AssertionError, "Duplicate exact"):
            p.analyze(self.root / "run", log)

    async def test_wrong_router_budget_stops_before_inference(self):
        self.router_budget = 5
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            await self.exercise()
        self.assertEqual(self.posts, 0)

    async def test_local_saturation_explicitly_invalidates_load_test(self):
        self.mode = "slow"
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            await self.exercise(dict(smoke_profile(), max_inflight=1))
        rows = self.rows()
        self.assertTrue(any(r.get("local_drop") for r in rows))
        self.assertTrue(all(r["completion_late"] is None for r in rows if r.get("local_drop")))
        summary = json.loads((self.root / "run/summary.json").read_text())
        self.assertFalse(summary["execution_checks"]["no_local_saturation"])

    async def test_tampered_prompt_rejected_before_any_http(self):
        p.write_plan(self.root / "plan", smoke_profile(), smoke=True)
        plan = self.root / "plan/plan.jsonl"
        rows = [json.loads(line) for line in plan.read_text().splitlines()]
        rows[0]["prompt_sha256"] = "0" * 64
        plan.write_text("\n".join(p.encoded(row).decode() for row in rows) + "\n")
        manifest_path = self.root / "plan/plan-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["plan_sha256"] = p.hashlib.sha256(plan.read_bytes()).hexdigest()
        p.save(manifest_path, manifest)
        args = argparse.Namespace(plan=self.root / "plan", output=self.root / "run", url=self.url,
                                  model="mock", engine="vllm", engine_metrics_url=self.url + "/metrics",
                                  router_metrics_url=self.url + "/metrics", router_events=None, run_id="smoke")
        with self.assertRaisesRegex(AssertionError, "Prompt differs"):
            await p.run(args)
        self.assertEqual(self.posts, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
