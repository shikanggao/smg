#!/usr/bin/env python3
"""Plan/replay mixed inference traffic with real client deadlines. No deployment.

Arrivals are independent of response completion. Local saturation is recorded
as a load-generator failure, never silently converted into a client-side queue.
Input sizes are words; output sizes are limits. Actual tokens come from usage.
"""
from __future__ import annotations

import argparse
import asyncio
import codecs
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import resource
import shutil
import time


HERE = Path(__file__).resolve().parent
TASKS = {
    "short-chat": "Answer briefly: which project has the latest delivery and what action should its owner take?",
    "rag": "Use the reference records to answer which projects need attention. Cite record identifiers and explain the supporting facts.",
    "summarization": "Summarize the reference records into an executive summary, risks, decisions, and next actions. Preserve the key numbers.",
    "long-generation": "Write a detailed tutorial for planning a fictional software delivery. Include worked examples, tradeoffs, a checklist, and an explanation of every step.",
    "code": "Write Python code to validate and aggregate these synthetic project records. Explain the algorithm, edge cases, complexity, and useful tests.",
    "long-context-generation": "Write a detailed review of these project records. Compare the projects, explain each risk, and propose a sequenced action plan with examples.",
}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def quantile(values, q):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    return values[min(len(values) - 1, math.ceil(q * len(values)) - 1)] if values else None


def observe_first_text(row, elapsed, deadline_seconds, pending_timeout=None):
    """Preserve the actual observation even if the event loop ran a timer late."""
    row["client_ttft_seconds"] = elapsed
    if pending_timeout is not None:
        if elapsed >= deadline_seconds:
            row["ttft_deadline_policy_deviation"] = "first_text_observed_after_planned_deadline"
            row["ttft_deadline_observation_lag_seconds"] = elapsed - deadline_seconds
        if not pending_timeout.expired():
            pending_timeout.reschedule(None)


def validate(profile, smoke=False):
    assert profile["schema_version"] == 1
    assert profile["deadline_seconds"] > 0
    assert math.isfinite(profile["deadline_seconds"])
    assert math.isfinite(profile["router_extra_ttft_budget_seconds"]) and profile["router_extra_ttft_budget_seconds"] > 0
    assert smoke or profile["deadline_seconds"] == 300, "Production coverage requires a 300-second deadline"
    if "serving_auth_env" in profile:
        assert profile["serving_auth_env"] == "SMG_EXPERIMENT_INGRESS_KEY", "Unexpected serving credential environment name"
    if "recovery_candidate_required" in profile:
        assert type(profile["recovery_candidate_required"]) is bool
    if "feature_context_required" in profile:
        assert type(profile["feature_context_required"]) is bool
    target = profile.get("measurement_target", "completion")
    assert target in {"completion", "ttft"}
    if target == "ttft":
        assert profile.get("startup_scenario", "isolated") in {"isolated", "busy_start"}
        if profile.get("startup_scenario") == "busy_start":
            assert 0 < profile["busy_start_timeout_seconds"] <= 60
            assert not profile["phases"][0].get("drain_before"), "Busy startup must not silently force a quiet drain"
        horizon = profile["observation_horizon_seconds"]
        assert math.isfinite(horizon) and profile["deadline_seconds"] < horizon <= (1 if smoke else 600)
        assert profile.get("ttft_stream_policy", "abort_after_first_text") in {"abort_after_first_text", "complete_bounded_response"}
        assert profile.get("ttft_deadline_policy", "observe") in {"observe", "cancel_if_no_first_text"}
        if profile.get("ttft_deadline_policy") == "cancel_if_no_first_text":
            assert profile.get("ttft_stream_policy") == "complete_bounded_response", "Pending-text deadlines must preserve bounded normal completion"
        readiness = profile["readiness"]
        if "require_entry_epoch_continuity" in readiness:
            assert type(readiness["require_entry_epoch_continuity"]) is bool
        assert readiness["timeout_seconds"] > 0 and readiness["timeout_seconds"] <= 300
        assert readiness["before_phase"] in {p["name"] for p in profile["phases"]}
        assert profile["phases"][0]["name"] != readiness["before_phase"], "Readiness needs a bounded learning phase"
        assert max(c["max_output_tokens"][1] for c in profile["cohorts"]) <= 128, "TTFT observation must bound unnecessary generation"
    for key in ["max_requests", "max_inflight", "prefix_sessions"]:
        assert type(profile[key]) is int and profile[key] > 0
    assert 0 <= profile["shared_prefix_fraction"] <= 1
    assert profile["sample_interval_seconds"] > 0 and profile["drain_timeout_seconds"] > 0
    assert profile["maximum_p99_dispatch_lag_seconds"] > 0
    capture = profile.get("capture_content_chars", 0)
    assert type(capture) is int and 0 <= capture <= 16384
    names = [c["name"] for c in profile["cohorts"]]
    assert names and len(names) == len(set(names)) and set(names) <= TASKS.keys()
    for cohort in profile["cohorts"]:
        assert cohort["weight"] > 0
        for key in ["input_words", "max_output_tokens"]:
            lo, hi = cohort[key]
            assert type(lo) is int and type(hi) is int and 0 < lo <= hi
        assert cohort["input_words"][0] >= 64
    phases = profile["phases"]
    assert phases and len({p["name"] for p in phases}) == len(phases)
    for phase in phases:
        assert phase["seconds"] > 0 and phase["rps"] > 0
        assert math.isfinite(phase["seconds"]) and math.isfinite(phase["rps"])
        assert phase["arrivals"] in ["periodic", "poisson"]
        for key in ["input_words", "max_output_tokens"]:
            if key in phase:
                lo, hi = phase[key]
                assert type(lo) is int and type(hi) is int and 0 < lo <= hi
                assert key != "input_words" or lo >= 64
                assert target != "ttft" or key != "max_output_tokens" or hi <= 128


def messages(item, profile):
    """Stable shared prefix precedes the unique task; request IDs stay in headers."""
    n = item["input_words"]
    session = item["session"]
    namespace = profile.get("cache_namespace")
    if namespace is not None:
        assert isinstance(namespace, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", namespace)
    if item["shared_prefix"]:
        prefix = f"Synthetic company reference for session {session}. Use only these fictional records. "
        prefix += (f"Project cedar has owner Morgan status review budget 1200 delivery week {session + 2}. " * 40)
        if namespace:
            prefix = f"Experiment {namespace}. " + prefix
        prefix_words = prefix.split()[:min(256, n // 3)]
    else:
        prefix_words = (f"Experiment {namespace}. " if namespace else "").split()
        prefix_words += f"Unique synthetic scenario {profile['seed']} request {item['index']}.".split()
    remaining = n - len(prefix_words)
    task = TASKS[item["cohort"]].split()
    task += f"Scenario {item['index']}. These are fictional data for an inference benchmark.".split()
    records = []
    k = 0
    while len(task) + len(records) < remaining:
        code = (profile["seed"] + item["index"] * 997 + k * 37) % 100003
        records.extend(f"Record {code} project cedar owner Taylor budget {100 + code % 5000} delivery week {1 + k % 12} status review.".split())
        k += 1
    return [dict(role="system", content=" ".join(prefix_words)),
            dict(role="user", content=" ".join((task + records)[:remaining]))]


def make_plan(profile):
    rng = random.Random(profile["seed"])
    cohorts = profile["cohorts"]
    rows = []
    for phase in profile["phases"]:
        elapsed, index = 0.0, 0
        while True:
            elapsed += (1 / phase["rps"] if phase["arrivals"] == "periodic" else rng.expovariate(phase["rps"]))
            if elapsed >= phase["seconds"]:
                break
            cohort = (cohorts[index % len(cohorts)] if phase.get("balanced_controls")
                      else rng.choices(cohorts, [c["weight"] for c in cohorts])[0])
            item = dict(index=len(rows), phase=phase["name"], scheduled_phase_seconds=elapsed,
                        cohort=cohort["name"], control=bool(phase.get("balanced_controls")),
                        input_words=rng.randint(*phase.get("input_words", cohort["input_words"])),
                        max_output_tokens=rng.randint(*phase.get("max_output_tokens", cohort["max_output_tokens"])),
                        shared_prefix=rng.random() < profile["shared_prefix_fraction"],
                        session=rng.randrange(profile["prefix_sessions"]))
            item["prompt_sha256"] = digest(messages(item, profile))
            rows.append(item)
            assert len(rows) <= profile["max_requests"], "Plan exceeds the request safety bound; no requests sent"
            index += 1
    return rows


def write_plan(output, profile, smoke=False):
    validate(profile, smoke)
    rows = make_plan(profile)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "profile.json", profile)
    with (output / "plan.jsonl").open("w") as stream:
        for row in rows:
            stream.write(encoded(row).decode() + "\n")
    manifest = dict(schema_version=1, synthetic=True, smoke=smoke, profile_sha256=digest(profile),
                    plan_sha256=hashlib.sha256((output / "plan.jsonl").read_bytes()).hexdigest(),
                    planner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    requests=len(rows), phases=dict(Counter(r["phase"] for r in rows)),
                    cohorts=dict(Counter(r["cohort"] for r in rows)))
    save(output / "plan-manifest.json", manifest)
    return manifest


class Frames:
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.pending = ""
        self.data = []

    def feed(self, chunk, final=False):
        self.pending += self.decoder.decode(chunk, final=final)
        lines = self.pending.split("\n")
        self.pending = lines.pop()
        if final:
            lines += [self.pending, ""]
            self.pending = ""
        result = []
        for line in lines:
            line = line.removesuffix("\r")
            if not line:
                if self.data:
                    result.append("\n".join(self.data))
                    self.data = []
            elif line.startswith("data:"):
                self.data.append(line[5:].removeprefix(" "))
        return result


def metric(text, names, maximum=False):
    for name in names:
        values = []
        pattern = re.compile(r"^" + re.escape(name) + r"(?:\{.*\})?\s+([^\s]+)(?:\s+.*)?$")
        for line in text.splitlines():
            match = pattern.match(line)
            if match:
                value = float(match[1])
                if math.isfinite(value) and value >= 0:
                    values.append(value)
        if values:
            return max(values) if maximum else sum(values)
    return None


def native_metrics(text, engine):
    if engine == "vllm":
        names = [["vllm:num_requests_waiting"], ["vllm:num_requests_running"],
                 ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"]]
    else:
        names = [["sglang:num_queue_reqs", "sglang_num_queue_reqs"],
                 ["sglang:num_running_reqs", "sglang_num_running_reqs"],
                 ["sglang:token_usage", "sglang_token_usage"]]
    result = dict(zip(["native_waiting", "native_running", "native_kv"],
                      [metric(text, n, i == 2) for i, n in enumerate(names)]))
    # Aggregate counters show progress, not per-request cancellation acknowledgement.
    # Missing metrics remain None; full original Prometheus is retained alongside them.
    counters = {
        "native_completed": ["vllm:request_success_total", "vllm:e2e_request_latency_seconds_count"] if engine == "vllm" else ["sglang:e2e_request_latency_seconds_count", "sglang_e2e_request_latency_seconds_count"],
        "native_prompt_tokens": ["vllm:prompt_tokens_total"] if engine == "vllm" else ["sglang:prompt_tokens_total", "sglang_prompt_tokens_total"],
        "native_generation_tokens": ["vllm:generation_tokens_total"] if engine == "vllm" else ["sglang:generation_tokens_total", "sglang_generation_tokens_total"],
        "native_first_tokens": ["vllm:time_to_first_token_seconds_count"] if engine == "vllm" else ["sglang:time_to_first_token_seconds_count", "sglang_time_to_first_token_seconds_count"],
        "native_preemptions": ["vllm:num_preemptions_total"] if engine == "vllm" else ["sglang:num_retracted_reqs_total", "sglang_num_retracted_reqs_total"],
        # Live aggregate progress/context, never request-specific cancellation ACKs.
        "native_iteration_tokens": ["vllm:iteration_tokens_total_sum"] if engine == "vllm" else [],
        "native_aborted": [] if engine == "vllm" else ["sglang:num_aborted_requests_total", "sglang_num_aborted_requests_total"],
        "native_generation_throughput": [] if engine == "vllm" else ["sglang:gen_throughput", "sglang_gen_throughput"],
    }
    result.update({name: metric(text, aliases) for name, aliases in counters.items()})
    return result


class Run:
    def __init__(self, args, profile):
        self.args, self.profile = args, profile
        self.serving_key = None
        if profile.get("serving_auth_env"):
            self.serving_key = os.environ.get(profile["serving_auth_env"])
            if not self.serving_key or any(c.isspace() for c in self.serving_key):
                raise RuntimeError("Required owned serving credential is unavailable")
        self.started = time.monotonic()
        self.active = set()
        self.phase = "startup"
        self.latest = {}
        self.prompts = {}
        self.rows, self.samples, self.stages = [], [], []
        self.stop = asyncio.Event()
        self.stream = (args.output / "requests.jsonl").open("w")
        self.metrics_stream = (args.output / "metrics.jsonl").open("w")
        self.client_events = (args.output / "client-events.jsonl").open("w")
        self.event_offset, self.event_pending = 0, b""
        self.initial_native_idle = False
        self.initial_native_busy = False
        self.initial_engine_prometheus = None
        self.readiness = None
        if profile.get("measurement_target") == "ttft":
            try:
                from .wait_ttft_readiness import Readiness
            except ImportError:
                try:
                    from wait_ttft_readiness import Readiness
                except ModuleNotFoundError:
                    from scripts.wait_ttft_readiness import Readiness
            assert args.router_events is not None, "TTFT readiness requires the live owned-router event stream"
            self.readiness = Readiness(f"prod-{args.run_id}-", profile.get("startup_scenario", "isolated"))

    def record(self, row):
        self.rows.append(row)
        self.stream.write(encoded(row).decode() + "\n")
        self.stream.flush()

    def client_event(self, row, phase):
        event = dict(event="client_transport", phase=phase, opc_request_id=row["opc_request_id"],
                     elapsed_seconds=time.monotonic() - self.started,
                     request_elapsed_seconds=time.monotonic() - self.started - row["started_elapsed_seconds"],
                     cancellation_reason=row.get("cancellation_reason"),
                     engine_cancellation_acknowledged=None)
        self.client_events.write(encoded(event).decode() + "\n")
        self.client_events.flush()
        return event["elapsed_seconds"]

    @asynccontextmanager
    async def cancellation_observer(self, row, pending_deadline=None, response_deadline=None):
        # The inner observer runs before aiohttp releases the response context;
        # the outer observer covers cancellation while awaiting response headers.
        try:
            yield
        except asyncio.CancelledError:
            if row.get("cancellation_delivered_elapsed_seconds") is None:
                if pending_deadline is not None and pending_deadline.expired():
                    row["cancellation_reason"] = "ttft_deadline"
                elif response_deadline is not None and response_deadline.expired():
                    row["cancellation_reason"] = ("response_completion_horizon" if
                        row.get("ttft_deadline_policy") == "cancel_if_no_first_text"
                        and row.get("client_ttft_seconds") is not None else "observation_horizon")
                else:
                    row["cancellation_reason"] = "workload_interrupted"
                row["cancellation_delivered_elapsed_seconds"] = self.client_event(row, "cancellation_delivered")
                planned = (row["ttft_deadline_elapsed_seconds"] if row["cancellation_reason"] == "ttft_deadline"
                           else row["observation_deadline_elapsed_seconds"] if row["cancellation_reason"] in
                           {"observation_horizon", "response_completion_horizon"} else None)
                if planned is not None:
                    row["cancellation_deadline_lag_seconds"] = row["cancellation_delivered_elapsed_seconds"] - planned
            raise

    async def sample(self, session):
        import aiohttp
        while not self.stop.is_set():
            row = dict(elapsed_seconds=time.monotonic() - self.started, phase=self.phase,
                       client_active=len(self.active))
            for kind, url in [("engine", self.args.engine_metrics_url), ("router", self.args.router_metrics_url)]:
                if not url:
                    continue
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        response.raise_for_status()
                        text = await response.text()
                    row[kind + "_prometheus"] = text
                    if kind == "engine":
                        if self.initial_engine_prometheus is None:
                            self.initial_engine_prometheus = text
                        row.update(native_metrics(text, self.args.engine))
                    else:
                        for key, name in [("raw_ttft", "smg_estimated_wait_seconds"),
                                          ("calibrated_ttft", "smg_estimated_wait_calibrated_seconds"),
                                          ("scale", "smg_estimated_wait_calibration_scale"),
                                          ("capacity", "smg_estimated_wait_effective_prefill_capacity"),
                                          ("baseline", "smg_estimated_wait_baseline_seconds"),
                                          ("data_usable", "smg_estimated_wait_data_usable"),
                                          ("delay_floor", "smg_estimated_wait_observed_delay_floor_seconds")]:
                            row[key] = metric(text, [name], maximum=True)
                        for key in ["baseline_seconds", "baseline_provisional", "baseline_measured", "epoch",
                                    "observation_sequence", "history_revision", "fitted_revision", "sample_count", "native_capacity_ready", "input_excess_seconds"]:
                            row["calibration_" + key] = metric(text, ["smg_estimated_wait_calibration_" + key], maximum=True)
                except Exception as error:
                    row[kind + "_error"] = type(error).__name__ + ": " + str(error)
            row["sample_completed_elapsed_seconds"] = time.monotonic() - self.started
            self.latest = row
            self.samples.append({k: v for k, v in row.items() if not k.endswith("_prometheus")})
            self.metrics_stream.write(encoded(row).decode() + "\n")
            self.metrics_stream.flush()
            try:
                await asyncio.wait_for(self.stop.wait(), self.profile["sample_interval_seconds"])
            except asyncio.TimeoutError:
                pass

    async def request(self, session, item, scheduled):
        import aiohttp
        prompt = self.prompts.get(item["index"]) or messages(item, self.profile)
        assert digest(prompt) == item["prompt_sha256"], "Prompt differs from frozen plan"
        start = time.monotonic()
        rid = f"prod-{self.args.run_id}-{item['index']}"
        row = dict(item, opc_request_id=rid, started_elapsed_seconds=start - self.started,
                   scheduled_elapsed_seconds=scheduled - self.started,
                   dispatch_lag_seconds=max(0, start - scheduled), status=0,
                   client_ttft_seconds=None, first_answer_seconds=None, first_sse_seconds=None,
                   done_seen=False, finish_reason=None, complete=False, timed_out=False,
                   cancelled=False, error=None, usage={},
                   cancellation_delivered_elapsed_seconds=None,
                   cancellation_reason=None, engine_cancellation_acknowledged=None,
                   response_error_code=None,
                   upstream_response_request_id=None, upstream_sse_response_id=None)
        ttft_only = self.profile.get("measurement_target") == "ttft"
        horizon = self.profile.get("observation_horizon_seconds", self.profile["deadline_seconds"])
        pending_only_deadline = (ttft_only and self.profile.get("ttft_deadline_policy") == "cancel_if_no_first_text")
        response_timeout = asyncio.timeout_at(start + horizon)
        pending_timeout = asyncio.timeout_at(start + self.profile["deadline_seconds"] if pending_only_deadline else None)
        if ttft_only:
            row.update(measurement_target="ttft", observation_horizon_seconds=horizon, intentional_stream_abort=False,
                       ttft_stream_policy=self.profile.get("ttft_stream_policy", "abort_after_first_text"),
                       ttft_deadline_policy=self.profile.get("ttft_deadline_policy", "observe"),
                       first_text_observation_horizon_seconds=self.profile["deadline_seconds"] if pending_only_deadline else horizon,
                       response_completion_horizon_seconds=horizon)
        row["ttft_deadline_elapsed_seconds"] = start - self.started + self.profile["deadline_seconds"]
        row["observation_deadline_elapsed_seconds"] = start - self.started + horizon
        self.client_event(row, "request_started")
        capture = self.profile.get("capture_content_chars", 0)
        if capture:
            row.update(content_preview="", content_characters=0)
        frames = Frames()
        def consume(data):
            elapsed = time.monotonic() - start
            if data == "[DONE]":
                row["done_seen"] = True
                return
            obj = json.loads(data)
            if row["upstream_sse_response_id"] is None and isinstance(obj.get("id"), str):
                row["upstream_sse_response_id"] = obj["id"][:256]
            if "error" in obj:
                raise RuntimeError("SSE error: " + json.dumps(obj["error"])[:300])
            if row["first_sse_seconds"] is None:
                row["first_sse_seconds"] = elapsed
            if obj.get("usage"):
                row["usage"] = obj["usage"]
            for choice in obj.get("choices", []):
                delta = choice.get("delta", {})
                answer = delta.get("content")
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if row["client_ttft_seconds"] is None and any(isinstance(v, str) and v for v in [answer, reasoning]):
                    observe_first_text(row, elapsed, self.profile["deadline_seconds"],
                                       pending_timeout if pending_only_deadline else None)
                    self.client_event(row, "first_text")
                if row["first_answer_seconds"] is None and isinstance(answer, str) and answer:
                    row["first_answer_seconds"] = elapsed
                if capture and isinstance(answer, str):
                    row["content_characters"] += len(answer)
                    row["content_preview"] += answer[:max(0, capture - len(row["content_preview"]))]
                if choice.get("finish_reason") is not None:
                    row["finish_reason"] = choice["finish_reason"]
        payload = dict(model=self.args.model, messages=prompt, max_tokens=item["max_output_tokens"],
                       stream=True, temperature=0, stream_options=dict(include_usage=True))
        headers = {"opc-request-id": rid, "x-request-id": rid}
        if self.serving_key:
            headers["Authorization"] = "Bearer " + self.serving_key
        try:
            # The pending-text timer is disarmed by first text. The independent
            # response horizon still bounds headers, streaming and generation.
            async with response_timeout, pending_timeout, self.cancellation_observer(row, pending_timeout, response_timeout):
                async with session.post(self.args.url.rstrip("/") + "/v1/chat/completions", json=payload,
                                        headers=headers, timeout=aiohttp.ClientTimeout(total=None)) as response:
                    row["status"] = response.status
                    row["upstream_response_request_id"] = (response.headers.get("x-request-id") or response.headers.get("opc-request-id") or "")[:256] or None
                    self.client_event(row, "response_headers_received")
                    async with self.cancellation_observer(row, pending_timeout, response_timeout):
                        if response.status != 200:
                            # Preserve error headers even if reading their body
                            # reaches the deadline. That is not model-wait truth.
                            row["error"] = "HTTP " + str(response.status)
                            body = await response.text()
                            row["error"] += ": " + body[:1000]
                            try:
                                error_body = json.loads(body)
                                error_body = error_body.get("error") if isinstance(error_body, dict) else None
                                code = error_body.get("code") if isinstance(error_body, dict) else None
                                row["response_error_code"] = code if isinstance(code, str) else None
                            except ValueError:
                                pass
                        else:
                            async for chunk in response.content.iter_any():
                                for data in frames.feed(chunk):
                                    consume(data)
                                if (ttft_only and row["ttft_stream_policy"] == "abort_after_first_text"
                                        and row["client_ttft_seconds"] is not None):
                                    row["intentional_stream_abort"] = not (row["done_seen"] and row["finish_reason"] is not None)
                                    if row["intentional_stream_abort"]:
                                        row["cancellation_reason"] = "intentional_first_text_abort"
                                        row["cancellation_delivered_elapsed_seconds"] = self.client_event(row, "intentional_response_release")
                                    break
                            if not row.get("intentional_stream_abort"):
                                for data in frames.feed(b"", final=True):
                                    consume(data)
                            row["complete"] = bool(row["done_seen"] and row["finish_reason"] is not None and row["client_ttft_seconds"] is not None)
                            if not row["complete"] and not row.get("intentional_stream_abort"):
                                row["error"] = "Incomplete SSE response"
        except TimeoutError:
            row["timed_out"] = True
            row["cancellation_reason"] = ("ttft_deadline" if pending_timeout.expired() else
                "response_completion_horizon" if pending_only_deadline and row["client_ttft_seconds"] is not None else "observation_horizon")
        except asyncio.CancelledError:
            row["cancelled"] = True
            row["cancellation_reason"] = "workload_interrupted"
            row["error"] = "Workload interrupted before its own deadline"
        except Exception as error:
            row["error"] = type(error).__name__ + ": " + str(error)
        row["response_context_exited_elapsed_seconds"] = self.client_event(row, "request_context_exited")
        row["completed_elapsed_seconds"] = time.monotonic() - self.started
        row["e2e_seconds"] = time.monotonic() - start
        row["controlled_rejection"] = (row["status"] == 503
            and row["response_error_code"] == "worker_overload_protection_shed"
            and "estimated wait budgets" in (row["error"] or ""))
        valid_timeout = row["timed_out"] and row["status"] in (0, 200) and row["error"] is None
        row["completion_late"] = (True if valid_timeout and not ttft_only else row["e2e_seconds"] >= self.profile["deadline_seconds"] if row["complete"] else None)
        row["first_text_late"] = (row["client_ttft_seconds"] >= self.profile["deadline_seconds"] if row["client_ttft_seconds"] is not None else True if valid_timeout and horizon >= self.profile["deadline_seconds"] else None)
        # A deadline without first text is censored, not an invented 300s TTFT.
        self.record(row)

    def read_readiness_events(self):
        path = self.args.router_events
        if not path.exists():
            return
        with path.open("rb") as stream:
            stream.seek(self.event_offset)
            data = stream.read()
            self.event_offset = stream.tell()
        lines = (self.event_pending + data).split(b"\n")
        self.event_pending = lines.pop()
        for line in lines:
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            self.readiness.observe(obj.get("fields", {}) | obj)

    async def require_readiness(self):
        until = time.monotonic() + self.profile["readiness"]["timeout_seconds"]
        while True:
            self.read_readiness_events()
            proof = self.readiness.snapshot(self.initial_native_idle, self.initial_native_busy)
            now = time.monotonic() - self.started
            sample = self.latest
            proof["fresh_native_metrics"] = (sample.get("sample_completed_elapsed_seconds", -1) >= now - 3 * self.profile["sample_interval_seconds"]
                and not sample.get("engine_error") and not sample.get("router_error")
                and sample.get("data_usable") == 1
                and sample.get("calibration_native_capacity_ready") == 1
                and sample.get("calibration_sample_count", 0) >= 4
                and any(w["epoch"] == sample.get("calibration_epoch") for w in proof["workers"]))
            proof["ready"] = proof["ready"] and proof["fresh_native_metrics"]
            if self.profile["readiness"].get("require_native_completion_stats"):
                try:
                    from .wait_ttft_readiness import native_completion_stats_proof
                except ImportError:
                    try:
                        from wait_ttft_readiness import native_completion_stats_proof
                    except ModuleNotFoundError:
                        from scripts.wait_ttft_readiness import native_completion_stats_proof
                stats = native_completion_stats_proof(self.initial_engine_prometheus or "",
                    sample.get("engine_prometheus", ""), self.args.engine)
                stats["completed_owned_responses"] = sum(r.get("complete") is True for r in self.rows)
                stats["minimum_completed_owned_responses"] = 4
                stats["ok"] = stats["ok"] and stats["completed_owned_responses"] >= 4
                proof["native_completion_stats"] = stats
                proof["ready"] = proof["ready"] and stats["ok"]
            proof["elapsed_seconds"] = now
            save(self.args.output / "readiness.json", proof)
            if proof["ready"]:
                self.stages.append(dict(name="readiness_verified", elapsed_seconds=now, proof=proof))
                return
            if proof["errors"] or time.monotonic() >= until:
                raise RuntimeError("Readiness failed before main pressure; preserve readiness.json")
            await asyncio.sleep(min(.25, self.profile["sample_interval_seconds"]))

    async def verify_busy_start(self):
        """Full-suite scenario: require actual busy startup and a fresh provisional state."""
        until = time.monotonic() + self.profile["busy_start_timeout_seconds"]
        newer_than = time.monotonic() - self.started
        previous, samples = None, []
        while time.monotonic() < until:
            row = self.latest
            stamp = row.get("sample_completed_elapsed_seconds")
            if stamp is not None and stamp != previous and row["elapsed_seconds"] >= newer_than:
                previous = stamp
                busy = (not row.get("engine_error") and not row.get("router_error")
                    and row.get("native_running") is not None and row.get("native_waiting") is not None
                    and row["native_running"] + row["native_waiting"] > 0
                    and row.get("calibration_baseline_provisional") == 1
                    and row.get("calibration_baseline_measured") == 0
                    and row.get("calibration_sample_count") == 0 and row.get("calibration_fitted_revision") == 0)
                samples = samples + [row] if busy else []
                if len(samples) >= 2:
                    self.initial_native_busy = True
                    save(self.args.output / "busy-start.json", dict(verified=True, samples=samples[-2:],
                         note="Busy engine and empty new-router calibration state; no quiet drain was required."))
                    self.stages.append(dict(name="busy_start_verified", elapsed_seconds=time.monotonic() - self.started))
                    return
            await asyncio.sleep(min(.1, self.profile["sample_interval_seconds"]))
        raise RuntimeError("Busy-start scenario did not prove busy native load with fresh provisional calibration")

    async def drained(self):
        self.phase = "drain"
        self.stages.append(dict(name="drain", elapsed_seconds=time.monotonic() - self.started))
        until = time.monotonic() + self.profile["drain_timeout_seconds"]
        newer_than = time.monotonic() - self.started
        previous, zeros = None, 0
        while time.monotonic() < until:
            row = self.latest
            stamp = row.get("sample_completed_elapsed_seconds")
            if stamp is not None and stamp != previous and row["elapsed_seconds"] >= newer_than:
                previous = stamp
                zeros = zeros + 1 if row.get("native_waiting") == 0 and row.get("native_running") == 0 and not row.get("engine_error") else 0
                if zeros >= 2:
                    return
            await asyncio.sleep(min(.1, self.profile["sample_interval_seconds"]))
        raise TimeoutError("Native engine did not prove drain with two fresh zero-load samples")

    async def exercise(self, session, rows):
        for phase in self.profile["phases"]:
            if self.readiness and phase["name"] == self.profile["readiness"]["before_phase"]:
                await self.require_readiness()
            if phase.get("drain_before"):
                if self.active:
                    await asyncio.gather(*list(self.active))
                await self.drained()
            self.phase = phase["name"]
            start = time.monotonic()
            self.stages.append(dict(name=self.phase, elapsed_seconds=start - self.started))
            for item in (r for r in rows if r["phase"] == self.phase):
                scheduled = start + item["scheduled_phase_seconds"]
                if phase.get("sequential_seed"):
                    if self.active:
                        await asyncio.gather(*list(self.active))
                    await self.drained()
                    self.phase = phase["name"]
                    self.stages.append(dict(name="sequential_seed_idle_verified", index=item["index"], elapsed_seconds=time.monotonic() - self.started))
                    scheduled = time.monotonic()
                await asyncio.sleep(max(0, scheduled - time.monotonic()))
                if len(self.active) >= self.profile["max_inflight"]:
                    self.record(dict(item, opc_request_id=f"prod-{self.args.run_id}-{item['index']}",
                                     local_drop=True, complete=False, timed_out=False,
                                     completion_late=None, first_text_late=None, status=0,
                                     started_elapsed_seconds=None, dispatch_lag_seconds=None,
                                     error="Load generator reached max_inflight; not an engine rejection"))
                    continue
                task = asyncio.create_task(self.request(session, item, scheduled))
                self.active.add(task)
                task.add_done_callback(self.active.discard)
                if phase.get("sequential_seed"):
                    await task
                    if not self.rows[-1].get("complete"):
                        raise RuntimeError("Sequential baseline seed failed; no further seed requests sent")
            await asyncio.sleep(max(0, start + phase["seconds"] - time.monotonic()))
        if self.active:
            await asyncio.gather(*list(self.active))
        await self.drained()


def fd_check(inflight):
    wanted = 2 * inflight + 128
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and soft < wanted:
        resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]


async def run(args):
    import aiohttp
    profile = json.loads((args.plan / "profile.json").read_text())
    manifest = json.loads((args.plan / "plan-manifest.json").read_text())
    validate(profile, manifest["smoke"])
    assert digest(profile) == manifest["profile_sha256"]
    assert hashlib.sha256((args.plan / "plan.jsonl").read_bytes()).hexdigest() == manifest["plan_sha256"]
    rows = [json.loads(line) for line in (args.plan / "plan.jsonl").read_text().splitlines()]
    assert len(rows) == manifest["requests"] and len({r["index"] for r in rows}) == len(rows)
    prompts = {row["index"]: messages(row, profile) for row in rows}
    assert all(digest(prompts[row["index"]]) == row["prompt_sha256"] for row in rows), "Prompt differs from frozen plan"
    assert re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id)
    fd_limit = fd_check(profile["max_inflight"])
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ["profile.json", "plan.jsonl", "plan-manifest.json"]:
        shutil.copy2(args.plan / name, args.output / name)
    save(args.output / "run-config.json", dict(model=args.model, engine=args.engine, run_id=args.run_id,
         synthetic=True, smoke=manifest["smoke"], runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         profile_sha256=manifest["profile_sha256"], plan_sha256=manifest["plan_sha256"],
         fd_soft_limit=fd_limit, aiohttp_version=aiohttp.__version__, deadline_seconds=profile["deadline_seconds"],
         measurement_target=profile.get("measurement_target", "completion"),
         observation_horizon_seconds=profile.get("observation_horizon_seconds", profile["deadline_seconds"])))
    workload = Run(args, profile)
    workload.prompts = prompts
    error, drained = None, False
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        sampler = asyncio.create_task(workload.sample(session))
        try:
            async with session.get(args.router_metrics_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                response.raise_for_status()
                router_metrics = await response.text()
            assert metric(router_metrics, ["smg_estimated_wait_threshold_seconds"], maximum=True) == profile["router_extra_ttft_budget_seconds"], "Router extra-TTFT budget does not match the recorded profile"
            (args.output / "router-before.prom").write_text(router_metrics)
            if profile.get("startup_scenario") == "busy_start":
                await workload.verify_busy_start()
            else:
                await workload.drained()
                workload.initial_native_idle = True
            await workload.exercise(session, rows)
            drained = True
        except BaseException as exc:
            error = type(exc).__name__ + ": " + str(exc)
            for task in list(workload.active):
                task.cancel()
            await asyncio.gather(*list(workload.active), return_exceptions=True)
            try:
                await workload.drained()
                drained = True
            except Exception as cleanup_error:
                error += "; cleanup: " + type(cleanup_error).__name__ + ": " + str(cleanup_error)
        finally:
            workload.stop.set()
            await sampler
            workload.stream.close()
            workload.metrics_stream.close()
            workload.client_events.close()
            save(args.output / "stages.json", workload.stages)
            save(args.output / "execution.json", dict(error=error, requests_recorded=len(workload.rows),
                 planned_requests=len(rows), native_drained=drained,
                 phase_at_end=workload.phase, smoke=manifest["smoke"]))
    result = analyze(args.output, args.router_events)
    if error or result["execution_status"] != "PASS":
        raise RuntimeError("Workload execution invalid; preserve recorded evidence: " + str(error or result["execution_checks"]))
    return result


def classify(rows, prediction, actual):
    result = dict(tp=0, fn=0, fp=0, tn=0, unknown=0)
    for row in rows:
        p, y = row.get(prediction), row.get(actual)
        if p is None or y is None or row.get("controlled_rejection") or row.get("local_drop"):
            result["unknown"] += 1
        else:
            result["tp" if p and y else "fn" if y else "fp" if p else "tn"] += 1
    return result


def analyze(path, events_path=None):
    profile = json.loads((path / "profile.json").read_text())
    ttft_only = profile.get("measurement_target") == "ttft"
    config = json.loads((path / "run-config.json").read_text())
    execution = json.loads((path / "execution.json").read_text())
    rows = [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()]
    samples = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
    forecasts, first_text, gates, recovery_forecasts, feature_forecasts = {}, {}, {}, {}, {}
    isolation_events = []
    if events_path:
        for line in events_path.read_text().splitlines():
            try:
                obj = json.loads(line[line.index("{"):])
            except (ValueError, TypeError):
                continue
            event = obj.get("fields", {}) | obj
            isolation_events.append(event)
            rid = event.get("request_id", "")
            if not rid.startswith("prod-" + config["run_id"] + "-"):
                continue
            destination = {"calibration_forecast": forecasts, "first_text": first_text, "forecast": gates,
                           "recovery_forecast": recovery_forecasts,
                           "feature_context_forecast": feature_forecasts}.get(event.get("event"))
            if destination is not None:
                assert rid not in destination, "Duplicate exact request event: " + rid
                destination[rid] = event
        if events_path.resolve() != (path / "router-events.jsonl").resolve():
            shutil.copy2(events_path, path / "router-events.jsonl")
    deadline = profile["deadline_seconds"]
    for row in rows:
        f = forecasts.get(row["opc_request_id"])
        gate = gates.get(row["opc_request_id"])
        if f:
            assert f["threshold_seconds"] == profile["router_extra_ttft_budget_seconds"], "Forecast uses a different router budget"
        if gate:
            assert gate["threshold_seconds"] == profile["router_extra_ttft_budget_seconds"], "Admission gate uses a different budget"
        row["smg_ttft_seconds"] = first_text.get(row["opc_request_id"], {}).get("ttft_seconds")
        for label, key in [("raw", "predicted_ttft_seconds"), ("calibrated", "calibrated_ttft_seconds")]:
            known = (f.get("calibration_baseline_available") if label == "calibrated" and f and f.get("calibration_method") == "mse_poll_two_wave_queue_v2"
                     else f.get("baseline_available", True) if f else False)
            value = f.get(key) if f and known else None
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                value = None
            row[label + "_ttft_seconds"] = value
            row[label + "_total_ttft_over_deadline"] = value >= deadline if value is not None else None
        row["raw_gate_would_reject"] = gate.get("would_reject") if gate else None
        row["calibrated_gate_would_reject"] = f.get("calibrated_would_reject") if f else None
        row["pre_admission_recorded_extra_flags"] = {
            "raw": gate.get("would_reject") if gate else None,
            "calibrated": gate.get("calibrated_would_reject") if gate else None}
        row["smg_first_text_late"] = row["smg_ttft_seconds"] >= deadline if row["smg_ttft_seconds"] is not None else None
        row["pre_admission_total_ttft_predictions"] = {}
        if gate:
            row["pre_admission_components"] = {k: gate.get(k) for k in ["baseline_seconds", "baseline_available",
                "calibration_baseline_seconds", "calibration_baseline_source", "calibration_baseline_available",
                "queue_excess_seconds", "calibration_input_excess_seconds", "observed_delay_floor",
                "predicted_excess_seconds", "calibration_scale", "calibration_epoch", "calibration_bank_index",
                "calibration_history_revision", "calibration_fitted_revision", "native_ready", "ready"]}
        for label in ("raw", "queue_only_raw", "calibrated"):
            value = None
            if gate:
                if label == "raw" and gate.get("baseline_available", True):
                    value = gate.get("predicted_ttft_seconds")
                elif label != "raw" and gate.get("calibration_baseline_available"):
                    value = (gate.get("calibrated_ttft_seconds") if label == "calibrated" else
                             gate["calibration_baseline_seconds"] + gate["calibration_input_excess_seconds"])
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                value = None
            row["pre_admission_total_ttft_predictions"][label] = value
            row[label + "_pre_admission_total_ttft_over_deadline"] = value >= deadline if value is not None else None
        candidate = recovery_forecasts.get(row["opc_request_id"])
        if candidate:
            if not gate or candidate.get("worker") != gate.get("worker"):
                raise ValueError("Recovery candidate lacks the exact original worker snapshot")
            for candidate_key, original_key in [("original_queue_excess_seconds", "calibration_input_excess_seconds"),
                                                 ("legacy_calibration_scale", "calibration_scale")]:
                original = gate.get(original_key)
                current = candidate.get(candidate_key)
                if original is None or current is None or not math.isclose(original, current, rel_tol=1e-8, abs_tol=1e-8):
                    raise ValueError("Recovery candidate does not match exact original admission components")
            try:
                from .wait_recovery_candidate import pair as pair_recovery
            except ImportError:
                from wait_recovery_candidate import pair as pair_recovery
            row["recovery_candidate"] = pair_recovery(candidate,
                row["pre_admission_total_ttft_predictions"], profile["router_extra_ttft_budget_seconds"],
                {"calibrated": row["pre_admission_recorded_extra_flags"]["calibrated"]}, gate.get("native_ready"))
        feature = feature_forecasts.get(row["opc_request_id"])
        if feature is not None:
            try:
                from .wait_feature_context import forecast as pair_feature
            except ImportError:
                try:
                    from wait_feature_context import forecast as pair_feature
                except ImportError:
                    from scripts.wait_feature_context import forecast as pair_feature
            if gate is None:
                raise ValueError("Feature candidate lacks its exact original admission forecast")
            row["feature_context_candidate"] = pair_feature(feature, gate=gate,
                                                            ttft_deadline_seconds=deadline)
    isolation = None
    if ttft_only or feature_forecasts or profile.get("feature_context_required"):
        try:
            from .wait_ttft_readiness import audit_isolation
        except ImportError:
            try:
                from wait_ttft_readiness import audit_isolation
            except ImportError:
                from scripts.wait_ttft_readiness import audit_isolation
        isolation = audit_isolation(isolation_events, {r["opc_request_id"] for r in rows})
    def valid_ttft_measurement(row):
        return ttft_only and row.get("error") is None and not row.get("cancelled") and (
            row.get("intentional_stream_abort") and row.get("client_ttft_seconds") is not None
            or row.get("timed_out") and (row.get("first_text_late") is True
                or row.get("status") == 200 and row.get("client_ttft_seconds") is not None))
    errors = [r for r in rows if not r.get("local_drop") and not (r.get("complete") or valid_ttft_measurement(r)
              or (r.get("timed_out") and r.get("completion_late") is True) or r.get("controlled_rejection"))]
    lag = quantile([r.get("dispatch_lag_seconds") for r in rows], .99)
    checks = dict(no_execution_error=execution["error"] is None,
                  all_planned_arrivals_recorded=len(rows) == execution["planned_requests"],
                  no_local_saturation=not any(r.get("local_drop") for r in rows),
                  no_transport_or_protocol_errors=not errors,
                  producer_kept_up=lag is not None and lag <= profile["maximum_p99_dispatch_lag_seconds"],
                  native_drained=execution["native_drained"])
    if isolation is not None:
        checks["exclusive_workload_isolation"] = isolation["ok"]
    feature_activity, feature_windows = None, None
    if feature_forecasts or profile.get("feature_context_required"):
        try:
            from .wait_feature_context import audit_activity, audit_windows, audit_method
        except ImportError:
            try:
                from wait_feature_context import audit_activity, audit_windows, audit_method
            except ImportError:
                from scripts.wait_feature_context import audit_activity, audit_windows, audit_method
        feature_activity = audit_activity(isolation_events, rows)
        feature_windows = audit_windows(isolation_events)
        if profile.get("expected_feature_method"):
            method_audit = audit_method(isolation_events, profile["expected_feature_method"])
            save(path / "feature-method-audit.json", method_audit)
            checks["frozen_feature_method"] = method_audit["ok"]
        if profile.get("expected_feature_deadline_seconds") is not None:
            try:
                from .wait_feature_deadline import audit_deadlines
            except ImportError:
                try:
                    from wait_feature_deadline import audit_deadlines
                except ImportError:
                    from scripts.wait_feature_deadline import audit_deadlines
            deadline_audit = audit_deadlines(isolation_events, rows)
            save(path / "feature-deadline-audit.json", deadline_audit)
            checks["once_only_router_deadline_ledger"] = (profile["expected_feature_deadline_seconds"] == 300
                and deadline_audit["ok"] and deadline_audit["applicable"])
        checks["feature_actual_decisions_and_terminal_accounting"] = feature_activity["ok"]
        checks["feature_causal_rate_windows"] = feature_windows["ok"]
        checks["exact_feature_forecasts"] = all(r["opc_request_id"] in feature_forecasts for r in rows
            if not r.get("local_drop"))
    controls = [r for r in rows if r["control"]]
    pressure = [r for r in rows if not r["control"]]
    high_seconds = 0
    for a, b in zip(samples, samples[1:]):
        interval = b["sample_completed_elapsed_seconds"] - a["sample_completed_elapsed_seconds"]
        if all(r.get("native_kv") is not None and r["native_kv"] >= .9 and not r.get("engine_error") for r in [a, b]) and 0 <= interval <= 5 * profile["sample_interval_seconds"]:
            high_seconds += interval
    coverage = profile["coverage"]
    healthy = Counter(r["cohort"] for r in controls if r.get("complete") and not r["completion_late"])
    long_outputs = (sum(r.get("complete", False) and (r.get("usage", {}).get("completion_tokens") or 0) >= coverage["long_output_tokens"] for r in rows)
                    if not ttft_only else None)
    covered = dict(actual_300_second_deadline=deadline == 300 and not config["smoke"],
                   exact_request_forecasts=bool(forecasts) and all(r["opc_request_id"] in forecasts and r["opc_request_id"] in gates for r in rows if not r.get("local_drop") and not r.get("controlled_rejection")),
                   controlled_rejections_audited=all(gates.get(r["opc_request_id"], {}).get("would_reject") is True and gates[r["opc_request_id"]].get("ready") is True for r in rows if r.get("controlled_rejection")),
                   exact_first_text_events=bool(first_text) and all(r["opc_request_id"] in first_text for r in rows if r.get("client_ttft_seconds") is not None))
    if not ttft_only:
        covered.update(pressure_deadline_misses=sum(r.get("timed_out", False) and r.get("completion_late") is True for r in pressure) >= coverage["minimum_pressure_timeouts"],
            healthy_controls_every_cohort=all(healthy[c["name"]] >= coverage["minimum_healthy_controls_per_cohort"] for c in profile["cohorts"]),
            sustained_native_kv_pressure=high_seconds >= coverage["minimum_kv_90_observed_seconds"],
            actual_long_outputs=long_outputs >= coverage["minimum_completed_long_outputs"],
            healthy_boundary_completions=sum(r.get("complete", False) and .8 * deadline <= r["e2e_seconds"] < deadline for r in pressure) >= coverage["minimum_healthy_boundary_completions"])
    if ttft_only:
        readiness_path = path / "readiness.json"
        readiness = json.loads(readiness_path.read_text()) if readiness_path.exists() else {}
        healthy = Counter(r["cohort"] for r in controls if r.get("first_text_late") is False)
        covered = {k: covered[k] for k in ["actual_300_second_deadline", "exact_request_forecasts", "controlled_rejections_audited", "exact_first_text_events"]}
        covered.update(observation_beyond_ttft_threshold=profile["observation_horizon_seconds"] > deadline,
                       readiness_verified_before_pressure=readiness.get("ready") is True)
        if profile.get("ttft_deadline_policy") == "cancel_if_no_first_text":
            covered.pop("observation_beyond_ttft_threshold")
            covered["pending_first_text_deadline_contract"] = all(
                r.get("cancellation_reason") == "ttft_deadline" and r.get("e2e_seconds", 0) >= deadline
                and r.get("cancellation_delivered_elapsed_seconds", -1) >= r["ttft_deadline_elapsed_seconds"]
                for r in rows if r.get("timed_out") and r.get("client_ttft_seconds") is None and r.get("error") is None)
            covered["post_first_text_completion_is_separate"] = all(
                r.get("cancellation_reason") != "ttft_deadline" for r in rows
                if r.get("client_ttft_seconds") is not None and r["client_ttft_seconds"] < deadline)
        if profile.get("readiness", {}).get("require_entry_epoch_continuity"):
            try:
                from .wait_ttft_readiness import audit_entry_continuity
            except ImportError:
                try:
                    from wait_ttft_readiness import audit_entry_continuity
                except ImportError:
                    from scripts.wait_ttft_readiness import audit_entry_continuity
            entry = audit_entry_continuity(isolation_events, rows, readiness, profile["readiness"]["before_phase"])
            save(path / "measurement-entry-continuity.json", entry)
            checks["measurement_training_epoch_continuity"] = entry["ok"]
        if coverage.get("kind") == "raw_hard_continuation":
            covered.update(actual_rejection_exposure=feature_activity is not None and feature_activity["actual_rejected"] >= coverage["minimum_actual_admission_rejections"],
                actual_admission_decision_exposure=feature_activity is not None and feature_activity["actual_decisions"] >= coverage["minimum_actual_admission_decisions"],
                resolved_feature_outcome_exposure=feature_activity is not None and feature_activity["known_terminal_outcomes"] >= coverage["minimum_resolved_feature_outcomes"])
        else:
            covered.update(healthy_ttft_controls_every_cohort=all(healthy[c["name"]] >= coverage["minimum_healthy_controls_per_cohort"] for c in profile["cohorts"]),
                pressure_ttft_misses=sum(r.get("first_text_late") is True for r in pressure) >= coverage["minimum_pressure_ttft_misses"],
                actual_observed_late_ttft=sum(r.get("client_ttft_seconds") is not None and r["client_ttft_seconds"] >= deadline for r in pressure) >= coverage["minimum_observed_late_ttft"])
    if feature_activity is not None:
        covered["controlled_rejections_audited"] &= feature_activity["ok"]
    if profile.get("recovery_candidate_required"):
        checks["exact_recovery_forecasts"] = all(r["opc_request_id"] in recovery_forecasts
            for r in rows if not r.get("local_drop") and not r.get("controlled_rejection"))
        methods = {event.get("recovery_method") for event in recovery_forecasts.values()}
        expected_method = profile.get("expected_recovery_method")
        checks["consistent_recovery_method"] = len(methods) == 1 and (
            expected_method is None or methods == {expected_method})
        checks["no_actual_recovery_probe_in_shadow_round"] = not any(
            e.get("event") in {"recovery_probe_reserved", "recovery_probe_response", "recovery_probe_released"}
            for e in isolation_events)
    scores = {}
    for group, subset in [("controls", controls), ("pressure", pressure)]:
        scores[group] = {}
        for policy in ["raw", "calibrated"]:
            scores[group][policy] = dict(
                total_ttft_prediction=classify(subset, policy + "_total_ttft_over_deadline", "first_text_late"),
                observed_smg_ttft_prediction=classify(subset, policy + "_total_ttft_over_deadline", "smg_first_text_late"),
                gate_vs_completion_deadline=classify(subset, policy + "_gate_would_reject", "completion_late"))
        for policy in ["raw", "queue_only_raw", "calibrated"]:
            scores[group].setdefault(policy, {})["pre_admission_total_ttft_prediction"] = classify(subset, policy + "_pre_admission_total_ttft_over_deadline", "first_text_late")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["phase"], row["cohort"])].append(row)
    by_group = [dict(phase=phase, cohort=cohort, requests=len(value),
        completed=sum(r.get("complete", False) for r in value), timeouts=sum(r.get("timed_out", False) for r in value),
        controlled_rejections=sum(r.get("controlled_rejection", False) for r in value),
        ttft_p50=quantile([r.get("client_ttft_seconds") for r in value], .5),
        first_answer_p50=quantile([r.get("first_answer_seconds") for r in value], .5),
        timeouts_after_first_text=sum(r.get("timed_out", False) and r.get("client_ttft_seconds") is not None for r in value),
        actual_output_tokens_p50=quantile([(r.get("usage", {}).get("completion_tokens")) for r in value if r.get("complete")], .5),
        completion_p95_observed=quantile([r.get("e2e_seconds") for r in value if r.get("complete")], .95),
        completed_latency_is_censored=any(r.get("timed_out") for r in value)) for (phase, cohort), value in groups.items()]
    result = dict(schema_version=1, synthetic=True, smoke=config["smoke"], execution_checks=checks,
                  execution_status="PASS" if all(checks.values()) else "INVALID",
                  coverage_checks=covered, production_coverage="COVERED" if all(covered.values()) else "INSUFFICIENT",
                  requests=len(rows), completed=sum(r.get("complete", False) for r in rows),
                  timeouts=sum(r.get("timed_out", False) for r in rows), actual_long_outputs=long_outputs,
                  kv_90_observed_seconds=high_seconds, dispatch_lag_p99_seconds=lag,
                  exact_selected_forecasts=len(forecasts), per_phase_cohort=by_group, scores=scores,
                  notes=["Synthetic proportions are not measured production traffic.",
                         "Output budgets are not forced; usage records actual completed token counts.",
                         "A timed-out request with no first text has a lower bound, not an exact 300s TTFT.",
                         "TTFT prediction and a gate decision versus full completion are separate scores.",
                         "Client first-text deadline scores are operational proxies; observed SMG scores and exact scatter pairs use the SMG clock.",
                         "Raw gate scores use the pre-admission forecast; calibrated gate scores use the selected shadow forecast, which is not an enforced decision.",
                         "Rejected counterfactual latency and locally dropped arrivals remain unknown.",
                         "Completed-only percentiles omit timeout tails; do not interpret them as all-request percentiles.",
                         "Without exact router events, forecast scores are unknown; metric snapshots are not paired predictions."])
    result.update(measurement_target=profile.get("measurement_target", "completion"),
                  observed_first_text=sum(r.get("client_ttft_seconds") is not None for r in rows),
                  observed_late_first_text=sum(r.get("client_ttft_seconds") is not None and r["client_ttft_seconds"] >= deadline for r in rows),
                  censored_late_first_text=sum(r.get("client_ttft_seconds") is None and r.get("first_text_late") is True for r in rows),
                  unknown_first_text_truth=sum(r.get("first_text_late") is None for r in rows),
                  intentional_first_text_aborts=sum(r.get("intentional_stream_abort", False) for r in rows))
    if recovery_forecasts or profile.get("recovery_candidate_required"):
        try:
            from .wait_recovery_candidate import scores as recovery_scores
        except ImportError:
            from wait_recovery_candidate import scores as recovery_scores
        result["recovery_candidate"] = {name: recovery_scores(subset, deadline)
            for name, subset in [("all_offered", rows), ("pressure", pressure)]
            + [("phase:" + phase, [r for r in rows if r["phase"] == phase])
               for phase in sorted({r["phase"] for r in rows})]}
    if feature_activity is not None:
        try:
            from .wait_feature_context import scores as feature_scores
        except ImportError:
            try:
                from wait_feature_context import scores as feature_scores
            except ImportError:
                from scripts.wait_feature_context import scores as feature_scores
        result["feature_context"] = dict(activity=feature_activity, causal_windows=feature_windows,
            scorecards={name: feature_scores(subset, deadline)
                for name, subset in [("all_offered", rows), ("pressure", pressure)]
                + [("phase:" + phase, [r for r in rows if r["phase"] == phase])
                   for phase in sorted({r["phase"] for r in rows})]})
    if isolation is not None:
        result["runtime_isolation"] = isolation
    if ttft_only:
        result["notes"] += ["Primary scores compare exact pre-admission total TTFT with the300s first-text threshold. Production extra-delay flags are distinct.",
            ("TTFT scoring allows capped responses to complete normally; horizon cancellation is distinct from first-text timing and answer quality."
             if profile.get("ttft_stream_policy") == "complete_bounded_response" else
             "TTFT mode intentionally closes the stream after first text; this does not establish full completion or answer quality."),
            "Missing first text after its configured waiting deadline is a censored late binary label only with full-bound observation and valid owned dispatch; numeric TTFT remains unavailable. Short cancellations/errors remain unknown."]
    save(path / "summary.json", result)
    with (path / "paired-production.jsonl").open("w") as stream:
        for row in rows:
            stream.write(encoded(row).decode() + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--profile", type=Path, default=HERE / "wait-production-mix.json")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--smoke", action="store_true", help="Explicitly excludes production coverage")
    replay = commands.add_parser("run")
    replay.add_argument("--plan", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--url", required=True)
    replay.add_argument("--model", required=True)
    replay.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    replay.add_argument("--engine-metrics-url", required=True)
    replay.add_argument("--router-metrics-url", required=True)
    replay.add_argument("--router-events", type=Path)
    replay.add_argument("--run-id", required=True)
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--input", type=Path, required=True)
    analysis.add_argument("--router-events", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        result = write_plan(args.output, json.loads(args.profile.read_text()), args.smoke)
    elif args.command == "run":
        result = asyncio.run(run(args))
    else:
        result = analyze(args.input, args.router_events)
    print(json.dumps({k: result[k] for k in result if k not in ["per_phase_cohort", "scores"]}, indent=2))
    if args.command == "run" and not result["smoke"] and result["production_coverage"] != "COVERED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
