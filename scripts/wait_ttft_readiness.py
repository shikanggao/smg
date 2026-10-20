"""Fail-closed readiness for an isolated, diagnostic TTFT experiment."""
from __future__ import annotations

import math
import hashlib
import json
import re
from collections import Counter


METHOD = "mse_poll_two_wave_queue_v2"
ISOLATION_REQUEST_EVENTS = frozenset({"forecast", "calibration_forecast", "calibration_feedback", "first_text", "request_transport"})


def native_completion_stats_proof(initial_text, current_text, backend):
    """Prove required native completion counters advanced on the same series.

    This proves updates, not correctness of the counters or their averages.
    Label values are hashed so receipts contain no raw scrape labels or text.
    """
    if backend == "vllm":
        required = ["vllm:request_prompt_tokens_count", "vllm:request_prompt_tokens_sum",
                    "vllm:request_prefill_kv_computed_tokens_count", "vllm:request_prefill_kv_computed_tokens_sum",
                    "vllm:e2e_request_latency_seconds_count"]
        names = {name: name for name in required}
    elif backend == "sglang":
        required = ["sglang:prompt_tokens_total", "sglang:e2e_request_latency_seconds_count"]
        names = {alias: name for name in required for alias in [name, name.replace(":", "_")]}
    else:
        raise ValueError("Expected vllm or sglang backend")
    sample = re.compile(r"([^\s{]+)(\{.*\})?\s+(\S+)(?:\s+\S+)?")
    label = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*("(?:\\.|[^"\\])*")\s*(?:,|$)')

    def parse(text):
        values, errors = {}, []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = re.split(r"[\s{]", line, maxsplit=1)[0]
            if name not in names:
                continue
            match = sample.fullmatch(line)
            labels, position = {}, 0
            try:
                if match is None:
                    raise ValueError
                raw_labels = (match[2] or "{}")[1:-1]
                while position < len(raw_labels):
                    item = label.match(raw_labels, position)
                    if item is None or item[1] in labels:
                        raise ValueError
                    labels[item[1]] = json.loads(item[2])
                    position = item.end()
                labels_hash = hashlib.sha256(json.dumps(labels, sort_keys=True,
                    separators=(",", ":")).encode()).hexdigest()
                key = (name, labels_hash)
                if key in values:
                    raise ValueError
                value = float(match[3])
                values[key] = value if math.isfinite(value) and value >= 0 else None
            except (ValueError, TypeError):
                errors.append(f"Malformed or duplicate required metric sample: {name}")
        return values, errors

    before, initial_errors = parse(initial_text)
    after, current_errors = parse(current_text)
    reasons = [f"Initial scrape: {error}" for error in initial_errors]
    reasons += [f"Current scrape: {error}" for error in current_errors]
    for name in required:
        for when, values in [("initial", before), ("current", after)]:
            if not any(names[metric] == name for metric, _ in values):
                reasons.append(f"Missing required metric in {when} scrape: {name}")
    series = []
    for metric, labels_hash in sorted(before.keys() | after.keys()):
        key = (metric, labels_hash)
        old, new = before.get(key), after.get(key)
        issues, delta = [], None
        if key not in before or key not in after:
            issues.append("Series is not present in both scrapes")
        elif old is None or new is None:
            issues.append("Counter is not finite and nonnegative")
        else:
            delta = new - old
            if not math.isfinite(delta):
                delta = None
                issues.append("Counter delta is not finite")
            elif delta < 0:
                issues.append("Counter reset or decreased")
            elif delta == 0:
                issues.append("Counter did not advance")
        series.append(dict(metric=metric, labels_sha256=labels_hash, before=old, after=new,
                           delta=delta, reasons=issues))
    if any(row["reasons"] for row in series):
        reasons.append("Required native completion counter series did not all advance")
    return dict(ok=not reasons, backend=backend, required_metrics=required, series=series,
                reasons=reasons)


def audit_isolation(events, owned_request_ids):
    """Audit the whole parsed router capture against exact planned request IDs.

    Raw ``forecast`` is the admission record for streaming and nonstreaming
    requests. It must be checked even when no calibration feedback follows.
    Every owned ID must have a raw forecast; absent/incomplete captures fail.
    Flat and tracing ``fields`` event dictionaries are accepted. Capture
    continuity and parsing of raw log lines remain the caller's responsibility.
    Only IDs, event kinds, timestamps and capture indices enter this receipt;
    request bodies, log messages and other payload fields are never returned.
    """
    if isinstance(owned_request_ids, str):
        raise ValueError("Expected an explicit collection of owned request IDs")
    owned = set(owned_request_ids)
    if not owned or any(not isinstance(rid, str) or not rid.strip() for rid in owned):
        raise ValueError("Expected nonempty, exact owned request IDs")
    kinds, foreign, unattributed = Counter(), {}, []
    owned_events, total, forecasts = 0, 0, set()
    for index, event in enumerate(events, 1):
        total += 1
        fields = event.get("fields", {})
        value = (fields if isinstance(fields, dict) else {}) | event
        kind = value.get("event")
        if kind not in ISOLATION_REQUEST_EVENTS:
            continue
        kinds[kind] += 1
        stamp = value.get("timestamp")
        occurrence = dict(capture_event_index=index, event_kind=kind,
                          timestamp=stamp if isinstance(stamp, str) else None)
        rid = value.get("request_id")
        if not isinstance(rid, str) or not rid.strip():
            unattributed.append(occurrence)
        elif rid in owned:
            owned_events += 1
            if kind == "forecast":
                forecasts.add(rid)
        else:
            foreign.setdefault(rid, []).append(occurrence)
    missing = sorted(owned - forecasts)
    reasons = []
    if foreign:
        reasons.append("Request events from outside the exact owned request ID set")
    if unattributed:
        reasons.append("Request events without an attributable request ID")
    if missing:
        reasons.append("Owned requests missing raw admission forecasts; capture is absent or incomplete")
    return dict(ok=not reasons, scope="whole_router_capture", owned_request_id_count=len(owned),
                capture_event_count=total, checked_event_count=sum(kinds.values()),
                ignored_event_count=total-sum(kinds.values()), event_kind_counts=dict(sorted(kinds.items())),
                owned_event_count=owned_events, owned_forecast_request_count=len(forecasts),
                missing_owned_forecasts=missing,
                foreign_event_count=sum(len(rows) for rows in foreign.values()), foreign_request_count=len(foreign),
                foreign_requests=[dict(request_id=rid, event_count=len(rows),
                                       event_kind_counts=dict(sorted(Counter(row["event_kind"] for row in rows).items())),
                                       events=rows) for rid, rows in sorted(foreign.items())],
                unattributed_event_count=len(unattributed), unattributed_events=unattributed, reasons=reasons)


class Readiness:
    def __init__(self, request_prefix, scenario="isolated"):
        assert scenario in {"isolated", "busy_start"}
        self.prefix = request_prefix
        self.scenario = scenario
        self.epochs, self.forecasts, self.fits, self.initial = {}, {}, set(), {}
        self.errors = []
        self.accepted = {}

    def observe(self, event):
        kind, worker = event.get("event"), event.get("worker")
        if not worker or kind not in {"forecast", "calibration_forecast", "calibration_feedback", "calibration_update", "calibration_reset"}:
            return
        if kind == "forecast" and not event.get("calibration_enabled"):
            return
        if kind != "calibration_reset" and event.get("calibration_method") != METHOD:
            self.errors.append("Unexpected calibration method")
            return
        rid = event.get("request_id")
        if rid and not rid.startswith(self.prefix):
            self.errors.append("Feedback or forecast from outside this isolated run")
        epoch = event.get("current_calibration_epoch", event.get("calibration_epoch"))
        if not isinstance(epoch, int):
            self.errors.append("Missing calibration epoch")
            return
        self.epochs[worker] = epoch
        if kind in {"forecast", "calibration_forecast"}:
            self.initial.setdefault(worker, all(event.get(k) == 0 for k in (
                "calibration_observation_sequence", "calibration_fit_sequence",
                "calibration_history_revision", "calibration_fitted_revision")))
            self.forecasts[worker] = event
        elif kind == "calibration_update":
            if (event.get("calibration_state") == "capacity_ready"
                    and event.get("sample_count_after", 0) >= 4
                    and event.get("fit_sequence_after", 0) >= 4
                    and event.get("calibration_fitted_revision", 0) > 0):
                self.fits.add((worker, epoch, event["calibration_fitted_revision"]))
        elif kind == "calibration_feedback" and event.get("reason") == "accepted":
            self.accepted[(worker, epoch)] = self.accepted.get((worker, epoch), 0) + 1

    def snapshot(self, native_idle_verified, native_busy_verified=False):
        workers = []
        for worker, epoch in self.epochs.items():
            f = self.forecasts.get(worker, {})
            baseline = f.get("calibration_baseline_seconds")
            baseline_known = (f.get("calibration_baseline_available") is True
                and f.get("calibration_baseline_source") in {"provisional", "measured"}
                and isinstance(baseline, (int, float)) and math.isfinite(baseline) and baseline >= 0)
            revision = f.get("calibration_fitted_revision", 0)
            checks = dict(empty_initial_history=self.initial.get(worker) is True,
                current_epoch=f.get("calibration_epoch") == epoch,
                baseline_with_provenance=baseline_known,
                native_capacity_ready=f.get("native_ready") is True,
                current_samples_at_least_four=f.get("calibration_sample_count", 0) >= 4,
                accepted_in_current_epoch=self.accepted.get((worker, epoch), 0) >= 4,
                actual_current_epoch_fit=(worker, epoch, revision) in self.fits)
            workers.append(dict(worker=worker, epoch=epoch, checks=checks,
                baseline_seconds=baseline, baseline_source=f.get("calibration_baseline_source"),
                samples=f.get("calibration_sample_count"), fitted_revision=revision))
        startup = native_busy_verified if self.scenario == "busy_start" else native_idle_verified
        return dict(ready=bool(workers) and startup and not self.errors
                    and all(all(w["checks"].values()) for w in workers),
                    scenario=self.scenario, initial_native_idle_verified=native_idle_verified,
                    initial_native_busy_verified=native_busy_verified, workers=workers, errors=self.errors[:20])
