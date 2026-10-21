"""Fail-closed readiness for an isolated, diagnostic TTFT experiment."""
from __future__ import annotations

import math
import hashlib
import json
import re
from collections import Counter


METHOD = "mse_poll_two_wave_queue_v2"
RECOVERY_REQUEST_EVENTS = frozenset({"recovery_forecast", "recovery_probe_attempt", "recovery_probe_reserved", "recovery_probe_response", "recovery_probe_released"})
FEATURE_REQUEST_EVENTS = frozenset({"feature_context_forecast", "feature_context_feedback",
    "feature_outcome", "feature_outcome_ignored", "feature_admission_decision",
    "feature_deadline_started", "feature_deadline_resolution", "feature_deadline_ignored"})
ISOLATION_REQUEST_EVENTS = frozenset({"forecast", "calibration_forecast", "calibration_feedback", "first_text", "request_transport"}) | RECOVERY_REQUEST_EVENTS | FEATURE_REQUEST_EVENTS


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
    Every owned ID needs a raw forecast, including rejected requests because
    the raw/feature snapshot is recorded before the cohort admission check.
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
    forecast_workers, recovery_bindings, feature_bindings = {}, [], []
    verified_rejections = set()
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
                forecast_workers[rid] = value.get("worker")
            if kind in RECOVERY_REQUEST_EVENTS:
                recovery_bindings.append((rid, value.get("worker"), occurrence))
            if kind in FEATURE_REQUEST_EVENTS:
                feature_bindings.append((rid, value.get("worker"), occurrence))
                if kind == "feature_admission_decision" and value.get("decision") == "actual_rejected":
                    try:
                        try:
                            from .wait_feature_context import payload, validate_context
                        except ImportError:
                            from wait_feature_context import payload, validate_context
                        snapshot = payload(value)
                        _, counts = validate_context(snapshot["activity"])
                        if snapshot["counts"] != counts or value.get("outcome_tracking_supported") is not False:
                            raise ValueError("Invalid actual rejection snapshot")
                        verified_rejections.add(rid)
                    except (ValueError, KeyError, TypeError):
                        unattributed.append(dict(occurrence, reason="Unverified actual rejection event"))
        else:
            foreign.setdefault(rid, []).append(occurrence)
    missing = sorted(owned - forecasts)
    mismatched_workers = [dict(request_id=rid, worker=worker, **occurrence)
        for rid, worker, occurrence in recovery_bindings
        if not isinstance(worker, str) or not worker or forecast_workers.get(rid) != worker]
    feature_mismatches = [dict(request_id=rid, worker=worker, **occurrence)
        for rid, worker, occurrence in feature_bindings
        if not isinstance(worker, str) or not worker or forecast_workers.get(rid) != worker]
    reasons = []
    if foreign:
        reasons.append("Request events from outside the exact owned request ID set")
    if unattributed:
        reasons.append("Request events without an attributable request ID")
    if missing:
        reasons.append("Owned requests missing raw admission forecasts; capture is absent or incomplete")
    if mismatched_workers:
        reasons.append("Recovery request events do not match the exact owned raw-forecast worker")
    if feature_mismatches:
        reasons.append("Feature request events do not match the exact owned raw-forecast worker")
    return dict(ok=not reasons, scope="whole_router_capture", owned_request_id_count=len(owned),
                capture_event_count=total, checked_event_count=sum(kinds.values()),
                ignored_event_count=total-sum(kinds.values()), event_kind_counts=dict(sorted(kinds.items())),
                owned_event_count=owned_events, owned_forecast_request_count=len(forecasts),
                owned_actual_rejected_request_count=len(verified_rejections),
                verified_preselection_rejections_without_forecasts=sorted(verified_rejections - forecasts),
                missing_owned_forecasts=missing,
                recovery_worker_mismatch_count=len(mismatched_workers), recovery_worker_mismatches=mismatched_workers,
                feature_worker_mismatch_count=len(feature_mismatches), feature_worker_mismatches=feature_mismatches,
                foreign_event_count=sum(len(rows) for rows in foreign.values()), foreign_request_count=len(foreign),
                foreign_requests=[dict(request_id=rid, event_count=len(rows),
                                       event_kind_counts=dict(sorted(Counter(row["event_kind"] for row in rows).items())),
                                       events=rows) for rid, rows in sorted(foreign.items())],
                unattributed_event_count=len(unattributed), unattributed_events=unattributed, reasons=reasons)


def audit_entry_continuity(events, rows, readiness, measurement_phase):
    """Bind the first measured admission to the epoch that actually passed readiness."""
    measured = [r for r in rows if r.get("phase") == measurement_phase and not r.get("local_drop")]
    errors = []
    first = min(measured, key=lambda r: r["index"]) if measured else None
    candidates = []
    if first:
        for event in events:
            value = (event.get("fields", {}) or {}) | event
            if value.get("event") == "forecast" and value.get("request_id") == first["opc_request_id"]:
                candidates.append(value)
    if readiness.get("ready") is not True or len(candidates) != 1:
        errors.append("Missing successful readiness or exact first measured admission")
    gate = candidates[0] if len(candidates) == 1 else {}
    expected = next((w for w in readiness.get("workers", []) if w["worker"] == gate.get("worker")), {})
    checks = dict(same_trained_worker=bool(expected),
        known_scalar_method=gate.get("calibration_method") == METHOD,
        same_trained_epoch=bool(expected) and gate.get("calibration_epoch") == expected.get("epoch"),
        native_capacity_fresh=gate.get("native_ready") is True,
        current_samples_retained=gate.get("calibration_sample_count", 0) >= 4,
        current_fit_retained=bool(expected) and gate.get("calibration_fitted_revision", 0) >= expected.get("fitted_revision", 0) > 0,
        baseline_retained=gate.get("calibration_baseline_available") is True)
    if not all(checks.values()):
        errors.append("Training epoch or usable fitted history was lost before measurement entry")
    return dict(ok=not errors, errors=errors, measurement_phase=measurement_phase,
                first_measured_request_id=first["opc_request_id"] if first else None,
                readiness_worker=expected, checks=checks,
                admission={key: gate.get(key) for key in ["worker", "calibration_epoch", "calibration_sample_count",
                    "calibration_fitted_revision", "native_ready", "calibration_baseline_available"]},
                scope="Readiness to first measured admission only; later genuine lifecycle or recovery resets remain observable.")


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
