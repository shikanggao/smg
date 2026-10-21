"""Exact-admission parsing/scoring for an untrained remaining-work ablation.

Legacy calibration keeps its original method, gain and fitter. Ineligible live
estimates are unknown in candidate-only scores; a separate policy explicitly
falls back to the same admission's original prediction. Neither is enforcement.
"""
from __future__ import annotations

from collections import Counter
import math


METHOD = "live_net_depletion_v1"
CURRENT_METHOD = "live_arrival_bounded_depletion_v2"
SUPPORTED_METHODS = frozenset((METHOD, CURRENT_METHOD))
TRAINING = "unchanged_original_queue_only"


def number(value, label, optional=False):
    if value is None and optional:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid recovery " + label)
    return value


def arrival_snapshot(event):
    """Validate REQUEST-count diagnostics; they are not token/service counters.

    These are the newest captured interval's fields. Checking every credited
    W/P bump still requires replaying the complete poll history.
    """
    def counter(key, optional=False):
        value = event.get(key)
        if value is None and optional:
            return None
        if type(value) is not int or not 0 <= value <= (1 << 64) - 1:
            raise ValueError("Invalid recovery REQUEST counter: " + key)
        return value

    total = counter("total_dispatched_requests")
    watermark = counter("poll_request_watermark", optional=True)
    delta = counter("requests_since_previous_poll", optional=True)
    allowance = counter("rebound_allowance_requests")
    retained = event.get("rebound_retained")
    if type(retained) is not bool or allowance != min(delta or 0, 1):
        raise ValueError("Recovery rebound allowance differs from captured REQUEST credit")
    if watermark is None:
        if delta is not None or allowance or retained or event["available"]:
            raise ValueError("Recovery REQUEST interval lacks a poll-start watermark")
    elif watermark > total or delta is not None and delta > watermark:
        raise ValueError("Recovery REQUEST watermark is inconsistent with cumulative counts")
    if retained and allowance != 1:
        raise ValueError("Recovery retained rebound lacks fresh REQUEST credit")
    if event["available"] and delta is None:
        raise ValueError("Eligible recovery window lacks its newest REQUEST interval")
    return {key: event.get(key) for key in ("poll_request_watermark", "total_dispatched_requests",
        "requests_since_previous_poll", "rebound_allowance_requests", "rebound_retained")}


def pair(event, original_totals, threshold, original_flags=None, original_native_ready=None):
    method = event.get("recovery_method")
    if event.get("event") != "recovery_forecast" or method not in SUPPORTED_METHODS:
        raise ValueError("Unknown recovery forecast method")
    if any(not isinstance(event.get(k), str) or not event[k] for k in ("request_id", "worker")):
        raise ValueError("Recovery exact request and worker identity are required")
    clock = event.get("diagnostic_time_seconds")
    if type(clock) not in (int, float) or not math.isfinite(clock):
        raise ValueError("Recovery shared diagnostic clock is required")
    if (event.get("gain_training") != TRAINING or event.get("candidate_is_calibrated") is not False
            or event.get("native_cancellation_acknowledgement") != "unavailable"):
        raise ValueError("Recovery candidate training/acknowledgement contract changed")
    if type(event.get("available")) is not bool or not isinstance(event.get("reason"), str):
        raise ValueError("Recovery eligibility and reason are required")
    arrivals = arrival_snapshot(event) if method == CURRENT_METHOD else {}
    if event.get("threshold_seconds") != threshold:
        raise ValueError("Recovery threshold differs from exact admission budget")
    for field in ("original_queue_excess_seconds", "queue_excess_seconds", "window_seconds", "dispatch_debt_seconds"):
        number(event.get(field), field)
    polls = event.get("window_poll_count")
    if type(polls) is not int or not 0 <= polls <= 121 or event["window_seconds"] > 120:
        raise ValueError("Recovery window exceeds reviewed bounds")
    baseline = number(event.get("baseline_seconds"), "baseline_seconds", optional=True)
    gain = number(event.get("legacy_calibration_scale"), "unchanged legacy gain")
    totals = {"queue_only_raw": number(event.get("predicted_ttft_seconds"), "total", optional=True),
              "legacy_gain_ablation": number(event.get("legacy_gain_ablation_ttft_seconds"), "gain ablation total", optional=True)}
    if baseline is None and any(v is not None for v in totals.values()):
        raise ValueError("Unknown recovery baseline cannot produce a known total")
    rate = number(event.get("net_population_depletion_per_second"), "net depletion", optional=True)
    population = number(event.get("native_population"), "native population", optional=True)
    waiting = number(event.get("native_waiting"), "native waiting", optional=True)
    if population is not None and waiting is not None and waiting > population:
        raise ValueError("Recovery waiting exceeds total population")
    if baseline is not None and (totals["queue_only_raw"] is None or not math.isclose(
            totals["queue_only_raw"], baseline + event["queue_excess_seconds"], rel_tol=1e-8, abs_tol=1e-8)):
        raise ValueError("Recovery total does not reproduce saved baseline plus queue")
    if baseline is not None and (totals["legacy_gain_ablation"] is None or not math.isclose(
            totals["legacy_gain_ablation"], baseline + gain * event["queue_excess_seconds"], rel_tol=1e-8, abs_tol=1e-8)):
        raise ValueError("Recovery ablation total does not reproduce unchanged legacy gain")
    if event["available"]:
        if event["window_seconds"] < 30 or polls < 5 or rate is None or rate <= 0 or population is None or waiting is None:
            raise ValueError("Recovery eligibility lacks reviewed window/depletion evidence")
        debt = event["dispatch_debt_seconds"]
        expected = min(max(event["original_queue_excess_seconds"] - debt, 0.), population / rate) + debt
        if not math.isclose(expected, event["queue_excess_seconds"], rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("Recovery queue does not reproduce net-depletion formula")
    elif not math.isclose(event["original_queue_excess_seconds"], event["queue_excess_seconds"], rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("Unavailable recovery queue is not the original fallback")
    flags = {"queue_only_raw": event.get("would_reject"),
             "legacy_gain_ablation": event.get("legacy_gain_ablation_would_reject")}
    total_flags = {"queue_only_raw": event.get("total_would_reject"),
                   "legacy_gain_ablation": event.get("legacy_gain_ablation_total_would_reject")}
    for policy, flag in total_flags.items():
        expected = totals[policy] >= threshold if event["available"] and totals[policy] is not None else None
        if type(flag) is not type(expected) or flag != expected:
            raise ValueError("Recovery saved total flag differs from exact total")
    for flag in flags.values():
        if flag is not None and (type(flag) is not bool or not event["available"]):
            raise ValueError("Recovery EXTRA flag lacks eligible evidence")
    if flags["queue_only_raw"] is not None and flags["queue_only_raw"] != (event["queue_excess_seconds"] >= threshold):
        raise ValueError("Recovery saved EXTRA flag differs from queue")
    if flags["legacy_gain_ablation"] is not None and flags["legacy_gain_ablation"] != (gain * event["queue_excess_seconds"] >= threshold):
        raise ValueError("Recovery saved ablation EXTRA flag differs from unchanged gain")
    eligible = totals if event["available"] else {k: None for k in totals}
    fallback = eligible.copy() if event["available"] else {
        "queue_only_raw": original_totals.get("queue_only_raw"),
        "legacy_gain_ablation": original_totals.get("calibrated"),
    }
    for value in fallback.values():
        number(value, "fallback total", optional=True)
    original_flags = original_flags or {}
    return dict(method=method, available=event["available"], reason=event["reason"],
                gain_training=TRAINING, candidate_is_calibrated=False,
                eligible_total_ttft_predictions=eligible,
                explicit_fallback_total_ttft_predictions=fallback,
                original_on_candidate_eligible_total_ttft_predictions={
                    "queue_only_raw": original_totals.get("queue_only_raw") if eligible["queue_only_raw"] is not None else None,
                    "legacy_gain_ablation": original_totals.get("calibrated") if eligible["legacy_gain_ablation"] is not None else None},
                eligible_recorded_extra_flags=flags,
                explicit_fallback_recorded_extra_flags=flags if event["available"] else {
                    # Preserve a plain-N comparator; the production raw flag can
                    # include a floor and remains a separate legacy scorecard.
                    "queue_only_raw": event["original_queue_excess_seconds"] >= threshold
                        if original_native_ready is True else None,
                    "legacy_gain_ablation": original_flags.get("calibrated")},
                original_native_capacity_ready=original_native_ready,
                components={k: event.get(k) for k in ("original_queue_excess_seconds", "queue_excess_seconds", "baseline_seconds",
                    "legacy_calibration_scale", "calibration_epoch", "calibration_history_revision", "calibration_fit_sequence",
                    "native_counts_profile", "window_started_time_seconds", "latest_poll_started_time_seconds",
                    "native_population", "native_waiting", "diagnostic_time_seconds", "worker",
                    "net_population_depletion_per_second", "window_seconds", "window_poll_count", "dispatch_debt_seconds")} | arrivals)


def scores(rows, threshold):
    result = dict(offered=len(rows), captured=sum(r.get("recovery_candidate") is not None for r in rows),
                  eligible=sum(bool(r.get("recovery_candidate", {}).get("available")) for r in rows),
                  healthy=sum(r.get("first_text_late") is False and not r.get("controlled_rejection") and not r.get("local_drop") for r in rows),
                  late=sum(r.get("first_text_late") is True and not r.get("controlled_rejection") and not r.get("local_drop") for r in rows),
                  reasons=dict(Counter(r.get("recovery_candidate", {}).get("reason", "not_captured") for r in rows)))
    for scope in ("eligible_total_ttft_predictions", "explicit_fallback_total_ttft_predictions",
                  "original_on_candidate_eligible_total_ttft_predictions",
                  "eligible_recorded_extra_flags", "explicit_fallback_recorded_extra_flags"):
        result[scope] = {}
        for policy in ("queue_only_raw", "legacy_gain_ablation"):
            card = dict(tp=0, fp=0, fn=0, tn=0, unknown=0, prediction_unknown=0,
                        truth_unknown=0, censored_known_late=0)
            for row in rows:
                candidate = row.get("recovery_candidate") or {}
                prediction = candidate.get(scope, {}).get(policy)
                truth = row.get("first_text_late")
                if row.get("controlled_rejection") or row.get("local_drop"):
                    truth = None
                card["prediction_unknown"] += prediction is None
                card["truth_unknown"] += truth is None
                card["censored_known_late"] += truth is True and row.get("client_ttft_seconds") is None
                if prediction is None or truth is None:
                    card["unknown"] += 1
                else:
                    over = prediction if scope.endswith("flags") else prediction >= threshold
                    card["tp" if over and truth else "fp" if over else "fn" if truth else "tn"] += 1
            card["fpr"] = card["fp"] / (card["fp"] + card["tn"]) if card["fp"] + card["tn"] else None
            card["fnr"] = card["fn"] / (card["fn"] + card["tp"]) if card["fn"] + card["tp"] else None
            result[scope][policy] = card
    return result
