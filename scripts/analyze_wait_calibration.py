#!/usr/bin/env python3
"""Audit causal online TTFT calibration and compare exact same-request forecasts.

Input is one harness suite directory containing requests.tsv, summary.json and
router-events.jsonl. Logs may include previous suites from the same router.
No offline fit is substituted for the live calibration forecasts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path


def median_scale(history):
    if len(history) < 4:
        return 1.0
    # Stable ratio-only sort matches Rust; tied ratios retain arrival order.
    values = sorted(history, key=lambda entry: entry[0])
    largest = max(weight for _, weight in values)
    # Match Rust f64 iterator summation, rather than Python >=3.12's
    # compensated sum: an exact half-weight tie can choose another endpoint.
    total = 0.0
    for _, weight in values:
        total += weight / largest
    target = total / 2
    cumulative = 0
    for ratio, weight in values:
        cumulative += weight / largest
        if cumulative >= target:
            return ratio
    return 1.0


def mse_scale(history):
    if len(history) < 4:
        return 1.0
    largest = max(x for _, x in history)
    total = target = 0.0
    for ratio, x in history:
        weight = (x / largest) ** 2
        if weight > 0:
            total += weight
            target += (ratio - target) * (weight / total)
    return target


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=2e-8, abs_tol=2e-8)


def events(path):
    for line in path.read_text().splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        yield obj.get("fields", {}) | obj


SCALE_UNITS_PER_SECOND = 0.1
TWO_WAVE_METHOD = "mse_poll_two_wave_v1"
QUEUE_METHOD = "mse_poll_two_wave_queue_v2"
TWO_WAVE_METHODS = (TWO_WAVE_METHOD, QUEUE_METHOD)
TWO_WAVE_RATE = 0.5


def applied_scale(version, at, rate=SCALE_UNITS_PER_SECOND):
    """Evaluate a saved target trajectory; elapsed time never banks a new target."""
    anchor, target, since, _ = version
    budget = rate * max(at - since, 0)
    return min(anchor + budget, max(anchor - budget, target))


def feedback_reason(event, eligibility_required):
    """Replay observable guards in engine order; freshness was not logged.

    Generation and stale skips remain authoritative because their saved Arc /
    freshness values are not present in these traces. All subsequent guards,
    including native queue eligibility, are directly observable.
    """
    if event["reason"] in ["calibration_generation_changed", "stale_at_admission"]:
        return event["reason"]
    if event["calibration_epoch"] != event["current_calibration_epoch"]:
        return "calibration_generation_changed"
    if (
        "feedback_age_seconds" in event
        and "calibration_max_age_seconds" in event
        and event["feedback_age_seconds"] >= event["calibration_max_age_seconds"]
    ):
        return "feedback_expired"
    queue_only = event.get("calibration_method") == QUEUE_METHOD
    if not event.get(
        "calibration_baseline_available" if queue_only else "baseline_available", True
    ):
        return "baseline_unavailable"
    baseline = event["calibration_baseline_seconds" if queue_only else "baseline_seconds"]
    ttft = event["ttft_seconds"]
    extra = event["calibration_input_excess_seconds" if queue_only else "raw_excess_seconds"]
    if not math.isfinite(baseline) or baseline < 0 or not math.isfinite(ttft) or ttft <= 0:
        return "invalid_observation"
    if not math.isfinite(extra) or extra <= 0:
        return "nonpositive_extra"
    if not math.isfinite(max(ttft - baseline, 0) / extra):
        return "invalid_ratio"
    if eligibility_required:
        if not event.get("admission_native_capacity_ready"):
            return "native_capacity_unready"
        if not event.get("admission_native_queue_visible"):
            return "native_queue_not_visible"
    return "accepted"


def two_wave_trace(log, keep_records=False):
    """Replay chronological banks; revisions identify removal-only poll fits.

    Bank age starts at the first accepted feedback for the worker epoch, not
    admission time. Ordinary rotations never create a new generation.
    """
    states, origins, clocks, selected, gates = {}, {}, {}, {}, {}
    rotation_states, rotation_times = defaultdict(set), {}
    records, errors = [], []
    counts = Counter()
    peak_samples = 0
    methods = {
        e["calibration_method"]
        for e in log
        if e.get("event")
        in (
            "calibration_forecast",
            "calibration_feedback",
            "calibration_update",
            "calibration_rotation",
            "calibration_baseline",
        )
        or e.get("event") == "forecast"
        and e.get("calibration_enabled")
    }
    method = next(iter(methods)) if len(methods) == 1 else "mixed_or_unknown"
    queue_only = method == QUEUE_METHOD
    baselines, last_resets, worker_epochs, pending_baseline_resets = {}, {}, {}, {}

    def baseline_fields(event):
        source = event["calibration_baseline_source"]
        value = event["calibration_baseline_seconds"]
        available = event["calibration_baseline_available"]
        if source not in ("unknown", "provisional", "measured") or type(available) is not bool:
            raise ValueError("unknown_calibration_baseline_source_or_availability")
        if source == "unknown":
            if available or value != -1.0:
                raise ValueError("unknown_baseline_must_remain_unavailable")
        elif (
            not available
            or not math.isfinite(value)
            or value < 0
            or source == "provisional"
            and value <= 0
        ):
            raise ValueError("invalid_calibration_baseline")
        return source, value

    def check(label, actual, expected):
        if not close(actual, expected):
            raise ValueError(f"{label}: recorded={actual!r}, expected={expected!r}")

    def samples(state):
        return list(state["previous"]) + list(state["current"])

    for event in log:
        kind = event.get("event")
        if kind == "calibration_reset" and event.get("reason") == "history_expired":
            errors.append(["two_wave_history_expiry_generation_reset", event.get("worker")])
            break
        if kind == "calibration_reset" and queue_only:
            try:
                if event["calibration_epoch"] <= worker_epochs.get(event["worker"], -1):
                    raise ValueError("generation_reset_did_not_advance_epoch")
                worker_epochs[event["worker"]] = event["calibration_epoch"]
                last_resets[event["worker"]] = (event["calibration_epoch"], event["reason"])
                # All generation resets clear provenance. Baseline-triggered
                # resets log the captured pre-reset value in the next event.
                previous = baselines.get(event["worker"], ("unknown", -1.0))
                if event["reason"] in ("baseline_promoted", "baseline_fallback_changed"):
                    pending_baseline_resets[event["worker"]] = previous
                else:
                    pending_baseline_resets.pop(event["worker"], None)
                baselines[event["worker"]] = ("unknown", -1.0)
            except (KeyError, ValueError) as error:
                errors.append(["malformed_calibration_reset", str(error)])
                break
        if kind not in (
            "calibration_forecast",
            "calibration_feedback",
            "calibration_update",
            "calibration_rotation",
            "calibration_baseline",
        ) and not (kind == "forecast" and event.get("calibration_enabled")):
            continue
        try:
            if method not in TWO_WAVE_METHODS or event["calibration_method"] != method:
                raise ValueError("mixed_or_unknown_calibration_methods")
            if queue_only:
                epoch = event.get("current_calibration_epoch", event["calibration_epoch"])
                previous_epoch = worker_epochs.setdefault(event["worker"], epoch)
                if epoch != previous_epoch:
                    raise ValueError("epoch_changed_without_generation_reset")
            if kind == "calibration_baseline":
                if not queue_only:
                    raise ValueError("baseline_transition_in_legacy_method")
                previous = pending_baseline_resets.get(
                    event["worker"], baselines.get(event["worker"], ("unknown", -1.0))
                )
                if event["baseline_previous_source"] != previous[0]:
                    raise ValueError("baseline_previous_source_mismatch")
                check("baseline_previous_seconds", event["baseline_previous_seconds"], previous[1])
                current = baseline_fields(event)
                if current == previous:
                    raise ValueError("redundant_baseline_transition")
                promoted = (
                    previous[0] == "provisional"
                    and current[0] == "measured"
                    and abs(current[1] - previous[1]) > max(0.1, 0.2 * previous[1])
                )
                if event["calibration_baseline_material_promotion"] is not promoted:
                    raise ValueError("incorrect_material_baseline_promotion")
                if promoted and last_resets.get(event["worker"]) != (
                    event["calibration_epoch"],
                    "baseline_promoted",
                ):
                    raise ValueError("material_baseline_promotion_missing_epoch_reset")
                fallback_changed = (
                    previous[0] != "unknown" and current[0] != "measured" and previous != current
                )
                if fallback_changed and last_resets.get(event["worker"]) != (
                    event["calibration_epoch"],
                    "baseline_fallback_changed",
                ):
                    raise ValueError("baseline_fallback_change_missing_epoch_reset")
                if event["worker"] in pending_baseline_resets and not (
                    promoted or fallback_changed
                ):
                    raise ValueError("baseline_reset_does_not_match_transition")
                baselines[event["worker"]] = current
                pending_baseline_resets.pop(event["worker"], None)
                counts["baseline_transitions"] += 1
                counts["baseline_promotions"] += promoted
                continue
            if queue_only and event["worker"] in pending_baseline_resets:
                raise ValueError("baseline_reset_missing_transition")
            epoch = event.get("current_calibration_epoch", event["calibration_epoch"])
            worker_epoch = (event["worker"], epoch)
            key = worker_epoch + (event["calibration_state"],)
            if key[2] not in ("capacity_ready", "capacity_unready"):
                raise ValueError("unknown_calibration_state")
            now = event["calibration_time_seconds"]
            if not math.isfinite(now) or now < clocks.get(worker_epoch, 0):
                raise ValueError("monotonic_time_regressed")
            clocks[worker_epoch] = now
            state = states.setdefault(
                key,
                dict(
                    previous=deque(maxlen=32),
                    current=deque(maxlen=32),
                    bank=0,
                    revision=0,
                    fitted_revision=0,
                    accepted=0,
                    fitted=0,
                    trajectory=(1.0, 1.0, 0.0, 0),
                ),
            )
            origins.setdefault(worker_epoch, None)
            origin = origins[worker_epoch]
            due = int((now - origin) // 300) if origin is not None else 0
            before = applied_scale(state["trajectory"], now, TWO_WAVE_RATE)
            before_count = len(samples(state))
            removed = 0
            if kind == "calibration_rotation":
                if origin is None or due <= state["bank"]:
                    raise ValueError("rotation_before_acceptance_or_duplicate")
                steps = due - state["bank"]
                rotation_key = worker_epoch + (due,)
                if rotation_key in rotation_times and now != rotation_times[rotation_key]:
                    raise ValueError("rotation_state_timestamps_differ")
                rotation_times[rotation_key] = now
                rotation_states[rotation_key].add(key[2])
                removed = before_count if steps >= 2 else len(state["previous"])
                state["previous"] = deque(state["current"] if steps == 1 else (), maxlen=32)
                state["current"].clear()
                state["bank"] = due
                if removed:
                    state["revision"] += 1
                    if len(samples(state)) < 4:
                        state["trajectory"] = (1.0, 1.0, now, len(samples(state)))
                check("rotation_bank_seconds", event["calibration_bank_seconds"], 300)
                check(
                    "rotation_previous_count",
                    event["calibration_previous_sample_count"],
                    len(state["previous"]),
                )
                check(
                    "rotation_current_count",
                    event["calibration_current_sample_count"],
                    len(state["current"]),
                )
                check("rotation_count_before", event["sample_count_before"], before_count)
                check("rotation_count_after", event["sample_count_after"], len(samples(state)))
                check("rotation_scale_before", event["scale_before"], before)
                check(
                    "rotation_scale_after",
                    event["scale_after"],
                    applied_scale(state["trajectory"], now, TWO_WAVE_RATE),
                )
                check("rotation_target_after", event["target_scale_after"], state["trajectory"][1])
                counts["rotations"] += 1
                counts["removed_samples"] += removed
            else:
                if due != state["bank"]:
                    raise ValueError("missing_bank_rotation")
                if kind in ("calibration_forecast", "forecast"):
                    if queue_only:
                        current_baseline = baseline_fields(event)
                        expected_baseline = baselines.get(event["worker"], ("unknown", -1.0))
                        if current_baseline[0] != expected_baseline[0]:
                            raise ValueError("forecast_baseline_source_without_transition")
                        check(
                            "forecast_calibration_baseline",
                            current_baseline[1],
                            expected_baseline[1],
                        )
                        if event["baseline_available"]:
                            if current_baseline[0] != "measured":
                                raise ValueError("measured_baseline_did_not_win")
                            check(
                                "measured_calibration_baseline",
                                current_baseline[1],
                                event["baseline_seconds"],
                            )
                        elif current_baseline[0] == "measured":
                            raise ValueError(
                                "measured_calibration_baseline_without_raw_measurement"
                            )
                        check(
                            "queue_only_input",
                            event["calibration_input_excess_seconds"],
                            event["queue_excess_seconds"],
                        )
                        if (
                            not math.isfinite(event["queue_excess_seconds"])
                            or event["queue_excess_seconds"] < 0
                        ):
                            raise ValueError("invalid_queue_calibration_input")
                        counts["baseline_source:" + current_baseline[0]] += 1
                    seen = selected if kind == "calibration_forecast" else gates
                    if event["request_id"] in seen:
                        raise ValueError("duplicate_request_forecast")
                    seen[event["request_id"]] = event
                    for field, value in (
                        ("calibration_observation_sequence", state["accepted"]),
                        ("calibration_fit_sequence", state["fitted"]),
                        ("calibration_history_revision", state["revision"]),
                        ("calibration_fitted_revision", state["fitted_revision"]),
                        ("calibration_bank_index", state["bank"]),
                        ("calibration_sample_count", before_count),
                        ("calibration_scale", before),
                        ("calibration_target_scale", state["trajectory"][1]),
                        ("calibration_max_scale_change_per_second", TWO_WAVE_RATE),
                        ("calibration_bank_seconds", 300),
                        ("calibration_bank_max_samples", 32),
                    ):
                        check("forecast_" + field, event[field], value)
                    if event["calibration_ready"] is not (before_count >= 4):
                        raise ValueError("incorrect_calibration_readiness")
                    base, extra = event["baseline_seconds"], event["predicted_excess_seconds"]
                    check("raw_forecast_arithmetic", event["predicted_ttft_seconds"], base + extra)
                    if queue_only:
                        base, extra = (
                            event["calibration_baseline_seconds"],
                            event["calibration_input_excess_seconds"],
                        )
                    check(
                        "calibrated_forecast_arithmetic",
                        event["calibrated_ttft_seconds"],
                        base + before * extra
                        if not queue_only or event["calibration_baseline_available"]
                        else -1.0,
                    )
                    check(
                        "calibrated_excess_arithmetic",
                        event["calibrated_excess_seconds"],
                        before * extra,
                    )
                    # Raw readiness can be promoted by the empirical floor.
                    # Queue calibration additionally requires native capacity;
                    # raw readiness supplies the selected snapshot's freshness.
                    reject_ready = event["ready"] and (
                        not queue_only
                        or event["native_ready"]
                        and event["calibration_baseline_available"]
                    )
                    reject = (
                        reject_ready
                        and math.isfinite(event["calibrated_excess_seconds"])
                        and event["calibrated_excess_seconds"] >= event["threshold_seconds"]
                    )
                    if event["calibrated_would_reject"] is not reject:
                        raise ValueError("incorrect_calibrated_readiness_gate")
                    if (
                        type(event["native_ready"]) is not bool
                        or type(event["native_queue_visible"]) is not bool
                        or event["native_queue_visible"] != (event["native_waiting"] > 0)
                    ):
                        raise ValueError("invalid_native_eligibility_snapshot")
                    mature = (
                        event["native_ready"]
                        if kind == "calibration_forecast"
                        else event.get("capacity_learned_ready")
                    )
                    if mature is not None and (key[2] == "capacity_ready") != mature:
                        raise ValueError("incorrect_native_capacity_state")
                    continue
                check("sequence_before", event["sequence_before"], state["accepted"])
                check("fit_sequence_before", event["fit_sequence_before"], state["fitted"])
                check("sample_count_before", event["sample_count_before"], before_count)
                check("scale_before", event["scale_before"], before)
                check("target_scale_before", event["target_scale_before"], state["trajectory"][1])
                if kind == "calibration_feedback":
                    counts["feedback"] += 1
                    counts["reason:" + event["reason"]] += 1
                    forecast = selected.get(event["request_id"])
                    if forecast is None:
                        raise ValueError("feedback_missing_selected_snapshot")
                    if (
                        forecast["worker"] != event["worker"]
                        or forecast["calibration_state"] != event["calibration_state"]
                    ):
                        raise ValueError("feedback_changed_saved_worker_or_state")
                    for source, target in (
                        ("baseline_seconds", "baseline_seconds"),
                        ("baseline_available", "baseline_available"),
                        ("predicted_excess_seconds", "raw_excess_seconds"),
                        ("calibration_scale", "admission_scale"),
                        ("calibration_sample_count", "admission_sample_count"),
                        ("calibration_observation_sequence", "admission_observation_sequence"),
                        ("calibration_fit_sequence", "admission_fit_sequence"),
                        ("calibration_epoch", "calibration_epoch"),
                        ("calibration_time_seconds", "admission_calibration_time_seconds"),
                        ("calibration_bank_index", "admission_bank_index"),
                        ("calibration_history_revision", "admission_history_revision"),
                        ("calibration_fitted_revision", "admission_fitted_revision"),
                        ("native_waiting", "admission_native_waiting"),
                        ("native_queue_visible", "admission_native_queue_visible"),
                        ("native_ready", "admission_native_capacity_ready"),
                    ):
                        check("saved_" + source, event[target], forecast[source])
                    if queue_only:
                        baseline_fields(event)
                        for field in (
                            "calibration_baseline_seconds",
                            "calibration_baseline_available",
                            "calibration_input_excess_seconds",
                            "queue_excess_seconds",
                        ):
                            check("saved_" + field, event[field], forecast[field])
                        if (
                            event["calibration_baseline_source"]
                            != forecast["calibration_baseline_source"]
                        ):
                            raise ValueError("feedback_changed_saved_baseline_source")
                    check("snapshot_max_age", event["calibration_max_age_seconds"], 600)
                    if (
                        type(event["admission_native_capacity_ready"]) is not bool
                        or type(event["admission_native_queue_visible"]) is not bool
                        or event["admission_native_queue_visible"]
                        != (event["admission_native_waiting"] > 0)
                    ):
                        raise ValueError("invalid_native_eligibility_feedback")
                    if (
                        now < event["admission_calibration_time_seconds"]
                        and epoch == event["calibration_epoch"]
                    ):
                        raise ValueError("feedback_before_its_admission")
                    if epoch == event["calibration_epoch"]:
                        check(
                            "feedback_age",
                            event["feedback_age_seconds"],
                            now - event["admission_calibration_time_seconds"],
                        )
                    reason = feedback_reason(event, True)
                    if event["reason"] != reason:
                        raise ValueError("incorrect_feedback_reason")
                    if reason == "accepted":
                        x = event[
                            "calibration_input_excess_seconds"
                            if queue_only
                            else "raw_excess_seconds"
                        ]
                        base = event[
                            "calibration_baseline_seconds" if queue_only else "baseline_seconds"
                        ]
                        state["current"].append(
                            (
                                max(event["ttft_seconds"] - base, 0) / x,
                                x,
                                now,
                                event["feedback_age_seconds"],
                            )
                        )
                        state["accepted"] += 1
                        state["revision"] += 1
                        if origin is None:
                            origins[worker_epoch] = now
                        counts["accepted"] += 1
                else:
                    if state["revision"] == state["fitted_revision"]:
                        raise ValueError("redundant_poll_fit")
                    target = mse_scale([(ratio, x) for ratio, x, _, _ in samples(state)])
                    state["trajectory"] = (before, target, now, len(samples(state)))
                    state["fitted"] = state["accepted"]
                    state["fitted_revision"] = state["revision"]
                    check(
                        "poll_fitted_revision",
                        event["calibration_fitted_revision"],
                        state["fitted_revision"],
                    )
                    counts["poll_fits"] += 1
                check("sequence_after", event["sequence_after"], state["accepted"])
                check("fit_sequence_after", event["fit_sequence_after"], state["fitted"])
                check("sample_count_after", event["sample_count_after"], len(samples(state)))
                check(
                    "scale_after",
                    event["scale_after"],
                    applied_scale(state["trajectory"], now, TWO_WAVE_RATE),
                )
                check("target_scale_after", event["target_scale_after"], state["trajectory"][1])
            check("bank_index", event["calibration_bank_index"], state["bank"])
            check("history_revision", event["calibration_history_revision"], state["revision"])
            peak_samples = max(peak_samples, len(samples(state)))
            if keep_records:
                records.append(
                    dict(
                        key=key,
                        event=event,
                        samples=samples(state),
                        removed=removed,
                        bank_index=state["bank"],
                        history_revision=state["revision"],
                        fitted_revision=state["fitted_revision"],
                        previous_sample_count=len(state["previous"]),
                        current_sample_count=len(state["current"]),
                    )
                )
        except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
            errors.append(["two_wave_trace_mismatch", event.get("request_id", kind), str(error)])
            break
    if not errors and any(
        states != {"capacity_ready", "capacity_unready"} for states in rotation_states.values()
    ):
        errors.append(["missing_rotation_state_record"])
    result = dict(
        ok=not errors,
        errors=errors,
        error_count=len(errors),
        calibration_method=method,
        selected_forecasts=len(selected),
        gate_forecasts=len(gates),
        feedback=counts["feedback"],
        accepted=counts["accepted"],
        poll_fits=counts["poll_fits"],
        rotations=counts["rotations"],
        bank_rotations=len(rotation_states),
        removed_samples=counts["removed_samples"],
        feedback_reasons={k[7:]: v for k, v in counts.items() if k.startswith("reason:")},
        states=len(states),
        scale_units_per_second=TWO_WAVE_RATE,
        bank_seconds=300,
        bank_max_samples=32,
        max_combined_samples=64,
        peak_sample_count=peak_samples,
        elapsed_time_limited=True,
        native_queue_eligibility_required=True,
        revision_semantics=(
            "Accepted feedback increments history revision; removal increments "
            "it once. Poll fits key by history/fitted revision because "
            "accepted/fit sequence can repeat on removal-only fits."
        ),
        final_states=[
            dict(
                worker=k[0],
                calibration_epoch=k[1],
                calibration_state=k[2],
                bank_index=s["bank"],
                history_revision=s["revision"],
                fitted_revision=s["fitted_revision"],
                previous_sample_count=len(s["previous"]),
                current_sample_count=len(s["current"]),
            )
            for k, s in states.items()
        ],
    )
    if queue_only:
        result.update(
            calibration_input="saved queue_excess_seconds N, excluding empirical floor and visibility",
            baseline_transitions=counts["baseline_transitions"],
            material_baseline_promotions=counts["baseline_promotions"],
            forecast_baseline_sources={
                source: counts["baseline_source:" + source]
                for source in ("unknown", "provisional", "measured")
            },
        )
    return result, records


def audit_two_wave(log):
    return two_wave_trace(log)[0]


def audit(log):
    """Replay windows/trajectories by observation and poll-fit sequence.

    Unmarked historical traces retain their feedback-time median semantics.
    """
    updates = defaultdict(dict)
    selected = []
    gate_forecasts = []
    feedback = []
    errors = []
    methods = {
        e.get("calibration_method", "median_feedback")
        for e in log
        if e.get("event")
        in (
            "calibration_forecast",
            "calibration_feedback",
            "calibration_update",
            "calibration_rotation",
            "calibration_baseline",
        )
        or e.get("event") == "forecast"
        and e.get("calibration_enabled")
    }
    if any(
        ("calibration_bank_index" in e or "calibration_history_revision" in e)
        and e.get("calibration_method") not in TWO_WAVE_METHODS
        for e in log
        if e.get("event")
        in (
            "calibration_forecast",
            "calibration_feedback",
            "calibration_update",
            "calibration_rotation",
        )
        or e.get("event") == "forecast"
        and e.get("calibration_enabled")
    ):
        return dict(
            ok=False,
            errors=[["mixed_or_unknown_calibration_methods", sorted(methods, key=str)]],
            error_count=1,
            calibration_method="mixed_or_unknown",
        )
    if len(methods) == 1 and methods <= set(TWO_WAVE_METHODS):
        return audit_two_wave(log)
    poll_fitted = "mse_poll" in methods
    if methods - {"median_feedback", "mse_poll"} or len(methods) > 1:
        return dict(
            ok=False,
            errors=[["mixed_or_unknown_calibration_methods", sorted(methods, key=str)]],
            error_count=1,
            calibration_method="mixed_or_unknown",
        )
    limited = any("calibration_time_seconds" in e for e in log)
    eligibility_required = any(
        e.get("event") == "calibration_forecast"
        and "native_queue_visible" in e
        or e.get("event") == "calibration_feedback"
        and "admission_native_queue_visible" in e
        for e in log
    )
    for event in log:
        if event.get("event") == "calibration_forecast":
            selected.append(event)
        if event.get("event") == "forecast" and event.get("calibration_enabled"):
            gate_forecasts.append(event)
        is_fit = event.get("event") == "calibration_update"
        if event.get("event") != "calibration_feedback" and not is_fit:
            continue
        if poll_fitted and any(
            field not in event for field in ("fit_sequence_before", "fit_sequence_after")
        ):
            errors.append(["missing_fit_sequence_fields", event.get("request_id", "poll")])
        if not is_fit:
            feedback.append(event)
        if is_fit or event["reason"] == "accepted":
            key = (event["worker"], event["current_calibration_epoch"], event["calibration_state"])
            sequence = (event["sequence_after"], event.get("fit_sequence_after", 0))
            if sequence in updates[key]:
                errors.append(["duplicate_accepted_sequence", key, sequence])
            updates[key][sequence] = dict(event, request_id=event.get("request_id", "poll"))
    versions = {}
    feedback_times = {}
    largest_rate = 0.0
    largest_same_time_change = 0.0
    largest_instant_update = 0.0
    busiest_second = 0
    for key, changes in updates.items():
        history = deque(maxlen=32)
        previous = (1.0, 1.0, 0.0, 0)  # applied anchor, target, epoch-relative time, count
        versions[key + (0, 0)] = previous
        prior_sequence = 0
        prior_fit = 0
        recent_times = deque()
        for (sequence, fit_sequence), event in sorted(changes.items()):
            is_fit = event["event"] == "calibration_update"
            if event.get("fit_sequence_before", 0) != prior_fit:
                errors.append(["missing_fit_sequence", key, sequence, fit_sequence])
            if is_fit:
                if (
                    not poll_fitted
                    or sequence != prior_sequence
                    or event["sequence_before"] != sequence
                    or fit_sequence != sequence
                    or fit_sequence <= prior_fit
                ):
                    errors.append(["invalid_poll_fit_sequence", key, sequence, fit_sequence])
            else:
                if sequence != event["sequence_before"] + 1 or sequence != prior_sequence + 1:
                    errors.append(["missing_history_sequence", key, sequence])
                if fit_sequence != prior_fit:
                    errors.append(["feedback_changed_fit_sequence", event["request_id"]])
                prior_sequence = sequence
            if event["calibration_epoch"] != event["current_calibration_epoch"]:
                errors.append(["accepted_old_epoch", event["request_id"]])
            now = event["calibration_time_seconds"] if limited else float(sequence)
            before = applied_scale(previous, now) if limited else previous[1]
            if limited and now < previous[2]:
                errors.append(["monotonic_time_regressed", event["request_id"]])
            if not close(event["scale_before"], before) or event["sample_count_before"] != len(
                history
            ):
                errors.append(["incorrect_feedback_before", event["request_id"]])
            if limited and not close(event["target_scale_before"], previous[1]):
                errors.append(["incorrect_target_before", event["request_id"]])
            if not is_fit:
                ratio = (
                    max(event["ttft_seconds"] - event["baseline_seconds"], 0)
                    / event["raw_excess_seconds"]
                )
                history.append((ratio, event["raw_excess_seconds"]))
            target = (
                mse_scale(history)
                if is_fit
                else previous[1]
                if poll_fitted
                else median_scale(history)
            )
            after = before if limited else target
            if (
                not close(after, event["scale_after"])
                or len(history) != event["sample_count_after"]
            ):
                errors.append(["incorrect_feedback_after", event["request_id"]])
            if limited:
                if not is_fit:
                    recent_times.append(now)
                while recent_times and now - recent_times[0] >= 1:
                    recent_times.popleft()
                busiest_second = max(busiest_second, len(recent_times))
                largest_instant_update = max(
                    largest_instant_update, abs(event["scale_after"] - event["scale_before"])
                )
                if not close(target, event["target_scale_after"]):
                    errors.append(["incorrect_target_after", event["request_id"]])
                dt = now - previous[2]
                movement = abs(after - previous[0])
                if movement > SCALE_UNITS_PER_SECOND * max(dt, 0) + 2e-8:
                    errors.append(["elapsed_time_rate_exceeded", event["request_id"]])
                if dt > 1e-8:
                    largest_rate = max(largest_rate, movement / dt)
                else:
                    largest_same_time_change = max(largest_same_time_change, movement)
            previous = (after, target, now, len(history))
            versions[key + (sequence, fit_sequence)] = previous
            prior_fit = fit_sequence
            if event.get("timestamp"):
                feedback_times[key + (sequence, fit_sequence)] = datetime.fromisoformat(
                    event["timestamp"].replace("Z", "+00:00")
                )
            if not is_fit and event["admission_observation_sequence"] > event["sequence_before"]:
                errors.append(["feedback_uses_own_or_future_observation", event["request_id"]])
    for event in selected + gate_forecasts:
        if poll_fitted and "calibration_fit_sequence" not in event:
            errors.append(["missing_selected_fit_sequence", event["request_id"]])
        seq = event["calibration_observation_sequence"]
        fit_seq = event.get("calibration_fit_sequence", 0)
        key = (
            event["worker"],
            event["calibration_epoch"],
            event["calibration_state"],
            seq,
            fit_seq,
        )
        expected = versions.get(key, (1.0, 1.0, 0.0, 0) if seq == fit_seq == 0 else None)
        now = event["calibration_time_seconds"] if limited else float(seq)
        applied = (
            applied_scale(expected, now)
            if limited and expected
            else expected[1]
            if expected
            else None
        )
        if (
            expected is None
            or not close(applied, event["calibration_scale"])
            or expected[3] != event["calibration_sample_count"]
        ):
            errors.append(["incorrect_selected_history", event["request_id"], event["event"], key])
        if limited and expected:
            if now < expected[2] or not close(expected[1], event["calibration_target_scale"]):
                errors.append(
                    ["incorrect_selected_target_or_time", event["request_id"], event["event"]]
                )
        elif key in feedback_times and event.get("timestamp"):
            if feedback_times[key] > datetime.fromisoformat(
                event["timestamp"].replace("Z", "+00:00")
            ):
                errors.append(["forecast_uses_future_feedback", event["request_id"]])
        base = event["baseline_seconds"]
        extra = event["predicted_excess_seconds"]
        if not close(event["predicted_ttft_seconds"], base + extra) or not close(
            event["calibrated_ttft_seconds"], base + event["calibration_scale"] * extra
        ):
            errors.append(["incorrect_forecast_arithmetic", event["request_id"], event["event"]])
        expected_reject = (
            event["ready"]
            and math.isfinite(event["calibrated_excess_seconds"])
            and event["calibrated_excess_seconds"] >= event["threshold_seconds"]
        )
        if expected_reject != event["calibrated_would_reject"]:
            errors.append(["incorrect_calibrated_readiness_gate", event["request_id"]])
        if eligibility_required:
            required = ["native_waiting", "native_queue_visible", "native_ready"]
            if any(field not in event for field in required):
                errors.append(
                    ["missing_native_eligibility_snapshot", event["request_id"], event["event"]]
                )
            else:
                if (
                    type(event["native_ready"]) is not bool
                    or type(event["native_queue_visible"]) is not bool
                ):
                    errors.append(
                        ["invalid_native_eligibility_snapshot", event["request_id"], event["event"]]
                    )
                if event["native_queue_visible"] != (event["native_waiting"] > 0):
                    errors.append(
                        ["incorrect_native_queue_visibility", event["request_id"], event["event"]]
                    )
                # Selected native_ready is saved maturity. The earlier general
                # forecast native_ready additionally includes scrape freshness.
                mature = (
                    event["native_ready"]
                    if event["event"] == "calibration_forecast"
                    else event.get("capacity_learned_ready")
                )
                if (
                    mature is not None
                    and (event["calibration_state"] == "capacity_ready") != mature
                ):
                    errors.append(
                        ["incorrect_native_capacity_state", event["request_id"], event["event"]]
                    )
    selected_by_id = {event["request_id"]: event for event in selected}
    for event in feedback:
        forecast = selected_by_id.get(event["request_id"])
        if not forecast:
            errors.append(["feedback_missing_selected_snapshot", event["request_id"]])
            continue
        inputs = [
            ("baseline_seconds", "baseline_seconds"),
            ("predicted_excess_seconds", "raw_excess_seconds"),
            ("calibration_scale", "admission_scale"),
            ("calibration_observation_sequence", "admission_observation_sequence"),
            ("calibration_epoch", "calibration_epoch"),
        ]
        if poll_fitted:
            inputs.append(("calibration_fit_sequence", "admission_fit_sequence"))
        if limited:
            inputs.append(("calibration_time_seconds", "admission_calibration_time_seconds"))
        if eligibility_required:
            inputs.extend(
                [
                    ("baseline_available", "baseline_available"),
                    ("native_waiting", "admission_native_waiting"),
                    ("native_queue_visible", "admission_native_queue_visible"),
                    ("native_ready", "admission_native_capacity_ready"),
                ]
            )
        for source, target in inputs:
            if source not in forecast or target not in event:
                errors.append(["feedback_missing_saved_input", event["request_id"], source])
            elif not close(forecast[source], event[target]):
                errors.append(["feedback_changed_saved_input", event["request_id"], source])
        if eligibility_required:
            required = [
                "admission_native_waiting",
                "admission_native_queue_visible",
                "admission_native_capacity_ready",
            ]
            if any(field not in event for field in required):
                errors.append(["missing_native_eligibility_feedback", event["request_id"]])
            else:
                if (
                    type(event["admission_native_capacity_ready"]) is not bool
                    or type(event["admission_native_queue_visible"]) is not bool
                ):
                    errors.append(["invalid_native_eligibility_feedback", event["request_id"]])
                if event["admission_native_queue_visible"] != (
                    event["admission_native_waiting"] > 0
                ):
                    errors.append(["incorrect_admission_queue_visibility", event["request_id"]])
                if event["reason"] == "accepted" and not (
                    event["admission_native_capacity_ready"]
                    and event["admission_native_queue_visible"]
                ):
                    errors.append(["accepted_ineligible_observation", event["request_id"]])
            expected_reason = feedback_reason(event, True)
            if event["reason"] != expected_reason:
                errors.append(
                    [
                        "incorrect_feedback_reason",
                        event["request_id"],
                        event["reason"],
                        expected_reason,
                    ]
                )
        if event["reason"] != "accepted":
            if event.get("fit_sequence_before", 0) != event.get("fit_sequence_after", 0):
                errors.append(["skipped_feedback_changed_fit_sequence", event["request_id"]])
            if (
                event["sequence_before"] != event["sequence_after"]
                or event["sample_count_before"] != event["sample_count_after"]
            ):
                errors.append(["skipped_feedback_changed_history", event["request_id"]])
            if (
                not close(event["scale_before"], event["scale_after"])
                or limited
                and not close(event["target_scale_before"], event["target_scale_after"])
            ):
                errors.append(["skipped_feedback_changed_trajectory", event["request_id"]])
            sequence = event["sequence_before"]
            fit_seq = event.get("fit_sequence_before", 0)
            key = (
                event["worker"],
                event["current_calibration_epoch"],
                event["calibration_state"],
                sequence,
                fit_seq,
            )
            version = versions.get(key, (1.0, 1.0, 0.0, 0) if sequence == fit_seq == 0 else None)
            now = event["calibration_time_seconds"] if limited else float(sequence)
            expected_scale = (
                applied_scale(version, now)
                if version is not None and limited
                else version[1]
                if version
                else None
            )
            if (
                version is None
                or version[3] != event["sample_count_before"]
                or not close(expected_scale, event["scale_before"])
            ):
                errors.append(["incorrect_skipped_history", event["request_id"]])
            elif limited and (
                now < version[2] or not close(version[1], event["target_scale_before"])
            ):
                errors.append(["incorrect_skipped_target_or_time", event["request_id"]])
    return dict(
        ok=not errors,
        errors=errors[:100],
        error_count=len(errors),
        selected_forecasts=len(selected),
        gate_forecasts=len(gate_forecasts),
        feedback=len(feedback),
        accepted=sum(e["reason"] == "accepted" for e in feedback),
        poll_fits=sum(e.get("event") == "calibration_update" for e in log),
        calibration_method="mse_poll" if poll_fitted else "median_feedback",
        feedback_reasons=dict(Counter(event["reason"] for event in feedback)),
        history_versions=len(versions),
        states=len(updates),
        elapsed_time_limited=limited,
        native_queue_eligibility_required=eligibility_required,
        scale_units_per_second=SCALE_UNITS_PER_SECOND if limited else None,
        largest_observed_feedback_scale_rate=largest_rate if limited else None,
        largest_same_instant_feedback_movement=largest_same_time_change if limited else None,
        largest_instant_update_movement=largest_instant_update if limited else None,
        peak_accepted_feedback_in_one_second=busiest_second if limited else None,
    )


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def score(rows):
    result = {"n": len(rows)}
    if not rows:
        return result
    result["actual_mean"] = statistics.mean(r["actual_ttft"] for r in rows)
    result["actual_p95"] = percentile([r["actual_ttft"] for r in rows], 0.95)
    result["client_minus_smg_p50_p95"] = [
        percentile([r["client_ttft"] - r["actual_ttft"] for r in rows], q) for q in [0.5, 0.95]
    ]
    for name in ["raw", "target", "calibrated"]:
        errors = [r[name + "_ttft"] - r["actual_ttft"] for r in rows]
        counts = Counter()
        for r in rows:
            late = r["actual_extra"] >= 5
            reject = r[name + "_would_reject"]
            counts["tp" if late and reject else "fn" if late else "fp" if reject else "tn"] += 1
        positives = counts["tp"] + counts["fn"]
        negatives = counts["fp"] + counts["tn"]
        result[name] = dict(
            mean=statistics.mean(r[name + "_ttft"] for r in rows),
            bias=statistics.mean(errors),
            mae=statistics.mean(abs(e) for e in errors),
            p95_abs_error=percentile([abs(e) for e in errors], 0.95),
            **{key: counts[key] for key in ["tp", "fn", "fp", "tn"]},
            positives=positives,
            negatives=negatives,
            fnr=counts["fn"] / positives if positives else None,
            fpr=counts["fp"] / negatives if negatives else None,
            ready_fn=sum(
                r["ready"] and r["actual_extra"] >= 5 and not r[name + "_would_reject"]
                for r in rows
            ),
            unready_fn=sum(not r["ready"] and r["actual_extra"] >= 5 for r in rows),
        )
    result["target_scale_p10_p50_p90"] = [
        percentile([r["target_scale"] for r in rows], q) for q in [0.1, 0.5, 0.9]
    ]
    result["max_target_scale"] = max(r["target_scale"] for r in rows)
    result["max_applied_scale"] = max(r["scale"] for r in rows)
    result["unready"] = sum(not r["ready"] for r in rows)
    result["calibration_mature"] = sum(r["calibration_count"] >= 4 for r in rows)
    result["baseline_missing"] = sum(not r["baseline_available"] for r in rows)
    result["scale_p10_p50_p90"] = [
        percentile([r["scale"] for r in rows], q) for q in [0.1, 0.5, 0.9]
    ]
    result["native_queue_eligibility_observed"] = sum(
        r.get("native_queue_eligibility_observed", False) for r in rows
    )
    result["native_queue_learning_eligible"] = sum(
        r.get("native_queue_learning_eligible") is True for r in rows
    )
    return result


def analyze(path):
    requests = list(csv.DictReader((path / "requests.tsv").open(), delimiter="\t"))
    metrics = list(csv.DictReader((path / "metrics.tsv").open(), delimiter="\t"))

    def clean_metric(key):
        return [v for row in metrics if math.isfinite(v := float(row.get(key) or "nan")) and v >= 0]

    sample_times = clean_metric("elapsed_seconds")
    loads = {
        key: max(clean_metric(key), default=None)
        for key in ["waiting", "running", "kv_usage", "pending_first_text", "effective_capacity"]
    }
    loads["samples"] = len(metrics)
    loads["max_sample_gap_seconds"] = max(
        (b - a for a, b in zip(sample_times, sample_times[1:])), default=None
    )
    summary = json.loads((path / "summary.json").read_text())
    log = list(events(path / "router-events.jsonl"))
    causality = audit(log)
    prefix = "classifier-" + str(summary["seed"]) + "-"
    by_event = defaultdict(dict)
    duplicates = Counter()
    for event in log:
        request_id = event.get("request_id", "")
        name = event.get("event")
        if not request_id.startswith(prefix) or name not in [
            "forecast",
            "calibration_forecast",
            "first_text",
            "calibration_feedback",
        ]:
            continue
        if request_id in by_event[name]:
            duplicates[name] += 1
        by_event[name][request_id] = event
    baselines = defaultdict(list)

    def shape(row):
        return (
            "cached"
            if row["case"] in ["baseline-cached", "cached-burst", "cached-burst-warm"]
            else row["prompt_words"] + "-out" + row["output_tokens"]
        )

    for row in requests:
        text = by_event["first_text"].get(row["opc_request_id"])
        if row["case"].startswith("baseline-") and text:
            baselines[shape(row)].append(text["ttft_seconds"])
    matched = {key: statistics.median(value) for key, value in baselines.items()}
    rows = []
    rejected = []
    missing = []
    for row in requests:
        rid = row["opc_request_id"]
        gate = by_event["forecast"].get(rid)
        if row["status"] == "503":
            rejected.append(dict(case=row["case"], request_id=rid, gate=gate, error=row["error"]))
            continue
        if row["status"] != "200":
            missing.append([rid, "HTTP " + row["status"]])
            continue
        forecast = by_event["calibration_forecast"].get(rid)
        text = by_event["first_text"].get(rid)
        feedback = by_event["calibration_feedback"].get(rid)
        if not all([forecast, text, feedback, gate]):
            missing.append([rid, "missing joined event"])
            continue
        baseline = matched[shape(row)]
        native_waiting = forecast.get("native_waiting", gate.get("native_waiting"))
        native_ready = forecast.get(
            "native_ready", forecast["calibration_state"] == "capacity_ready"
        )
        native_queue_visible = forecast.get("native_queue_visible")
        eligibility_observed = all(
            key in forecast for key in ["native_waiting", "native_queue_visible", "native_ready"]
        )
        rows.append(
            dict(
                case=row["case"],
                request_id=rid,
                started_elapsed_seconds=float(row["started_elapsed_seconds"]),
                prompt_words=int(row["prompt_words"]),
                actual_ttft=text["ttft_seconds"],
                client_ttft=float(row["ttft_seconds"]),
                actual_extra=max(text["ttft_seconds"] - baseline, 0),
                matched_baseline=baseline,
                admission_baseline=forecast["baseline_seconds"],
                baseline_available=forecast["baseline_available"],
                raw_ttft=forecast["predicted_ttft_seconds"],
                raw_extra=forecast["predicted_excess_seconds"],
                calibrated_ttft=forecast["calibrated_ttft_seconds"],
                calibrated_extra=forecast["calibrated_excess_seconds"],
                scale=forecast["calibration_scale"],
                target_scale=forecast.get(
                    "calibration_target_scale", forecast["calibration_scale"]
                ),
                calibration_time_seconds=forecast.get("calibration_time_seconds"),
                target_ttft=forecast["baseline_seconds"]
                + forecast.get("calibration_target_scale", forecast["calibration_scale"])
                * forecast["predicted_excess_seconds"],
                target_would_reject=bool(
                    forecast["ready"]
                    and forecast.get("calibration_target_scale", forecast["calibration_scale"])
                    * forecast["predicted_excess_seconds"]
                    >= forecast["threshold_seconds"]
                ),
                calibration_count=forecast["calibration_sample_count"],
                calibration_sequence=forecast["calibration_observation_sequence"],
                calibration_state=forecast["calibration_state"],
                calibration_epoch=forecast["calibration_epoch"],
                native_waiting=native_waiting,
                native_capacity_ready=native_ready,
                native_queue_visible=native_queue_visible,
                native_queue_eligibility_observed=eligibility_observed,
                native_queue_learning_eligible=bool(native_ready and native_queue_visible)
                if eligibility_observed
                else None,
                ready=forecast["ready"],
                raw_would_reject=forecast["would_reject"],
                calibrated_would_reject=forecast["calibrated_would_reject"],
                gate_would_reject=gate["would_reject"],
                gate_excess=gate["predicted_excess_seconds"],
                feedback_reason=feedback["reason"],
                effective_capacity=gate["effective_capacity"],
                queue_tokens=gate["queued_tokens"],
                dispatch_tokens=gate["dispatched_tokens_since_poll"],
                observed_delay_floor=gate["observed_delay_floor"],
                native_queue_delay=gate["native_queue_delay"],
            )
        )
    checks = dict(
        causal_replay=causality["ok"],
        no_duplicate_events=not duplicates,
        all_successes_joined=not missing,
        all_selected_receive_feedback=len(rows) == len(by_event["calibration_feedback"]),
        all_shadow_requests_200=all(row["status"] == "200" for row in requests)
        if summary["suite"] != "hard"
        else True,
        all_503_from_raw_gate=all(
            r["gate"]
            and r["gate"]["would_reject"]
            and r["gate"]["ready"]
            and r["gate"]["predicted_excess_seconds"] >= r["gate"]["threshold_seconds"]
            and "estimated wait budgets" in r["error"]
            for r in rejected
        ),
        hard_healthy_recovery_no_rejections=not any(
            r["case"] in ["hard-low", "hard-recovery"] for r in rejected
        ),
        raw_gate_budget_arithmetic=all(
            e["would_reject"]
            == bool(e["ready"] and e["predicted_excess_seconds"] >= e["threshold_seconds"])
            for e in by_event["forecast"].values()
        ),
    )
    nonbaseline = [row for row in rows if not row["case"].startswith("baseline-")]
    low_names = [
        "sparse-short",
        "sparse-long",
        "cached-burst",
        "cached-burst-warm",
        "recovery-sparse",
        "heldout-low",
        "heldout-recovery",
        "cold-recovery-sparse",
        "hard-low",
        "hard-recovery",
    ]
    low = [row for row in nonbaseline if row["case"] in low_names]
    high = [row for row in nonbaseline if row["case"] not in low_names]
    result = dict(
        suite=summary["suite"],
        engine=summary["engine"],
        image=summary["image"],
        requests=len(requests),
        pairs=len(rows),
        missing=missing,
        duplicates=dict(duplicates),
        checks=checks,
        sampled_load=loads,
        causal_audit=causality,
        baselines=matched,
        baseline_counts={k: len(v) for k, v in baselines.items()},
        feedback_reasons=dict(Counter(r["feedback_reason"] for r in rows)),
        rejections=len(rejected),
        rejections_by_phase=dict(Counter(r["case"] for r in rejected)),
        raw_rejections_calibrated_gate_would_permit=sum(
            not r["gate"].get("calibrated_would_reject", False) for r in rejected if r["gate"]
        ),
        all=score(nonbaseline),
        low=score(low),
        high=score(high),
        phases={
            case: score([r for r in rows if r["case"] == case])
            for case in dict.fromkeys(r["case"] for r in rows)
        },
        states={
            state: score([r for r in nonbaseline if r["calibration_state"] == state])
            for state in ["capacity_unready", "capacity_ready"]
        },
        notes=[
            (
                "Raw/calibrated accuracy uses the same selected snapshot. "
                "Original pre-admission gate is audited separately."
            ),
            (
                "Truth is prompt-shape-matched median low-load first-text "
                "TTFT plus the shared five-second extra-delay budget."
            ),
            "Rejected hard-mode requests have no observed counterfactual TTFT. Calibrated gate remains hypothetical.",
            "The holdout is later unseen traffic with online updates enabled; it is not a frozen fit.",
            (
                "Native queue learning eligibility uses the selected admission "
                "snapshot, never queue or capacity measured at response completion. "
                "Historical traces without the added fields do not claim "
                "eligibility verification."
            ),
            (
                "Target is the recorded unrestricted fitting target: MSE "
                "at the last poll for mse_poll/mse_poll_two_wave_v1, weighted "
                "median for historical traces. Applied scale is the live "
                "elapsed-time-limited trajectory. Neither is an offline fit."
            ),
        ],
    )
    (path / "paired-calibration.json").write_text(json.dumps(rows, indent=2) + "\n")
    (path / "calibration-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                path=str(path),
                checks=checks,
                pairs=len(rows),
                rejections=len(rejected),
                low=result["low"],
                high=result["high"],
            ),
            indent=2,
        )
    )
    if not all(checks.values()):
        raise RuntimeError("Correlation, causality or operational audit failed; preserve evidence")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    analyze(parser.parse_args().input)
