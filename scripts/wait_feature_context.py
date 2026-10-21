"""Validate the experimental feature-gain telemetry without inventing labels."""
from __future__ import annotations

import json
import math
from collections import Counter


METHOD = "feature_context_ridge_v1"
CURRENT_METHOD = "feature_context_deadline_ridge_v2"
METHODS = frozenset({METHOD, CURRENT_METHOD})
FEATURES = ("kv_usage", "pretext_cutoff_rate", "actual_rejection_rate")
CURRENT_FEATURES = ("kv_usage", "deadline_or_cutoff_rate", "actual_rejection_rate")
COUNTS = ("first_text", "upstream_timeout", "pretext_downstream_disconnect",
          "unknown_pretext_end", "admitted", "actual_rejected", "untracked_admitted")
DEADLINE_COUNTS = ("deadline_timely_first_text", "deadline_breach",
                   "deadline_or_cutoff_adverse", "deadline_or_cutoff_unknown")
REQUEST_EVENTS = frozenset({"feature_context_forecast", "feature_context_feedback",
                            "feature_outcome", "feature_outcome_ignored", "feature_admission_decision",
                            "feature_deadline_started", "feature_deadline_resolution", "feature_deadline_ignored"})


def number(value, name, *, nullable=False, maximum=None):
    if value is None and nullable:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
            or maximum is not None and value > maximum):
        raise ValueError("Invalid feature number: " + name)
    return value


def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError("Invalid feature counter: " + name)
    return value


def boolean(value, name):
    if type(value) is not bool:
        raise ValueError("Invalid feature flag: " + name)
    return value


def coefficient(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or abs(value) > 2):
        raise ValueError("Invalid bounded feature coefficient: " + name)
    return value


def clock(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Invalid feature clock: " + name)
    return value


def vector(value, name, validate):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("Expected three feature components: " + name)
    return [validate(v, name + "[" + str(i) + "]") for i, v in enumerate(value)]


def close(value, expected, name):
    if not math.isclose(value, expected, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("Feature arithmetic mismatch: " + name)


def payload(event):
    if event.get("feature_method") not in METHODS:
        raise ValueError("Unknown feature telemetry method")
    value = event.get("feature_json")
    if not isinstance(value, str):
        raise ValueError("Feature telemetry must contain serialized feature_json")
    try:
        value = json.loads(value, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid feature_json") from error
    if not isinstance(value, dict):
        raise ValueError("Feature telemetry payload is not an object")
    return value


def validate_context(context, method=None):
    integer(context["bucket_second"], "context.bucket_second")
    clock(context["activity_origin_time_seconds"], "activity origin")
    values = vector(context["values"], "context.values",
                    lambda v, n: number(v, n, nullable=True, maximum=1))
    if method is None:
        method = CURRENT_METHOD if "deadline_untracked_pending" in context or any(k in context["counts"] for k in DEADLINE_COUNTS) else METHOD
    if method not in METHODS:
        raise ValueError("Unknown feature context method")
    names = COUNTS + DEADLINE_COUNTS if method == CURRENT_METHOD else COUNTS
    if set(context["counts"]) != set(names):
        raise ValueError("Unexpected feature counter schema")
    counts = {name: integer(context["counts"][name], name) for name in names}
    if method == CURRENT_METHOD:
        pending_unknown = integer(context["deadline_untracked_pending"], "deadline_untracked_pending")
        cutoff = counts["deadline_or_cutoff_adverse"]
        known = counts["deadline_timely_first_text"] + cutoff
        unknown = counts["deadline_or_cutoff_unknown"]
    else:
        if "deadline_untracked_pending" in context:
            raise ValueError("Legacy context cannot contain deadline-v2 state")
        pending_unknown = 0
        cutoff = counts["upstream_timeout"] + counts["pretext_downstream_disconnect"]
        known = counts["first_text"] + cutoff
        unknown = counts["unknown_pretext_end"]
    resolved = known + unknown
    attempts = counts["admitted"] + counts["actual_rejected"]
    expected_cutoff = (cutoff / known if known >= 16 and not counts["untracked_admitted"]
                       and not pending_unknown and unknown < .1 * resolved else None)
    expected_rejection = counts["actual_rejected"] / attempts if attempts >= 16 else None
    for actual, expected, name in [(values[1], expected_cutoff, "cutoff rate/outcome coverage"),
                                   (values[2], expected_rejection, "actual rejection rate/admission coverage")]:
        if actual is None or expected is None:
            if actual != expected:
                raise ValueError("Feature rate availability mismatch: " + name)
        else:
            close(actual, expected, name)
    return values, counts


def audit_method(events, expected):
    """Historical parsing support must not relax a fresh run's frozen method."""
    if expected not in METHODS:
        raise ValueError("Unknown expected feature method")
    captured = [(i, e) for i, e in enumerate(events) if str(e.get("event", "")).startswith("feature_")]
    errors = [dict(event_index=i, event_kind=e.get("event"), request_id=e.get("request_id"),
                   captured_method=e.get("feature_method"), expected_method=expected)
              for i, e in captured if e.get("feature_method") != expected]
    return dict(ok=bool(captured) and not errors, expected_method=expected,
                event_count=len(captured), errors=errors)


def forecast(event, *, gate=None, ttft_deadline_seconds=300):
    """Keep TOTAL-TTFT comparisons separate from the recorded EXTRA flags."""
    if event.get("event") != "feature_context_forecast":
        raise ValueError("Expected feature admission forecast")
    value = payload(event)
    if value.get("method") != event.get("feature_method"):
        raise ValueError("Unknown feature payload method")
    context, counts = validate_context(value["context"], event["feature_method"])
    fit, prediction = value["fit"], value["prediction"]
    support = vector(fit["support"], "fit.support", boolean)
    active = vector(prediction["active"], "prediction.active", boolean)
    beta = vector(fit["beta"], "fit.beta", coefficient)
    mean = vector(fit["mean"], "fit.mean", lambda v, n: number(v, n, maximum=1))
    minimum = vector(fit["minimum"], "fit.minimum", lambda v, n: number(v, n, maximum=1))
    maximum = vector(fit["maximum"], "fit.maximum", lambda v, n: number(v, n, maximum=1))
    integer(fit["count"], "fit.count")
    integer(fit["revision"], "fit.revision")
    ess = number(fit["ess"], "fit.ess")
    if any(support) and (fit["count"] < 12 or ess < 8):
        raise ValueError("Supported feature fit lacks minimum evidence")
    for j in range(3):
        if support[j] and not minimum[j] <= mean[j] <= maximum[j]:
            raise ValueError("Feature mean lies outside retained support")
        if support[j] and maximum[j] - minimum[j] < .05 - 1e-10:
            raise ValueError("Supported feature lacks minimum range")
        if not support[j] and beta[j] != 0:
            raise ValueError("Unsupported feature has a non-neutral coefficient")
        expected = support[j] and context[j] is not None and minimum[j] <= context[j] <= maximum[j]
        if active[j] != expected:
            raise ValueError("Feature active mask differs from retained support")
    gain = number(prediction["gain"], "prediction.gain")
    target = number(prediction["target_gain"], "prediction.target_gain")
    contextual = boolean(prediction["contextual"], "prediction.contextual")
    if prediction["reason"] not in {"snapshot_unavailable", "context_supported", "legacy_fallback_no_current_support"}:
        raise ValueError("Unknown feature fallback reason")
    legacy = number(value["legacy_gain"], "legacy_gain")
    number(value["legacy_target_gain"], "legacy_target_gain")
    fresh = boolean(value["fresh"], "fresh")
    ready = boolean(value["native_ready"], "native_ready")
    scoped = boolean(value["supported_path"], "supported_path")
    if contextual:
        if not (fresh and ready and scoped and any(active)):
            raise ValueError("Context trajectory active without usable support")
        anchor = number(prediction["trajectory_anchor_gain"], "trajectory anchor gain")
        anchor_time = clock(prediction["trajectory_anchor_time_seconds"], "trajectory anchor time")
        elapsed = clock(event["diagnostic_time_seconds"], "admission time") - anchor_time
        if not math.isfinite(elapsed) or elapsed < -1e-8:
            raise ValueError("Feature trajectory anchor follows admission")
        change = min(abs(target - anchor), .5 * max(0, elapsed))
        expected_gain = anchor + change if target >= anchor else anchor - change
        close(gain, expected_gain, "logged trajectory gain")
        if prediction["reason"] != "context_supported":
            raise ValueError("Contextual trajectory has a fallback reason")
    else:
        close(gain, legacy, "legacy fallback gain")
        close(target, legacy, "legacy fallback target")
        if (prediction["trajectory_anchor_gain"] is not None
                or prediction["trajectory_anchor_time_seconds"] is not None):
            raise ValueError("Legacy fallback contains a feature trajectory")
    n = number(value["original_queue_excess_seconds"], "original N")
    baseline = number(value["baseline_seconds"], "baseline", nullable=True)
    extra = number(value["extra_seconds"], "feature extra", nullable=True)
    total = number(value["total_ttft_seconds"], "feature total", nullable=True)
    threshold = number(value["threshold_seconds"], "threshold")
    if threshold <= 0:
        raise ValueError("Feature threshold must be positive")
    expected_extra = gain * n
    expected_extra = expected_extra if math.isfinite(expected_extra) else None
    if expected_extra is None or extra is None:
        if extra != expected_extra:
            raise ValueError("Invalid unavailable feature extra")
    else:
        close(extra, expected_extra, "gain times original N")
    expected_total = baseline + extra if baseline is not None and extra is not None else None
    if expected_total is not None and not math.isfinite(expected_total):
        expected_total = None
    if total is None or expected_total is None:
        if total != expected_total:
            raise ValueError("Feature total differs from baseline/finite arithmetic availability")
    else:
        close(total, expected_total, "baseline plus extra")
    usable = fresh and ready and scoped and extra is not None and total is not None
    if boolean(value["prediction_available"], "prediction_available") != usable:
        raise ValueError("Feature prediction availability differs from its inputs")
    for name, expected in [("extra_would_reject", extra >= threshold if usable else None),
                           ("total_would_reject", total >= threshold if usable else None)]:
        actual = value[name]
        if actual is not None:
            boolean(actual, name)
        if actual != expected:
            raise ValueError("Feature decision mismatch: " + name)
    for name, expected in [("ridge", .1), ("max_delta", .5), ("max_abs_coefficient", 2)]:
        close(number(value[name], name), expected, name)
    if gate is not None:
        if event.get("request_id") != gate.get("request_id") or event.get("worker") != gate.get("worker"):
            raise ValueError("Feature forecast lacks the exact original request/worker")
        close(n, gate["calibration_input_excess_seconds"], "original admission N")
        close(legacy, gate["calibration_scale"], "original admission gain")
        close(threshold, gate["threshold_seconds"], "original admission threshold")
        original_b = gate.get("calibration_baseline_seconds") if gate.get("calibration_baseline_available") else None
        if baseline is None or original_b is None:
            if baseline != original_b:
                raise ValueError("Feature baseline availability differs from admission")
        else:
            close(baseline, original_b, "original admission baseline")
    return dict(method=event["feature_method"], total_ttft_seconds=total, extra_seconds=extra,
                total_ttft_over_deadline=None if total is None else total >= ttft_deadline_seconds,
                recorded_extra_would_reject=value["extra_would_reject"],
                recorded_total_would_reject=value["total_would_reject"],
                contextual=contextual, active_features=active, supported_features=support,
                feature_values=context, recent_counts=counts, reason=prediction["reason"],
                native_ready=ready, fresh=fresh, supported_path=scoped,
                prediction_available=usable, snapshot=value)


def audit_activity(events, rows):
    """Bind real decisions and once-only terminal observations to offered IDs."""
    owned = {r["opc_request_id"]: r for r in rows if not r.get("local_drop")}
    decisions, outcomes, errors = {}, {}, []
    for event in events:
        kind = event.get("event")
        if kind not in {"feature_admission_decision", "feature_outcome", "feature_outcome_ignored"}:
            continue
        rid = event.get("request_id")
        if rid not in owned:
            errors.append("Unowned feature activity request: " + str(rid))
            continue
        if kind != "feature_outcome_ignored":
            value = payload(event)
            context = value["activity"]
            _, counts = validate_context(context, event["feature_method"])
            if value["counts"] != counts:
                errors.append("Activity counts differ from the captured bucket context")
        elif (event.get("feature_method") not in METHODS
              or event.get("reason") not in {"generation_changed", "worker_missing"}):
            errors.append("Unknown ignored-outcome provenance")
        if kind == "feature_admission_decision":
            decision = event.get("decision")
            if decision not in {"admitted", "actual_rejected"}:
                errors.append("Unknown actual admission decision")
            boolean(event.get("outcome_tracking_supported"), "outcome_tracking_supported")
            if rid in decisions:
                errors.append("Duplicate actual admission decision: " + rid)
            decisions[rid] = event
        else:
            if event.get("outcome") not in set(COUNTS[:4]):
                errors.append("Unknown terminal outcome category")
            if rid in outcomes:
                errors.append("Duplicate terminal outcome: " + rid)
            outcomes[rid] = event
    for rid, row in owned.items():
        decision = decisions.get(rid)
        if decision is None:
            errors.append("Missing actual admission decision: " + rid)
            continue
        rejected = decision.get("decision") == "actual_rejected"
        if rejected != bool(row.get("controlled_rejection")):
            errors.append("HTTP rejection and actual decision differ: " + rid)
        if rejected:
            if row.get("status") != 503 or row.get("response_error_code") != "worker_overload_protection_shed":
                errors.append("Actual rejection lacks the expected HTTP rejection response: " + rid)
            if rid in outcomes:
                errors.append("Rejected request has an upstream terminal outcome: " + rid)
            if row.get("first_text_late") is not None:
                errors.append("Rejected TTFT truth was invented: " + rid)
        elif not decision["outcome_tracking_supported"]:
            if rid in outcomes:
                errors.append("Unsupported admission invented a terminal identity: " + rid)
        elif rid not in outcomes:
            errors.append("Missing admitted terminal observation: " + rid)
        else:
            outcome = outcomes[rid]
            if outcome.get("worker") != decision.get("worker"):
                errors.append("Terminal worker differs from admission: " + rid)
            if outcome["diagnostic_time_seconds"] < decision["diagnostic_time_seconds"]:
                errors.append("Terminal outcome preceded admission: " + rid)
    counts = Counter(e.get("outcome") for e in outcomes.values() if e.get("event") == "feature_outcome")
    ignored = Counter(e.get("reason") for e in outcomes.values() if e.get("event") == "feature_outcome_ignored")
    actual = sum(e.get("decision") == "actual_rejected" for e in decisions.values())
    admitted = sum(e.get("decision") == "admitted" for e in decisions.values())
    by_phase = {}
    for phase in sorted({r["phase"] for r in rows}):
        subset = [r for r in rows if r["phase"] == phase]
        rejected = sum(decisions.get(r["opc_request_id"], {}).get("decision") == "actual_rejected" for r in subset)
        by_phase[phase] = dict(offered=len(subset), actual_rejected=rejected,
                              actual_rejected_per_offered=rejected / len(subset))
    return dict(ok=not errors, errors=errors, offered=len(rows), actual_decisions=len(decisions),
                selected_admissions=admitted, actual_rejected=actual,
                actual_rejected_per_offered=actual / len(rows) if rows else None,
                actual_rejected_per_decided_attempt=actual / (admitted + actual) if admitted + actual else None,
                terminal_outcomes=dict(counts), known_terminal_outcomes=sum(counts[k] for k in COUNTS[:3]),
                unsupported_tracking_admissions=sum(e.get("decision") == "admitted" and not e["outcome_tracking_supported"]
                                                    for e in decisions.values()),
                ignored_terminal_outcomes=dict(ignored),
                per_phase=by_phase,
                limits="Local observed terminal categories, not native engine cancellation acknowledgement; rejected TTFT truth unknown.")


def audit_windows(events):
    """Replay the recorded router-local buckets, including each event's own count."""
    history, generations, errors, snapshots, activity_events = {}, {}, [], 0, 0
    for index, event in enumerate(events):
        kind = event.get("event")
        worker = event.get("worker")
        if kind == "calibration_reset":
            epoch = integer(event["calibration_epoch"], "reset epoch")
            previous = generations.get(worker)
            if previous is not None and epoch != previous["epoch"] + 1:
                errors.append(dict(event_index=index, reason="Unexplained calibration reset epoch"))
            generations[worker] = dict(epoch=epoch, origin=None)
            continue
        if kind not in {"feature_context_forecast", "feature_context_poll",
                        "feature_admission_decision", "feature_outcome",
                        "feature_deadline_started", "feature_deadline_resolution"}:
            continue
        value = payload(event)
        context = value["context"] if kind.startswith("feature_context_") else value["activity"]
        _, captured = validate_context(context, event["feature_method"])
        names = tuple(captured)
        second = context["bucket_second"]
        origin = context["activity_origin_time_seconds"]
        elapsed = clock(event["diagnostic_time_seconds"], "diagnostic time") - origin
        if not second - 1e-8 <= elapsed < second + 1 + 1e-8:
            errors.append(dict(event_index=index, reason="Activity bucket/clock mismatch"))
        epoch = integer(event["calibration_epoch"], "activity epoch")
        generation = generations.setdefault(worker, dict(epoch=epoch, origin=None))
        if epoch != generation["epoch"]:
            errors.append(dict(event_index=index, reason="Activity epoch changed without captured reset"))
        if generation["origin"] is None:
            generation["origin"] = origin
        elif origin != generation["origin"]:
            errors.append(dict(event_index=index, reason="Activity origin changed without captured reset"))
        key = (worker, epoch)
        past = history.setdefault(key, [])
        category = (event.get("decision") if kind == "feature_admission_decision"
                    else event.get("outcome") if kind == "feature_outcome" else None)
        if category is not None:
            if category not in COUNTS:
                raise ValueError("Unknown feature activity category")
            past.append((second, category))
            if kind == "feature_admission_decision" and category == "admitted" and not event["outcome_tracking_supported"]:
                past.append((second, "untracked_admitted"))
            activity_events += 1
        if kind == "feature_deadline_resolution":
            categories = {"timely_first_text": ["deadline_timely_first_text"],
                          "deadline_breach": ["deadline_breach", "deadline_or_cutoff_adverse"],
                          "actual_cutoff": ["deadline_or_cutoff_adverse"],
                          "unknown_pretext_end": ["deadline_or_cutoff_unknown"]}.get(event.get("resolution"))
            if categories is None or event["feature_method"] != CURRENT_METHOD:
                raise ValueError("Unknown deadline resolution category")
            past.extend((second, name) for name in categories)
            activity_events += 1
        expected = Counter(category for bucket, category in past if second - 60 < bucket <= second)
        if any(captured[name] != expected[name] for name in names):
            errors.append(dict(event_index=index, event_kind=kind,
                               request_id=event.get("request_id"), reason="Activity window counts differ",
                               expected={name: expected[name] for name in names}, captured=captured))
        snapshots += 1
    return dict(ok=not errors, errors=errors, snapshots=snapshots,
                activity_events=activity_events, worker_generations=len(history),
                window="Captured current integer-second bucket plus59 preceding buckets; not a client future-label rate.")


def scores(rows, threshold=300):
    """Unknown candidate flags never disappear from whole-cohort denominators."""
    policies = ("original_scalar_total", "feature_reported_total", "feature_context_only_total",
                "feature_legacy_fallback_total", "original_scalar_extra_flag",
                "feature_reported_extra_flag", "feature_legacy_fallback_extra_flag")

    def prediction(row, policy):
        candidate = row.get("feature_context_candidate") or {}
        original_total = row.get("pre_admission_total_ttft_predictions", {}).get("calibrated")
        original_extra = row.get("pre_admission_recorded_extra_flags", {}).get("calibrated")
        if policy == "original_scalar_total":
            return original_total
        if policy == "original_scalar_extra_flag":
            return original_extra
        if policy == "feature_reported_extra_flag":
            return candidate.get("recorded_extra_would_reject")
        if policy == "feature_legacy_fallback_extra_flag":
            value = candidate.get("recorded_extra_would_reject")
            return original_extra if value is None else value
        value = candidate.get("total_ttft_seconds")
        if policy == "feature_context_only_total" and not candidate.get("contextual"):
            return None
        return original_total if value is None and policy == "feature_legacy_fallback_total" else value

    result = {}
    for policy in policies:
        card = dict(offered=len(rows), tp=0, fp=0, tn=0, fn=0, unknown=0,
                    prediction_unknown=0, truth_unknown=0, censored_known_late=0)
        for row in rows:
            truth = row.get("first_text_late")
            if row.get("controlled_rejection") or row.get("local_drop"):
                truth = None
            value = prediction(row, policy)
            card["prediction_unknown"] += value is None
            card["truth_unknown"] += truth is None
            card["censored_known_late"] += truth is True and row.get("client_ttft_seconds") is None
            if truth is None or value is None:
                card["unknown"] += 1
            else:
                over = value if policy.endswith("flag") else value >= threshold
                card["tp" if over and truth else "fp" if over else "fn" if truth else "tn"] += 1
        card["fpr"] = card["fp"] / (card["fp"] + card["tn"]) if card["fp"] + card["tn"] else None
        card["fnr"] = card["fn"] / (card["fn"] + card["tp"]) if card["fn"] + card["tp"] else None
        result[policy] = card
    common = [r for r in rows if (r.get("feature_context_candidate") or {}).get("recorded_extra_would_reject") is not None
              and r.get("pre_admission_recorded_extra_flags", {}).get("calibrated") is not None]
    result["coverage"] = dict(contextual=sum(bool((r.get("feature_context_candidate") or {}).get("contextual")) for r in rows),
                              common_extra_prediction_known=len(common), offered=len(rows))
    return result
