"""Audit recorded, once-only router TTFT deadline classifications.

This ledger uses router events, not client outcomes or future TTFT labels. The
four union counters are recorded at publication time, independently of the real
terminal diagnostic counters validated by wait_feature_context.audit_windows.
"""
from collections import Counter
import json
import math

LEGACY_METHOD = "feature_context_ridge_v1"
METHOD = "feature_context_deadline_ridge_v2"
JOINT_METHOD = "feature_joint_gains_deadline_ridge_v3"
DEADLINE_METHODS = {METHOD, JOINT_METHOD}
DEADLINE_COUNTS = ("deadline_timely_first_text", "deadline_breach",
                   "deadline_or_cutoff_adverse", "deadline_or_cutoff_unknown")
RESOLUTIONS = {"timely_first_text": ("deadline_timely_first_text",),
               "deadline_breach": ("deadline_breach", "deadline_or_cutoff_adverse"),
               "actual_cutoff": ("deadline_or_cutoff_adverse",),
               "unknown_pretext_end": ("deadline_or_cutoff_unknown",)}
TERMINALS = {"first_text", "upstream_timeout", "pretext_downstream_disconnect", "unknown_pretext_end"}
SNAPSHOTS = {"feature_context_forecast", "feature_context_poll", "feature_admission_decision",
             "feature_outcome", "feature_deadline_started", "feature_deadline_resolution"}
REQUEST_EVENTS = SNAPSHOTS - {"feature_context_poll"} | {
    "feature_context_feedback", "feature_outcome_ignored", "feature_deadline_ignored"}


def require(value, message):
    if not value:
        raise ValueError(message)


def clock(value, name):
    require(type(value) in (int, float) and math.isfinite(value), "Invalid clock: " + name)
    return value


def integer(value, name):
    require(type(value) is int and 0 <= value < 2**64, "Invalid counter: " + name)
    return value


def close(value, expected, name):
    require(math.isclose(clock(value, name), expected, rel_tol=0, abs_tol=1e-7),
            "Clock mismatch: " + name)


def payload(event):
    value = json.loads(event["feature_json"], parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    require(isinstance(value, dict), "Feature payload is not an object")
    return value


def distribution(values):
    values = sorted(values)
    return dict(n=len(values), minimum=values[0] if values else None,
                p50=values[(len(values)-1)//2] if values else None,
                maximum=values[-1] if values else None)


def audit_deadlines(events, rows):
    """Return a fail-closed receipt for one complete ordered feature capture."""
    counts, errors, poll_lags, publication_lags = Counter(), [], [], []
    methods = {e.get("feature_method") for e in events if str(e.get("event", "")).startswith("feature_")}
    unknown = methods - {LEGACY_METHOD, *DEADLINE_METHODS}
    if unknown or len(methods) > 1:
        return dict(ok=False, applicable=True, errors=[dict(reason="Unknown or mixed feature method")], counts={})
    if not methods & DEADLINE_METHODS:
        bad = any(str(e.get("event", "")).startswith("feature_deadline_") for e in events)
        return dict(ok=not bad, applicable=False, status="not_applicable",
                    errors=[dict(reason="Deadline events require deadline-v2 method")] if bad else [], counts={})
    owned = {r.get("opc_request_id"): r for r in rows}
    if None in owned or "" in owned or len(owned) != len(rows):
        return dict(ok=False, applicable=True, errors=[dict(reason="Missing/duplicate owned request ID")], counts={})
    states, selected, decisions, started, resolved, terminals, ignored = {}, {}, {}, {}, {}, {}, {}
    future_terminal = {}
    for i, event in enumerate(events):
        if event.get("event") == "feature_deadline_resolution" and event.get("resolution_source") in TERMINALS:
            future_terminal.setdefault(event.get("request_id"), []).append((i, event))

    def state(worker, epoch):
        require(isinstance(worker, str) and bool(worker), "Missing worker identity")
        integer(epoch, "calibration_epoch")
        if worker not in states:
            states[worker] = dict(epoch=epoch, tracked=set(), untracked=set(), history=[], origin=None, bucket=None)
        require(states[worker]["epoch"] == epoch, "Epoch changed without recorded reset")
        return states[worker]

    def clear(worker, reason):
        s = states.get(worker)
        if s:
            for rid, item in started.items():
                if item["worker"] == worker and item["epoch"] == s["epoch"] and rid not in terminals:
                    item["cleared"] = reason
            counts["pending_cleared_" + reason] += len(s["tracked"]) + len(s["untracked"])
            del states[worker]

    def identity(rid, worker):
        require(rid in decisions, "Deadline lacks preceding actual admission decision")
        require(decisions[rid]["decision"] == "admitted" and decisions[rid]["outcome_tracking_supported"] is True,
                "Rejected/unsupported admission cannot register a deadline")
        require(rid in selected, "Deadline lacks preceding selected calibration forecast")
        require(selected[rid]["worker"] == worker == decisions[rid]["worker"], "Selected deadline worker mismatch")
        return integer(selected[rid]["calibration_epoch"], "selected admission epoch")

    def snapshot(event, s, categories):
        p = payload(event)
        context = p["context" if event["event"].startswith("feature_context_") else "activity"]
        now = clock(event["diagnostic_time_seconds"], "diagnostic_time_seconds")
        origin = clock(context["activity_origin_time_seconds"], "activity origin")
        second = integer(context["bucket_second"], "bucket_second")
        require(second - 1e-8 <= now-origin < second+1+1e-8, "Activity bucket uses wrong publication clock")
        if s["origin"] is None:
            s["origin"] = origin
        require(origin == s["origin"], "Activity origin changed without reset")
        require(s["bucket"] is None or second >= s["bucket"], "Activity buckets moved backwards")
        s["bucket"] = second
        pending = integer(context["deadline_untracked_pending"], "deadline_untracked_pending")
        require(pending == len(s["untracked"]), "Untracked pending ledger mismatch")
        require(not pending or context["values"][1] is None, "Untracked pending must suppress union rate")
        s["history"] = [(b, names) for b, names in s["history"] if b > second-60]
        if categories:
            s["history"].append((second, categories))
        expected = Counter(name for b, names in s["history"] for name in names if b <= second)
        for name in DEADLINE_COUNTS:
            require(integer(context["counts"][name], name) == expected[name], "Deadline union window mismatch: " + name)
            if "counts" in p:
                require(integer(p["counts"][name], name) == context["counts"][name], "Duplicated deadline counters differ")
        counts["context_snapshots"] += 1
        if event["event"] == "feature_context_poll":
            lower = clock(p["poll_started_time_seconds"], "poll started")
            require(lower <= now, "Poll start follows publication")
            for rid in s["tracked"]:
                item = started[rid]
                if item["deadline"] > lower:
                    continue
                # Atomic scan/terminal claims are not logged. A later router
                # terminal observation can explain why this poll skipped it;
                # its count is still added only when its resolution is logged.
                claimed = any(j > current_index and e.get("worker") == event["worker"]
                              and e.get("calibration_epoch") == s["epoch"]
                              and type(e.get("observed_time_seconds")) in (int, float)
                              and item["start"] <= e["observed_time_seconds"] <= now
                              for j, e in future_terminal.get(rid, []))
                if claimed:
                    counts["poll_due_pending_with_later_terminal_observation"] += 1
                else:
                    # A metadata-only scan can claim then release without a
                    # terminal event. Its unlogged state cannot be disproved.
                    counts["poll_due_pending_without_observed_terminal_unverified"] += 1

    for current_index, event in enumerate(events):
        kind, worker, rid = event.get("event"), event.get("worker"), event.get("request_id")
        try:
            if kind == "calibration_reset":
                epoch = integer(event["calibration_epoch"], "reset epoch")
                if worker in states:
                    require(epoch == (states[worker]["epoch"]+1) % 2**64, "Reset epoch is not consecutive")
                clear(worker, "reset")
                state(worker, epoch)
                counts["resets"] += 1
                continue
            if kind == "calibration_forecast" and rid in owned:
                require(rid not in selected, "Duplicate selected calibration forecast")
                selected[rid] = event
                continue
            if not str(kind).startswith("feature_"):
                continue
            if kind in REQUEST_EVENTS:
                require(rid in owned and not owned[rid].get("local_drop"), "Unowned/no-identity feature request")
            categories = ()
            if kind == "feature_deadline_ignored":
                require(rid not in ignored and rid not in started, "Duplicate deadline registration/ignore")
                epoch = identity(rid, worker)
                require(event["admission_epoch"] == epoch and event["reason"] == "generation_changed_or_worker_missing",
                        "Invalid ignored deadline generation")
                if event.get("calibration_epoch") is None:
                    clear(worker, "worker_missing")
                else:
                    state(worker, event["calibration_epoch"])
                ignored[rid] = event
                counts["registration_ignored"] += 1
                continue
            if kind == "feature_outcome_ignored":
                require(rid not in terminals and event["outcome"] in TERMINALS, "Duplicate/invalid ignored terminal")
                require(event["admission_epoch"] == identity(rid, worker), "Ignored terminal admission epoch differs")
                require(event["reason"] in ("generation_changed", "worker_missing"), "Unknown ignored terminal reason")
                if event["reason"] == "worker_missing":
                    clear(worker, "worker_missing")
                require(rid in ignored or rid in started and started[rid]["cleared"], "Ignored terminal lacks reset/missing-worker provenance")
                terminals[rid] = event
                counts["ignored_terminals"] += 1
                continue
            if kind not in SNAPSHOTS:
                continue
            s = state(worker, event["calibration_epoch"])
            now = clock(event["diagnostic_time_seconds"], "diagnostic_time_seconds")
            if kind == "feature_admission_decision":
                require(rid not in decisions, "Duplicate actual admission decision")
                require(event["decision"] in ("admitted", "actual_rejected") and type(event["outcome_tracking_supported"]) is bool,
                        "Invalid actual admission decision")
                decisions[rid] = event
                counts[event["decision"]] += 1
                counts["unsupported_tracking_admissions"] += event["decision"] == "admitted" and not event["outcome_tracking_supported"]
            elif kind == "feature_deadline_started":
                require(rid not in started and rid not in ignored, "Duplicate deadline registration")
                require(identity(rid, worker) == event["calibration_epoch"], "Deadline registered in wrong admission epoch")
                start = clock(event["request_started_time_seconds"], "request start")
                deadline = clock(event["deadline_time_seconds"], "deadline")
                require(start <= now and type(event["deadline_seconds"]) is int and event["deadline_seconds"] == 300,
                        "Invalid request start/deadline duration")
                close(deadline, start+300, "deadline=start+300")
                tracked = event["deadline_tracking_supported"]
                require(type(tracked) is bool, "Invalid tracking support flag")
                started[rid] = dict(worker=worker, epoch=s["epoch"], start=start, deadline=deadline, tracked=tracked, cleared=None)
                s["tracked" if tracked else "untracked"].add(rid)
                require(integer(event["pending_deadline_count"], "pending_deadline_count") == len(s["tracked"]) <= 8192,
                        "Tracked pending count differs from started ledger")
                counts["started"] += 1
                counts["tracked_started" if tracked else "untracked_started"] += 1
                counts["peak_untracked_pending"] = max(counts["peak_untracked_pending"], len(s["untracked"]))
            elif kind == "feature_deadline_resolution":
                require(rid in started and rid not in resolved, "Missing registration or duplicate union resolution")
                item = started[rid]
                require(item["worker"] == worker and item["epoch"] == s["epoch"] and not item["cleared"],
                        "Resolution crosses worker/reset generation")
                observed = clock(event["observed_time_seconds"], "observed time")
                require(item["start"] <= observed <= now, "Observed time is outside request/publication clock")
                close(event["request_started_time_seconds"], item["start"], "resolution start")
                close(event["deadline_time_seconds"], item["deadline"], "resolution deadline")
                require(type(event["deadline_seconds"]) is int and event["deadline_seconds"] == 300, "Resolution deadline differs")
                close(event["elapsed_seconds"], observed-item["start"], "observed elapsed")
                source = event["resolution_source"]
                require(source == "poll" or source in TERMINALS, "Unknown resolution source")
                if source == "poll":
                    require(item["tracked"], "Poll cannot resolve untracked registration")
                    close(observed, now, "poll observed/publication time")
                    require(observed >= item["deadline"], "Early poll deadline breach")
                    poll_lags.append(observed-item["deadline"])
                expected = ("deadline_breach" if observed >= item["deadline"] else "timely_first_text" if source == "first_text"
                            else "actual_cutoff" if source in {"upstream_timeout", "pretext_downstream_disconnect"} else "unknown_pretext_end")
                require(event["resolution"] == expected, "Resolution class differs from observed time/source")
                require(event.get("native_cancellation_acknowledgement") == "unavailable", "Invented native cancellation ACK")
                s["tracked" if item["tracked"] else "untracked"].remove(rid)
                resolved[rid] = event
                categories = RESOLUTIONS[expected]
                counts["resolved"] += 1; counts[expected] += 1
                publication_lags.append(now-observed)
            elif kind == "feature_outcome":
                require(rid not in terminals and event["outcome"] in TERMINALS, "Duplicate/invalid actual terminal")
                require(rid in resolved and rid in started and started[rid]["worker"] == worker
                        and started[rid]["epoch"] == s["epoch"] and not started[rid]["cleared"], "Terminal lacks current once-only resolution")
                resolution = resolved[rid]
                require(resolution["resolution_source"] in ("poll", event["outcome"]), "Terminal source differs from resolution")
                if resolution["resolution_source"] != "poll":
                    close(now, resolution["diagnostic_time_seconds"], "terminal/resolution publication")
                if event["outcome"] == "first_text" and resolution["resolution_source"] == "poll":
                    counts["first_text_after_poll_breach"] += 1
                require(event.get("native_cancellation_acknowledgement") == "unavailable", "Invented native cancellation ACK")
                terminals[rid] = event; counts["actual_terminals"] += 1
            snapshot(event, s, categories)
        except (ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
            errors.append(dict(event_index=current_index, event=kind, request_id=rid, reason=str(error)))
            break
    if not errors:
        try:
            for rid, row in owned.items():
                if row.get("local_drop"):
                    continue
                require(rid in decisions, "Missing actual admission decision: " + rid)
                d = decisions[rid]
                if d["decision"] == "actual_rejected" or not d["outcome_tracking_supported"]:
                    require(rid not in started and rid not in ignored and rid not in terminals, "Rejected/unsupported request has deadline identity")
                else:
                    identity(rid, d["worker"])
                    require(rid in started or rid in ignored, "Missing deadline registration/ignored record: " + rid)
                    require(rid in terminals, "Missing terminal/ignored observation: " + rid)
                    require(rid in ignored or rid in resolved or started[rid]["cleared"], "Unresolved deadline at complete capture: " + rid)
        except (ValueError, KeyError) as error:
            errors.append(dict(reason=str(error)))
    return dict(ok=not errors, applicable=True, method=next(iter(methods)), errors=errors, counts=dict(counts),
                poll_resolution_lag_seconds=distribution(poll_lags), publication_lag_seconds=distribution(publication_lags),
                pending_at_end=sum(len(s["tracked"])+len(s["untracked"]) for s in states.values()),
                limits="Router observations only; no client future labels or native cancellation ACK. Atomic scan/terminal claims are unlogged; due pending observations without a terminal claim remain unverified diagnostics rather than failures. Eventual once-only resolution and completeness remain required. Poll-start time is a conservative advance bound. Union counts enter the diagnostic-publication bucket, never the earlier observation bucket.")
