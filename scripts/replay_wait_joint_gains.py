#!/usr/bin/env python3
"""Causal v3 replay and fixed-cohort comparison; never runs a DEV workload."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

IGNORED_POLLS = {"duplicate", "out_of_order", "generation_changed", "worker_removed", "untracked"}
try:
    from . import wait_feature_joint as joint
except ImportError:
    import wait_feature_joint as joint


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path):
    spec = importlib.util.spec_from_file_location("frozen_v2_replay", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def require(value, message):
    if not value:
        raise ValueError(message)


def close(a, b, path="value"):
    if isinstance(a, dict):
        require(isinstance(b, dict), path + ": missing object")
        for key, value in a.items():
            close(value, b.get(key), path + "." + key)
    elif isinstance(a, list):
        require(isinstance(b, list) and len(a) == len(b), path + ": wrong vector")
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, path + "[" + str(i) + "]")
    elif a is None or isinstance(a, (str, bool, int)):
        require(type(a) is type(b) and a == b, path + ": state mismatch")
    else:
        require(
            type(b) in (int, float)
            and math.isfinite(b)
            and math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-7),
            path + ": arithmetic mismatch",
        )


def candidate_from_v2(events, audited_rows):
    """Use independently audited causal retention, not outcome-sorted client rows.

    Historical v2 captures do not always log the exact v3 retarget instant. This
    counterfactual updates at each logged feature-poll publication (causal and
    slightly later); live-v3 fidelity uses the actual logged vector anchor.
    """
    indexed = {r["event_index"]: r for r in audited_rows}
    states, profiles, output = {}, {}, []
    for index, event in enumerate(events):
        worker = event.get("worker")
        kind = event.get("event")
        if kind == "calibration_reset":
            for key in list(states):
                if key[0] == worker:
                    del states[key]
            profiles.pop(worker, None)
        if kind == "calibration_poll" and event.get("outcome") not in {"success", *IGNORED_POLLS}:
            for key, model in states.items():
                if key[0] == worker:
                    model.invalidate()
        if (
            kind == "calibration_rotation"
            and event["sample_count_before"] > event["sample_count_after"]
        ):
            for key, model in states.items():
                if key[0] == worker and key[2] == (event["calibration_state"] == "capacity_ready"):
                    model.invalidate()
        if index not in indexed:
            continue
        row = indexed[index]
        p = json.loads(event["feature_json"])
        key = (worker, row["epoch"], p["native_ready"])
        model = states.setdefault(key, joint.Model())
        now = event["diagnostic_time_seconds"]
        if kind == "feature_context_poll":
            if profiles.get(worker) != p["profile"]:
                for k in list(states):
                    if k[0] == worker:
                        states[k] = joint.Model()
                model = states[key]
                profiles[worker] = p["profile"]
            # Failed polls have no update; captured v2 predicate records invalidation.
            prior_poll = next(
                (
                    e
                    for e in reversed(events[max(0, index - 12) : index])
                    if e.get("event") == "calibration_poll" and e.get("worker") == worker
                ),
                None,
            )
            require(prior_poll is not None, "Feature poll lacks nearby scalar poll")
            if prior_poll["outcome"] == "success":
                if model.fit["revision"] != row["history_revision"]:
                    model.fit = joint.learn(
                        row["retained_sample_evidence"], row["history_revision"]
                    )
                model.update(p["context"]["values"], p["legacy_gain"], now)
        prediction = model.prediction(
            p["context"]["values"],
            p["legacy_gain"],
            now,
            p["fresh"] and p["native_ready"] and p["supported_path"],
            p["original_queue_excess_seconds"],
        )
        extra = prediction["gain"] * p["original_queue_excess_seconds"]
        extra = extra if math.isfinite(extra) else None
        total = (
            p["baseline_seconds"] + extra
            if p["baseline_seconds"] is not None and extra is not None
            else None
        )
        total = total if total is not None and math.isfinite(total) else None
        output.append(
            dict(
                event_index=index,
                event=kind,
                request_id=event.get("request_id"),
                worker=worker,
                epoch=row["epoch"],
                diagnostic_time_seconds=now,
                fit=model.fit.copy(),
                prediction=prediction,
                total_ttft_seconds=total,
                extra_seconds=extra,
                prediction_available=p["fresh"]
                and p["native_ready"]
                and p["supported_path"]
                and total is not None,
                original_queue_excess_seconds=p["original_queue_excess_seconds"],
                baseline_seconds=p["baseline_seconds"],
                feature_values=p["context"]["values"],
            )
        )
    return output


def score(rows, key, phase=None):
    selected = [r for r in rows if phase is None or r["phase"] == phase]
    card = dict(
        offered=len(selected),
        tp=0,
        fp=0,
        tn=0,
        fn=0,
        unknown=0,
        truth_unknown=0,
        prediction_unknown=0,
        censored_known_late=0,
    )
    errors = []
    for row in selected:
        truth = row["truth"]
        value = row[key]
        card["truth_unknown"] += truth is None
        card["prediction_unknown"] += value is None
        card["censored_known_late"] += truth is True and row["actual_ttft_seconds"] is None
        if truth is None or value is None:
            card["unknown"] += 1
        else:
            card[
                "tp"
                if value >= 300 and truth
                else "fp"
                if value >= 300
                else "fn"
                if truth
                else "tn"
            ] += 1
        if row["actual_ttft_seconds"] is not None and truth is not None and value is not None:
            errors.append(value - row["actual_ttft_seconds"])
    card.update(
        fpr=card["fp"] / (card["fp"] + card["tn"]) if card["fp"] + card["tn"] else None,
        fnr=card["fn"] / (card["fn"] + card["tp"]) if card["fn"] + card["tp"] else None,
        exact_n=len(errors),
        mae_seconds=sum(abs(x) for x in errors) / len(errors) if errors else None,
        mse_seconds_squared=sum(x * x for x in errors) / len(errors) if errors else None,
    )
    return card


def censor_bound(row, truth):
    if (
        truth is not True
        or row.get("client_ttft_seconds") is not None
        or row.get("cancellation_reason") not in {"ttft_deadline", "observation_horizon"}
    ):
        return None
    bound = row.get("first_text_observation_horizon_seconds")
    started = row.get("started_elapsed_seconds")
    delivered = row.get("cancellation_delivered_elapsed_seconds")
    if any(
        type(x) not in (int, float) or not math.isfinite(x) for x in (bound, started, delivered)
    ):
        return None
    return bound if delivered - started >= bound else None


def compare(paired, candidate, historical=True):
    predictions = {
        r["request_id"]: r for r in candidate if r["event"] == "feature_context_forecast"
    }
    require(len(predictions) == len(paired), "Candidate/admission count mismatch")
    rows = []
    for row in paired:
        rid = row["opc_request_id"]
        p = predictions[rid]
        scalar = row["pre_admission_total_ttft_predictions"]["calibrated"]
        live = row.get("recovery_candidate") or {}
        live_total = live.get("explicit_fallback_total_ttft_predictions", {}).get(
            "legacy_gain_ablation"
        )
        if live_total is None:
            live_total = scalar
        feature_candidate = row.get("feature_context_candidate") or {}
        feature = (
            feature_candidate.get("total_ttft_seconds")
            if feature_candidate.get("prediction_available")
            else None
        )
        truth = row.get("first_text_late")
        if row.get("controlled_rejection") or row.get("local_drop"):
            truth = None
        rows.append(
            dict(
                request_id=rid,
                phase=row["phase"],
                arrival_seconds=row.get("started_elapsed_seconds"),
                truth=truth,
                actual_ttft_seconds=row.get("client_ttft_seconds"),
                scalar=scalar,
                feature_v2=feature if feature is not None else scalar,
                live_same_scalar=live_total,
                joint_v3=p["total_ttft_seconds"]
                if p["prediction_available"] and p["total_ttft_seconds"] is not None
                else scalar,
                censor_bound_seconds=censor_bound(row, truth),
                prediction_available=p["prediction_available"],
                **{
                    k: p[k]
                    for k in ("prediction", "feature_values", "original_queue_excess_seconds")
                },
            )
        )
    if not historical:
        for row in rows:
            row.pop("feature_v2")
    policies = (
        ("scalar", "feature_v2", "live_same_scalar", "joint_v3")
        if historical
        else ("scalar", "live_same_scalar", "joint_v3")
    )
    cards = {
        phase or "all": {key: score(rows, key, phase) for key in policies}
        for phase in [None] + sorted({r["phase"] for r in rows})
    }
    return rows, cards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-helper", type=Path)
    ap.add_argument("--receipt", action="append", type=Path)
    ap.add_argument("--router", type=Path)
    ap.add_argument("--state", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    require(not args.out.exists(), "Preserve existing replay outputs")
    if args.router:
        require(
            args.state is not None and not args.receipt and args.v2_helper is None,
            "Live audit needs --router --state, without historical arguments",
        )
        return current_main(args.router, args.state, args.out)
    require(
        args.v2_helper is not None and args.receipt,
        "Historical replay needs --v2-helper and --receipt",
    )
    old = module(args.v2_helper)
    args.out.mkdir(parents=True)
    summaries = {}
    for path in args.receipt:
        previous = json.loads(path.read_text())
        require(previous["ok"], "Historical receipt failed")
        require(sha(args.v2_helper) == previous["helper_sha256"], "Historical helper hash mismatch")
        router = Path(previous["input_path"])
        old.verify_source(Path(previous["source"]["state_path"]), router)
        events, _ = old.read_events(router)
        audit = old.replay(events)
        require(audit["ok"], "Historical v2 deterministic replay failed: " + str(audit["errors"]))
        candidate = candidate_from_v2(events, audit["rows"])
        repeat = candidate_from_v2(events, audit["rows"])
        require(candidate == repeat, "Nondeterministic joint replay")
        paired = [
            json.loads(line)
            for line in router.with_name("paired-production.jsonl").read_text().splitlines()
        ]
        rows, cards = compare(paired, candidate)
        name = path.parent.name
        out = args.out / name
        out.mkdir()
        (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in candidate))
        (out / "admissions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary = dict(
            method=joint.METHOD,
            historical_method=old.METHOD,
            source_receipt=str(path),
            source_receipt_sha256=sha(path),
            router_sha256=sha(router),
            paired_sha256=sha(router.with_name("paired-production.jsonl")),
            historical_replay_ok=audit["ok"],
            historical_replay_counts=audit["counts"],
            deterministic_repeat_equal=True,
            replay_helper_sha256=sha(__file__),
            joint_math_sha256=sha(joint.__file__),
            scoring=(
                "Only eligible candidate TOTAL predictions; unavailable candidate "
                "uses exact same-admission scalar fallback."
            ),
            scorecards=cards,
            contextual_admissions=sum(r["prediction"]["contextual"] for r in rows),
            limitations=[
                (
                    "Historical counterfactual at logged feature-poll publication; "
                    "exact earlier internal v3 retarget instant was not captured."
                ),
                (
                    "Same captured arrival workload/outcomes; shadow policy only. "
                    "Enforced traffic and rejected TTFT cannot be inferred."
                ),
                (
                    "Native inputs and scalar acceptance remain recorded prerequisites; "
                    "exact labels only, censored outcomes binary only."
                ),
            ],
        )
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries[name] = summary
        print(
            json.dumps(
                dict(arm=name, contextual=summary["contextual_admissions"], all=cards["all"])
            ),
            flush=True,
        )
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


def audit_saved_context(context, admission, admission_index, activities, selected_snapshot):
    """Rebuild saved rates at selected admission, not earlier pre-credit check."""
    try:
        from .wait_feature_context import COUNTS, DEADLINE_COUNTS, validate_context
    except ImportError:
        from wait_feature_context import COUNTS, DEADLINE_COUNTS, validate_context
    require(selected_snapshot is not None, "Saved context lacks selected native snapshot")
    close(context["values"][0], selected_snapshot["values"][0], "saved native KV snapshot")
    close(
        context["activity_origin_time_seconds"],
        selected_snapshot["activity_origin_time_seconds"],
        "saved activity origin",
    )
    validate_context(context, joint.METHOD)
    second = context["bucket_second"]
    elapsed = admission["diagnostic_time_seconds"] - context["activity_origin_time_seconds"]
    require(
        second - 1e-8 <= elapsed < second + 1 + 1e-8,
        "Saved context clock differs from selected admission",
    )
    counts = Counter()
    overflow = set()
    for index, e in activities:
        if index >= admission_index:
            break
        if (
            e["worker"] != admission["worker"]
            or e["calibration_epoch"] != admission["calibration_epoch"]
        ):
            continue
        a = json.loads(e["feature_json"])["activity"]
        names = []
        kind = e["event"]
        close(
            a["activity_origin_time_seconds"],
            context["activity_origin_time_seconds"],
            "saved context activity epoch",
        )
        if kind == "feature_deadline_started" and not e["deadline_tracking_supported"]:
            overflow.add(e["request_id"])
        elif kind == "feature_deadline_resolution":
            overflow.discard(e["request_id"])
            names = {
                "timely_first_text": ["deadline_timely_first_text"],
                "deadline_breach": ["deadline_breach", "deadline_or_cutoff_adverse"],
                "actual_cutoff": ["deadline_or_cutoff_adverse"],
                "unknown_pretext_end": ["deadline_or_cutoff_unknown"],
            }[e["resolution"]]
        elif kind == "feature_outcome":
            names = [e["outcome"]]
        elif kind == "feature_admission_decision":
            names = [e["decision"]]
            if e["decision"] == "admitted" and not e["outcome_tracking_supported"]:
                names.append("untracked_admitted")
        if second - 60 < a["bucket_second"] <= second:
            counts.update(names)
    close(
        {name: counts[name] for name in COUNTS + DEADLINE_COUNTS},
        context["counts"],
        "causal saved context counts",
    )
    close(len(overflow), context["deadline_untracked_pending"], "saved pending overflow")


def replay_current(events):
    """Reconstruct v3 fit, retention and vector anchors from ordered router events.

    Native parsing and scalar acceptance are recorded prerequisites, as in v2.
    This pass independently checks saved sample arithmetic and joint fitting.
    """
    states = {}
    selected = {}
    selected_indices = {}
    selected_profiles = {}
    selected_contexts = {}
    scalar_feedback = {}
    first_text = {}
    snapshots = []
    counts = Counter()
    last_poll = {}
    latest_snapshot = {}
    activities = []
    for event_index, event in enumerate(events):
        if event.get("event") == "first_text":
            require(event["request_id"] not in first_text, "Duplicate observed first text")
            first_text[event["request_id"]] = event
        if event.get("event") in {
            "feature_admission_decision",
            "feature_outcome",
            "feature_deadline_started",
            "feature_deadline_resolution",
        }:
            activities.append((event_index, event))
        if event.get("event") == "calibration_feedback":
            key = (event["worker"], event["request_id"])
            require(key not in scalar_feedback, "Duplicate scalar feedback")
            scalar_feedback[key] = event

    def state(worker, epoch):
        if worker not in states:
            states[worker] = dict(
                epoch=epoch,
                samples=[],
                previous=0,
                bank=0,
                bank_origin=None,
                revision=0,
                profile=None,
                models=[joint.Model(), joint.Model()],
            )
        require(states[worker]["epoch"] == epoch, "Epoch changed without reset")
        return states[worker]

    for index, e in enumerate(events):
        kind = e.get("event")
        worker = e.get("worker")
        if kind == "calibration_reset":
            if worker in states:
                require(e["calibration_epoch"] == states[worker]["epoch"] + 1, "Reset epoch gap")
            states.pop(worker, None)
            latest_snapshot.pop(worker, None)
            state(worker, e["calibration_epoch"])
            counts["resets"] += 1
            continue
        if kind == "calibration_forecast":
            key = (worker, e["request_id"])
            require(key not in selected, "Duplicate selected admission")
            selected[key] = e
            selected_indices[key] = index
            selected_profiles[key] = states.get(worker, {}).get("profile")
            selected_contexts[key] = latest_snapshot.get(worker)
        if kind == "calibration_poll":
            if e.get("outcome") in IGNORED_POLLS:
                continue
            if e.get("calibration_epoch") is None:
                continue
            s = state(worker, e["calibration_epoch"])
            last_poll[worker] = e
            if e["outcome"] != "success":
                for model in s["models"]:
                    model.invalidate()
        if kind == "calibration_rotation":
            s = state(worker, e["calibration_epoch"])
            if e["calibration_state"] != "capacity_ready":
                continue
            close(300, e["calibration_bank_seconds"], "bank width")
            require(s["bank_origin"] is not None, "Rotation without accepted feedback origin")
            expected_bank = int(max(0.0, e["calibration_time_seconds"] - s["bank_origin"]) // 300)
            close(expected_bank, e["calibration_bank_index"], "clock-derived bank index")
            close(len(s["samples"]), e["sample_count_before"], "rotation before count")
            steps = e["calibration_bank_index"] - s["bank"]
            require(steps > 0, "Nonincreasing bank")
            remove = len(s["samples"]) if steps >= 2 else s["previous"]
            s["samples"] = s["samples"][remove:]
            s["previous"] = len(s["samples"])
            s["bank"] = e["calibration_bank_index"]
            if remove:
                s["revision"] += 1
                s["models"][1].invalidate()
            close(s["revision"], e["calibration_history_revision"], "rotation revision")
            close(len(s["samples"]), e["sample_count_after"], "rotation count")
            close(s["previous"], e["calibration_previous_sample_count"], "rotation previous count")
            close(0, e["calibration_current_sample_count"], "rotation current count")
        if kind not in (
            "feature_context_feedback",
            "feature_context_poll",
            "feature_context_forecast",
        ):
            continue
        require(e["feature_method"] == joint.METHOD, "Wrong live method")
        p = json.loads(e["feature_json"])
        s = state(worker, e["calibration_epoch"])
        now = e["diagnostic_time_seconds"]
        if kind == "feature_context_feedback":
            key = (worker, e["request_id"])
            require(
                key in selected and key in scalar_feedback,
                "Feedback lacks selected admission/scalar evidence",
            )
            raw = scalar_feedback.pop(key)
            admission = selected[key]
            close(now, raw["diagnostic_time_seconds"], "shared feedback publication time")
            close(p["admission_profile"], selected_profiles[key], "selected admission profile")
            close(p["current_profile"], s["profile"], "current feedback profile")
            require(e["request_id"] in first_text, "Feedback lacks observed first-text provenance")
            close(
                p["ttft_seconds"],
                first_text[e["request_id"]]["ttft_seconds"],
                "observed first-text TTFT",
            )
            if p["saved_context"] is not None:
                audit_saved_context(
                    p["saved_context"],
                    admission,
                    selected_indices[key],
                    activities,
                    selected_contexts[key],
                )
            close(p["reason"], raw["reason"], "shared acceptance")
            close(p["ttft_seconds"], raw["ttft_seconds"], "shared exact TTFT")
            close(
                p["queue_excess_seconds"], admission["calibration_input_excess_seconds"], "saved N"
            )
            close(
                p["baseline_seconds"],
                admission["calibration_baseline_seconds"]
                if admission["calibration_baseline_available"]
                else None,
                "saved baseline",
            )
            close(p["admission_epoch"], admission["calibration_epoch"], "saved epoch")
            close(p["admission_time_seconds"], admission["diagnostic_time_seconds"], "saved time")
            if p["reason"] == "accepted":
                require(
                    admission["calibration_epoch"] == s["epoch"]
                    and 0 <= now - p["admission_time_seconds"] < 600,
                    "Ineligible age/epoch",
                )
                n = p["queue_excess_seconds"]
                b = p["baseline_seconds"]
                ttft = p["ttft_seconds"]
                require(
                    n > 0 and b is not None and ttft > 0 and math.isfinite(ttft),
                    "Not an exact accepted sample",
                )
                ratio = max(0.0, ttft - b) / n
                close(ratio, p["ratio"], "ratio")
                expected = p["saved_context"] is not None and p["admission_profile"] == s["profile"]
                close(expected, p["feature_context_accepted"], "context acceptance")
                if s["bank_origin"] is None:
                    s["bank_origin"] = raw["calibration_time_seconds"]
                s["samples"].append(
                    dict(
                        n=n, ratio=ratio, context=p["saved_context"]["values"] if expected else None
                    )
                )
                s["revision"] += 1
                if len(s["samples"]) - s["previous"] > 32:
                    s["samples"].pop(s["previous"])
                close(len(s["samples"]), raw["sample_count_after"], "retained count")
                counts["accepted_feedback"] += 1
            close(
                s["revision"] if raw["admission_native_capacity_ready"] else 0,
                p["history_revision"],
                "feedback revision",
            )
            continue
        latest_snapshot[worker] = p["context"]
        ready = int(p["native_ready"])
        model = s["models"][ready]
        if kind == "feature_context_poll":
            poll = last_poll.get(worker)
            require(poll is not None, "Missing scalar poll")
            close(
                poll["poll_started_time_seconds"], p["poll_started_time_seconds"], "poll identity"
            )
            if poll["outcome"] == "success":
                if s["profile"] != p["profile"]:
                    for sample in s["samples"]:
                        sample["context"] = None
                    s["models"] = [joint.Model(), joint.Model()]
                    model = s["models"][ready]
                    s["profile"] = p["profile"]
                revision = s["revision"] if ready else 0
                if model.fit["revision"] != revision:
                    model.fit = joint.learn(s["samples"] if ready else [], revision)
                    counts["fits"] += 1
                active = joint.active_features(model.fit, p["context"]["values"])
                if any(model.fit["support"]) and active == model.fit["support"]:
                    at = p["prediction"]["trajectory_anchor_time_seconds"]
                    require(
                        type(at) in (int, float)
                        and poll["diagnostic_time_seconds"] - 1e-7 <= at <= now + 1e-7,
                        "Anchor outside poll publication",
                    )
                    # Scalar moves at .5/s: derive its value at the update instant
                    # from the preceding scalar poll's captured target/scale.
                    delta = min(
                        abs(poll["target_scale"] - poll["scale"]),
                        0.5 * max(0.0, at - poll["diagnostic_time_seconds"]),
                    )
                    scalar = poll["scale"] + (
                        delta if poll["target_scale"] >= poll["scale"] else -delta
                    )
                    model.update(p["context"]["values"], scalar, at)
                else:
                    model.invalidate()
            counts["polls"] += 1
        close(model.fit, p["fit"], "joint fitted model")
        close(s["revision"] if ready else 0, p["history_revision"], "snapshot revision")
        expected = model.prediction(
            p["context"]["values"],
            p["legacy_gain"],
            now,
            p["fresh"] and ready and p["supported_path"],
            p["original_queue_excess_seconds"],
        )
        close(expected, p["prediction"], "joint trajectory")
        snapshots.append(
            dict(
                event_index=index,
                event=kind,
                request_id=e.get("request_id"),
                fit=model.fit.copy(),
                prediction=expected,
            )
        )
        counts["forecasts"] += kind == "feature_context_forecast"
    require(not scalar_feedback, "Unpaired scalar feedback")
    require(counts["forecasts"] and counts["polls"], "No complete snapshot evidence")
    return dict(ok=True, method=joint.METHOD, counts=dict(counts), rows=snapshots)


def read_events(path):
    lines = path.read_text().splitlines(keepends=True)
    banner = "8431f29e164fc857bc681e0045fe57e51dd43f9e6bd7a7ff70b03a9b165ab72d"
    if hashlib.sha256("".join(lines[:16]).encode()).hexdigest() == banner:
        lines = lines[16:]
    result = []
    for line in lines:
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), "Router record must be an object")
        result.append(value.get("fields", value))
    return result


def current_main(router, state_path, out):
    try:
        from . import wait_feature_context as context
        from . import wait_feature_deadline as deadline
    except ImportError:
        import wait_feature_context as context
        import wait_feature_deadline as deadline
    state = json.loads(state_path.read_text())
    require(state.get("results_valid") is True, "Capture is not results_valid")
    root = Path(state["result_path"]).resolve()
    require(router.resolve() == root / "router-events.jsonl", "Not canonical delivered router file")
    for name, digest in state["result_files"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), "Result path escaped canonical root")
        require(sha(path) == digest, "Result hash mismatch: " + name)
    events = read_events(router)
    paired = [
        json.loads(line) for line in (root / "paired-production.jsonl").read_text().splitlines()
    ]
    audits = dict(
        method=context.audit_method(events, joint.METHOD),
        windows=context.audit_windows(events),
        activity=context.audit_activity(events, paired),
        deadlines=deadline.audit_deadlines(events, paired),
    )
    require(all(v["ok"] for v in audits.values()), "Independent lifecycle/context audit failed")
    gates = {(e["worker"], e["request_id"]): e for e in events if e.get("event") == "forecast"}
    # Standard harness already paired exact request/worker gates. Validate each
    # forecast here as well; selected calibration capture has a later clock.
    forecast_count = 0
    for e in events:
        if e.get("event") == "feature_context_forecast":
            require((e["worker"], e["request_id"]) in gates, "Missing exact pre-admission gate")
            context.forecast(e, gate=gates[(e["worker"], e["request_id"])])
            forecast_count += 1
    result = replay_current(events)
    repeat = replay_current(events)
    require(result == repeat, "Nondeterministic live replay")
    snapshots = result.pop("rows")
    out.mkdir(parents=True)
    actual_predictions = []
    for e in events:
        if e.get("event") == "feature_context_forecast":
            p = json.loads(e["feature_json"])
            actual_predictions.append(
                dict(
                    event=e["event"],
                    request_id=e["request_id"],
                    **{
                        k: p[k]
                        for k in (
                            "prediction",
                            "total_ttft_seconds",
                            "extra_seconds",
                            "prediction_available",
                            "original_queue_excess_seconds",
                            "baseline_seconds",
                        )
                    },
                    feature_values=p["context"]["values"],
                )
            )
    admission_rows, scorecards = compare(paired, actual_predictions, historical=False)
    (out / "admissions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in admission_rows))
    result["scorecards"] = scorecards
    result.update(
        deterministic_repeat_equal=True,
        forecast_count=forecast_count,
        audits=audits,
        input_path=str(router),
        input_sha256=sha(router),
        source_state=str(state_path),
        source_state_sha256=sha(state_path),
        product_commit=state.get("artifact", {}).get("revision"),
        method=joint.METHOD,
        helper_sha256=sha(__file__),
        math_helper_sha256=sha(joint.__file__),
    )
    (out / "fit-replay.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "fit-replay-rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in snapshots))
    print(
        json.dumps({k: result[k] for k in ("ok", "method", "counts", "deterministic_repeat_equal")})
    )


if __name__ == "__main__":
    main()
