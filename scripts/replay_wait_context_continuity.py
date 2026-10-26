#!/usr/bin/env python3
"""Fixed causal v4 ablations on already-inspected closed v3 development traces."""

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

try:
    from . import wait_decision_context_continuity as current
    from . import wait_decision_direct_unpenalized as previous
    from .replay_wait_decision import payload, score_admissions
    from .replay_wait_joint_gains import close, read_events, require, sha
except ImportError:
    import wait_decision_context_continuity as current
    import wait_decision_direct_unpenalized as previous
    from replay_wait_decision import payload, score_admissions
    from replay_wait_joint_gains import close, read_events, require, sha
VARIANTS = ("previous", "transition_only", "range_only", "combined")


def replay_variant(events, variant):
    require(variant in VARIANTS, "Unknown frozen ablation")
    parity = variant == "previous"
    engine = (
        previous
        if parity
        else SimpleNamespace(
            METHOD=current.METHOD,
            State=lambda: current.State(
                preserve_transitions=variant != "range_only",
                range_fallback=variant != "transition_only",
            ),
        )
    )
    states, epochs, selected, identities = {}, {}, {}, {}
    rows, labels, polls, opportunities = [], [], [], []
    expected_labels, actual_labels = [], []
    counts = Counter()
    # Look-ahead only binds provenance of the same outcome transaction. Labels
    # are generated below at original publication time, never at admission.
    first = {e["request_id"]: e for e in events if e.get("event") == "first_text"}
    outcomes = {
        (e["worker"], e["calibration_epoch"], e["request_id"], e["diagnostic_time_seconds"]): e
        for e in events
        if e.get("event") == "feature_outcome"
    }

    def state(e):
        worker, epoch = e["worker"], e["calibration_epoch"]
        if worker not in states:
            states[worker], epochs[worker] = engine.State(), epoch
        require(epochs[worker] == epoch, "Epoch changed without reset")
        return states[worker]

    def snapshot(value):
        return deepcopy(value)

    def observe(e, identity, observed_at, reason, terminal):
        s = state(e)
        now = e["diagnostic_time_seconds"]
        pending = s.pending.get(identity)
        if pending is not None:
            if not terminal:
                require(observed_at - pending["started"] >= 300 - 1e-7, "Premature poll bound")
            else:
                admission = identities[(e["worker"], e["calibration_epoch"], identity)]
                proof = outcomes.get(
                    (e["worker"], e["calibration_epoch"], admission["request_id"], now)
                )
                require(
                    proof is not None and proof["outcome"] == reason,
                    "Outcome opportunity lacks provenance",
                )
                if reason == "first_text":
                    require(admission["request_id"] in first, "Exact opportunity lacks first text")
                    close(
                        first[admission["request_id"]]["ttft_seconds"],
                        observed_at - pending["started"],
                        "Opportunity exact elapsed",
                    )
        value = s.label(
            identity,
            observed_at,
            now,
            reason == "first_text",
            terminal,
            None if terminal else 300.0,
        )
        opportunities.append(
            dict(
                worker=e["worker"],
                epoch=e["calibration_epoch"],
                id=identity,
                at=now,
                observed_at=observed_at,
                reason=reason,
                accepted=value is not None,
            )
        )
        if value is not None:
            item = dict(worker=e["worker"], reason=reason, label=value)
            labels.append(item)
            actual_labels.append(item)

    for index, e in enumerate(events):
        kind, worker = e.get("event"), e.get("worker")
        if kind == "calibration_reset":
            states[worker], epochs[worker] = engine.State(), e["calibration_epoch"]
            for key in [key for key in selected if key[0] == worker]:
                selected.pop(key)
            counts["resets"] += 1
            continue
        if kind == "calibration_forecast":
            selected[(worker, e["request_id"])] = deepcopy(state(e))
        if (
            kind == "feature_admission_decision"
            and e.get("decision") == "admitted"
            and e.get("outcome_tracking_supported") is False
        ):
            state(e).mark_incomplete()
        if kind == "feature_deadline_resolution" and e.get("resolution_source") == "poll":
            key = (worker, e["calibration_epoch"], e["request_id"])
            admission = next(
                (a for (w, ep, _), a in identities.items() if (w, ep, a["request_id"]) == key), None
            )
            require(admission is not None, "Poll deadline lacks original admission")
            close(e["request_started_time_seconds"], admission["started"], "Poll original start")
            observe(e, admission["id"], e["observed_time_seconds"], "poll_deadline", False)
        if kind not in (
            "decision_poll",
            "decision_forecast",
            "decision_admission",
            "decision_label",
            "decision_terminal",
        ):
            continue
        p = payload(e, previous.METHOD)
        s = state(e)
        now = e["diagnostic_time_seconds"]
        counts[kind] += 1
        if kind == "decision_label":
            expected_labels.append(dict(worker=worker, reason=e["reason"], label=p["label"]))
            continue
        if kind == "decision_poll":
            s.poll(now, p["profile"], p["fresh"], p["max_gap"])
            if parity:
                close(s.diagnostics(), p["state"], "Reference v3 replay poll state")
            polls.append(
                dict(
                    worker=worker,
                    epoch=e["calibration_epoch"],
                    at=now,
                    state=deepcopy(s.diagnostics()),
                )
            )
        if kind in ("decision_poll", "decision_forecast") and p["snapshot"] is not None:
            snap = snapshot(p["snapshot"])
            c = snap["context"]
            close(
                s.context(
                    snap["at"],
                    c["progress_tokens_per_second"],
                    c["progress_capacity"],
                    c["values"][2] is not None,
                ),
                c,
                "Unchanged causal context",
            )
            if parity:
                close(
                    s.predictions(snap, now, p["available"]),
                    p["predictions"],
                    "Reference v3 replay prediction",
                )
        if kind == "decision_admission":
            saved = selected.pop((worker, e["request_id"]), None)
            snap = snapshot(p["snapshot"])
            predictions = None
            model_state = None
            if snap is not None:
                require(saved is not None, "Missing selected admission clock")
                predictions = saved.predictions(
                    snap, snap["at"], snap["fresh"] and snap["native_ready"]
                )
                model_state = deepcopy(saved.diagnostics()["models"])
                if parity:
                    close(predictions, p["predictions"], "Reference v3 replay frozen admission")
            require(
                s.register(p["id"], p["started"], snap) == p["tracked"],
                "Changed admission tracking",
            )
            row = dict(
                event_index=index,
                worker=worker,
                epoch=e["calibration_epoch"],
                request_id=e["request_id"],
                id=p["id"],
                tracked=p["tracked"],
                started=p["started"],
                snapshot=snap,
                predictions=predictions,
                selected_models=model_state,
            )
            rows.append(row)
            if p["id"] is not None:
                key = (worker, e["calibration_epoch"], p["id"])
                require(key not in identities, "Duplicate tracking identity")
                identities[key] = row
        if kind == "decision_terminal":
            if p["outcome"] in ("first_text", "upstream_timeout", "pretext_downstream_disconnect"):
                observe(e, p["id"], p["observed_at"], p["outcome"], True)
            else:
                s.terminal_unknown(p["id"])
    close(actual_labels, expected_labels, "All accepted opportunity labels unchanged from v3")
    require(not selected, "Unconsumed selected snapshots")
    return dict(
        method=engine.METHOD,
        counts=dict(counts),
        admissions=rows,
        labels=labels,
        polls=polls,
        opportunities=opportunities,
        v3_parity=parity,
    )


def verify_binding(path, digest, historical_planner=None):
    # The only replaceable location is the historical planner. Its expected
    # digest remains the immutable receipt's digest, never a supplied override.
    bound_path = (
        historical_planner
        if historical_planner is not None and path.endswith("/scripts/run_wait_production_mix.py")
        else path
    )
    require(sha(bound_path) == digest, "Qualified completeness source changed: " + path)


def verify_sources(audit, historical_planner=None):
    receipt = json.loads((audit / "decision-replay.json").read_text())
    review = json.loads((audit / "independent-review.json").read_text())
    require(
        receipt["ok"] and review["ok"] and receipt["method"] == previous.METHOD,
        "Source must be an independently audited closed v3 capture",
    )
    require(
        review["replay_sha256"] == sha(audit / "decision-replay.json")
        and review["admissions_sha256"] == sha(audit / "admissions.jsonl"),
        "Audit binding changed",
    )
    require(sha(receipt["source_state"]) == receipt["source_state_sha256"], "State changed")
    capture = json.loads(Path(receipt["source_state"]).read_text())
    require(
        all(
            capture.get(k) is True
            for k in ("results_valid", "original_restored", "owned_resources_deleted")
        ),
        "Capture not restored/valid",
    )
    checkpoint = capture["host_checkpoint"]
    require(checkpoint.get("owned_process_closed") is True, "Monitor not closed")
    qualification = review.get("qualified_completeness")
    if checkpoint["exit_code"] != 0:
        q = json.loads((audit / "qualified-completeness.json").read_text())
        require(
            q["pre_scoring_completeness"] is True
            and q["sidecar_exit_code"] == 2
            and q["sidecar_status"] == "STOPPED_PARTIAL"
            and not q["sidecar_continuity_claimed"],
            "Missing qualified canonical-completeness proof",
        )
        require(qualification == q, "Independent review lacks exact qualification")
        for path, digest in q["source_bindings"].items():
            verify_binding(path, digest, historical_planner)
    root = Path(capture["result_path"])
    require(
        receipt["source_files"] == capture["result_files"] == review["source_files"],
        "Manifest mismatch",
    )
    require(
        {"router-events.jsonl", "paired-production.jsonl"} <= set(receipt["source_files"]),
        "Unbound consumed source",
    )
    for name, digest in receipt["source_files"].items():
        require(sha(root / name) == digest, "Canonical source changed: " + name)
    return receipt, root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--historical-planner",
        type=Path,
        help="Exact hash-matching archived planner when its original repository path has advanced",
    )
    args = parser.parse_args()
    require(not args.out.exists(), "Preserve previous experiment output")
    source, root = verify_sources(args.audit, args.historical_planner)
    events = read_events(root / "router-events.jsonl")
    paired = [json.loads(x) for x in (root / "paired-production.jsonl").read_text().splitlines()]
    prior = [json.loads(x) for x in (args.audit / "admissions.jsonl").read_text().splitlines()]
    client = {r["opc_request_id"]: r.get("client_ttft_seconds") for r in paired}
    args.out.mkdir(parents=True)
    reference = None
    for variant in VARIANTS:
        value = replay_variant(events, variant)
        rows, evaluation = score_admissions(
            value["admissions"], paired, root.name, root.parent.name, value["method"]
        )
        require(len(rows) == len(prior), "Changed cohort size")
        for row, old in zip(rows, prior):
            require(
                row["request_id"] == old["request_id"] and row["truth"] == old["truth"],
                "Changed identities/truth",
            )
            require(row["snapshot"] == old["snapshot"], "Changed saved input/eligibility")
            require(
                row["total_predictions"]["live_n_scalar"]
                == old["total_predictions"]["live_n_scalar"]
                and row["predictions"]["live_n_scalar"] == old["predictions"]["live_n_scalar"],
                "Changed primary comparator",
            )
            row["client_ttft_seconds"] = client[row["request_id"]]
        if variant == "previous":
            close(evaluation, source["evaluation"], "Authoritative v3 scoring parity")
            reference = deepcopy(value)
        else:
            require(
                value["labels"] == reference["labels"]
                and value["opportunities"] == reference["opportunities"],
                "Changed label lifecycle",
            )
            for a, b in zip(value["admissions"], reference["admissions"]):
                if a["predictions"] is None:
                    require(b["predictions"] is None, "Changed prediction coverage")
                    continue
                for i in (0, 2):
                    stripped = {
                        k: v
                        for k, v in a["predictions"][i].items()
                        if k not in ("fallback_reason", "requested_within_fitted_ranges")
                    }
                    require(stripped == b["predictions"][i], "Queue prediction not byte-equivalent")
                    require(
                        a["selected_models"][i] == b["selected_models"][i], "Queue state changed"
                    )
            for a, b in zip(value["polls"], reference["polls"]):
                for i in (0, 2):
                    require(
                        a["state"]["models"][i] == b["state"]["models"][i], "Queue poll fit changed"
                    )
        folder = args.out / variant
        folder.mkdir()
        value["admissions"] = rows
        for key in ("admissions", "labels", "polls", "opportunities"):
            (folder / (key + ".jsonl")).write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in value.pop(key))
            )
        value.update(
            evaluation=evaluation,
            source_audit=str(args.audit),
            source_audit_sha256=sha(args.audit / "decision-replay.json"),
            source_state=source["source_state"],
            source_state_sha256=source["source_state_sha256"],
            source_files=source["source_files"],
            historical_planner=str(args.historical_planner) if args.historical_planner else None,
            historical_planner_sha256=sha(args.historical_planner)
            if args.historical_planner
            else None,
            variant=variant,
            queue_controls_byte_equivalent=True,
            label_lifecycle_unchanged=True,
            evidence_role=(
                "Previously inspected v3 development trace; causal counterfactual, "
                "not held-out or actual rejection evidence"
            ),
            source_qualification=json.loads(
                (args.audit / "independent-review.json").read_text()
            ).get("qualified_completeness"),
            replay_sha256=sha(__file__),
            method_module_sha256=sha(current.__file__),
        )
        (folder / "replay.json").write_text(json.dumps(value, indent=2) + "\n")
        print(json.dumps(dict(variant=variant, counts=value["counts"])), flush=True)
    verify_sources(args.audit, args.historical_planner)


if __name__ == "__main__":
    main()
