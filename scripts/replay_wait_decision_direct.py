#!/usr/bin/env python3
"""Causal direct-basis counterfactual on previously audited immutable v1 traces.

Replays all original outcome opportunities, not the previously accepted fit rows.
No network, traffic, tuning or automatic promotion actions.
"""

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

try:
    from . import (
        wait_decision_calibration as legacy,
    )
    from . import (
        wait_decision_direct as direct,
    )
    from . import (
        wait_decision_direct_unpenalized as unpenalized,
    )
    from .replay_wait_decision import payload, score_admissions
    from .replay_wait_joint_gains import close, read_events, require, sha
except ImportError:
    import wait_decision_calibration as legacy
    import wait_decision_direct as direct
    import wait_decision_direct_unpenalized as unpenalized
    from replay_wait_decision import payload, score_admissions
    from replay_wait_joint_gains import close, read_events, require, sha


def replay_opportunities(events, *, direct_basis, method=direct.METHOD):
    require(method in (direct.METHOD, unpenalized.METHOD), "Unsupported direct replay method")
    engine = (unpenalized if method == unpenalized.METHOD else direct) if direct_basis else legacy
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
        return direct.snapshot_from_v1(value) if direct_basis else deepcopy(value)

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
        p = payload(e)
        s = state(e)
        now = e["diagnostic_time_seconds"]
        counts[kind] += 1
        if kind == "decision_label":
            expected_labels.append(dict(worker=worker, reason=e["reason"], label=p["label"]))
            continue
        if kind == "decision_poll":
            s.poll(now, p["profile"], p["fresh"], p["max_gap"])
            if not direct_basis:
                close(s.diagnostics(), p["state"], "Opportunity replay v1 poll state")
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
            if not direct_basis:
                close(
                    s.predictions(snap, now, p["available"]),
                    p["predictions"],
                    "Opportunity replay v1 prediction",
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
                if not direct_basis:
                    close(predictions, p["predictions"], "Opportunity replay v1 frozen admission")
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
    if not direct_basis:
        close(actual_labels, expected_labels, "All accepted opportunity labels equal v1")
    require(not selected, "Unconsumed selected snapshots")
    return dict(
        method=engine.METHOD,
        counts=dict(counts),
        admissions=rows,
        labels=labels,
        polls=polls,
        opportunities=opportunities,
        v1_parity=not direct_basis,
    )


def verify_sources(audit):
    receipt = json.loads((audit / "decision-replay.json").read_text())
    require(
        receipt["ok"]
        and receipt["deterministic_repeat_equal"]
        and receipt["method"] == legacy.METHOD,
        "Source is not an audited v1 capture",
    )
    state = Path(receipt["source_state"])
    require(sha(state) == receipt["source_state_sha256"], "Source state changed")
    capture = json.loads(state.read_text())
    require(
        all(
            capture.get(k) is True
            for k in ("results_valid", "original_restored", "owned_resources_deleted")
        ),
        "Source case is not closed/restored",
    )
    require(
        capture["host_checkpoint"]["owned_process_closed"] is True
        and capture["host_checkpoint"]["exit_code"] == 0,
        "Checkpoint not closed",
    )
    root = Path(capture["result_path"])
    require(receipt["source_files"] == capture["result_files"], "Source binding manifest differs")
    require(
        all(sha(root / name) == digest for name, digest in receipt["source_files"].items()),
        "Canonical source changed",
    )
    return receipt, root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--method", choices=(direct.METHOD, unpenalized.METHOD), default=direct.METHOD
    )
    args = parser.parse_args()
    engine = unpenalized if args.method == unpenalized.METHOD else direct
    require(not args.out.exists(), "Preserve existing replay output")
    source, root = verify_sources(args.audit)
    events = read_events(root / "router-events.jsonl")
    paired = [
        json.loads(line) for line in (root / "paired-production.jsonl").read_text().splitlines()
    ]
    parity = replay_opportunities(events, direct_basis=False)
    parity_rows, parity_scores = score_admissions(
        parity["admissions"], paired, root.name, root.parent.name
    )
    prior = [
        json.loads(line) for line in (args.audit / "admissions.jsonl").read_text().splitlines()
    ]
    require(
        [r["request_id"] for r in parity_rows] == [r["request_id"] for r in prior],
        "Changed source identities",
    )
    close(parity_scores, source["evaluation"], "V1 score parity")
    value = replay_opportunities(events, direct_basis=True, method=args.method)
    require(
        value == replay_opportunities(events, direct_basis=True, method=args.method),
        "Direct replay nondeterministic",
    )
    rows, evaluation = score_admissions(
        value["admissions"], paired, root.name, root.parent.name, args.method
    )
    for a, b in zip(rows, prior):
        require(
            a["request_id"] == b["request_id"] and a["truth"] == b["truth"],
            "Changed truth or identities",
        )
        close(
            a["total_predictions"]["live_n_scalar"],
            b["total_predictions"]["live_n_scalar"],
            "Unchanged comparator",
        )
        require(
            a["predictions"]["live_n_scalar"] == b["predictions"]["live_n_scalar"],
            "Changed baseline flag",
        )
        a["client_ttft_seconds"] = next(
            p.get("client_ttft_seconds") for p in paired if p["opc_request_id"] == a["request_id"]
        )
    value["admissions"] = rows
    verify_sources(args.audit)
    args.out.mkdir(parents=True)
    for key in ("admissions", "labels", "polls", "opportunities"):
        (args.out / (key + ".jsonl")).write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in value.pop(key))
        )
    value.update(
        evaluation=evaluation,
        source_audit=str(args.audit.resolve()),
        source_audit_sha256=sha(args.audit / "decision-replay.json"),
        source_state=source["source_state"],
        source_state_sha256=source["source_state_sha256"],
        source_files=source["source_files"],
        deterministic_repeat_equal=True,
        v1_parity=True,
        v1_accepted_labels=len(parity["labels"]),
        opportunity_count=len(parity["opportunities"]),
        method_source_sha256=sha(engine.__file__),
        solver_source_sha256=sha(
            (unpenalized if args.method == unpenalized.METHOD else legacy).__file__
        ),
        replay_source_sha256=sha(__file__),
        evidence_role=(
            "Previously inspected development captures; causal admission-clock "
            "counterfactual, not unseen validation or rejection outcomes"
        ),
    )
    (args.out / "direct-replay.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(dict(selection=evaluation["selection"], counts=value["counts"])))


if __name__ == "__main__":
    main()
