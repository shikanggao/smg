#!/usr/bin/env python3
"""Development-only forecast-clock counterfactuals on frozen historical captures."""

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

try:
    from . import wait_decision_calibration as d
    from .replay_wait_joint_gains import read_events, require, sha
    from .wait_decision_selection import compare, score
except ImportError:
    import wait_decision_calibration as d
    from replay_wait_joint_gains import read_events, require, sha
    from wait_decision_selection import compare, score


def replay(events, max_gap):
    states = {}
    epochs = {}
    polls = {}
    recoveries = {}
    gates = {}
    features = {}
    saved = {}
    late_exact = {}
    resolved = set()
    rows = []
    labels = []
    fitrows = []
    clockgaps = []

    def state(e):
        w = e["worker"]
        epoch = e["calibration_epoch"]
        if w not in states:
            states[w] = d.State()
            epochs[w] = epoch
        require(epochs[w] == epoch, "Epoch mismatch")
        return states[w]

    for index, e in enumerate(events):
        kind = e.get("event")
        w = e.get("worker")
        rid = e.get("request_id")
        key = (w, rid)
        now = e.get("diagnostic_time_seconds")
        if kind == "calibration_reset":
            states[w] = d.State()
            epochs[w] = e["calibration_epoch"]
            continue
        if kind == "forecast":
            gates[key] = e
        if kind == "recovery_forecast":
            recoveries[key] = e
        if kind == "calibration_poll":
            polls[w] = e
        if kind == "feature_context_poll":
            s = state(e)
            p = json.loads(e["feature_json"])
            fresh = p["fresh"] and polls[w]["outcome"] == "success"
            s.poll(now, p["profile"], fresh, max_gap)
            fitrows.append(dict(index=index, worker=w, time=now, state=deepcopy(s.diagnostics())))
        if kind == "feature_context_forecast":
            features[key] = e
        if kind == "forecast" and key in features:
            feature = features[key]
            now = feature["diagnostic_time_seconds"]
            s = state(feature)
            p = json.loads(feature["feature_json"])
            r = recoveries[key]
            gate = gates[key]
            live = r["queue_excess_seconds"] if r["available"] else None
            x = p["legacy_gain"] * (
                live if live is not None else p["original_queue_excess_seconds"]
            )
            baseline = p["baseline_seconds"]
            available = (
                p["fresh"] and p["native_ready"] and p["supported_path"] and baseline is not None
            )
            if baseline is None:
                baseline = 0.0
            snap = dict(
                at=now,
                profile=p["profile"],
                baseline=baseline,
                x=x,
                source="live_n_scalar" if live is not None else "original_n_scalar_fallback",
                eligible=available and gate.get("native_waiting", 0) > 0 and x >= 0.001,
                context=s.context(now),
            )
            saved[key] = snap
            predictions = s.predictions(snap, now, available)
            # Explicit scalar fallback is the recorded same-clock actual baseline.
            fallback = gate.get("calibrated_would_reject")
            scalar_total = p["baseline_seconds"] + x if p["baseline_seconds"] is not None else None
            scalar_flag = x >= p["threshold_seconds"] if available else fallback
            rows.append(
                dict(
                    event_index=index,
                    worker=w,
                    epoch=e["calibration_epoch"],
                    request_id=rid,
                    time=now,
                    snapshot=deepcopy(snap),
                    available=available,
                    threshold=p["threshold_seconds"],
                    predictions={
                        "live_n_scalar": scalar_flag,
                        **{
                            v["arm"]: (
                                v["extra"] >= p["threshold_seconds"] if v["available"] else fallback
                            )
                            for v in predictions
                        },
                    },
                    total_predictions={
                        "live_n_scalar": scalar_total
                        if available
                        else gate.get("calibrated_ttft_seconds"),
                        **{
                            v["arm"]: v["total"]
                            if v["available"]
                            else gate.get("calibrated_ttft_seconds")
                            for v in predictions
                        },
                    },
                    candidates=predictions,
                    fit_diagnostics=deepcopy(s.diagnostics()["models"]),
                )
            )
        if kind == "calibration_forecast" and key in saved:
            clockgaps.append(dict(request_id=rid, gap_seconds=now - saved[key]["at"]))
        if (
            kind == "feature_admission_decision"
            and e.get("decision") == "admitted"
            and e.get("outcome_tracking_supported") is False
        ):
            state(e).mark_incomplete()
        if kind == "feature_deadline_started":
            s = state(e)
            s.register(
                rid if e["deadline_tracking_supported"] else None,
                e["request_started_time_seconds"],
                saved.get(key),
            )
        if kind == "feature_deadline_resolution":
            s = state(e)
            source = e["resolution_source"]
            is_poll = source == "poll"
            if source in (
                "poll",
                "first_text",
                "upstream_timeout",
                "pretext_downstream_disconnect",
            ):
                label = s.label(
                    rid,
                    e["observed_time_seconds"],
                    now,
                    source == "first_text",
                    not is_poll,
                    300.0 if is_poll else None,
                )
                if label:
                    labels.append(dict(index=index, worker=w, reason=source, label=label))
            elif not is_poll:
                s.terminal_unknown(rid)
            if not is_poll:
                resolved.add(key)
        if kind == "feature_outcome":
            s = state(e)
            if e["outcome"] == "first_text" and key not in resolved and rid in s.pending:
                late_exact[key] = deepcopy(s.pending[rid])
            s.terminal_unknown(rid)
        if kind == "feature_context_feedback" and key in late_exact:
            s = state(e)
            p = json.loads(e["feature_json"])
            pending = late_exact.pop(key)
            # Exact label appears after its terminal event; accept at this later
            # publication clock, never backdate training into earlier forecasts.
            s.pending[rid] = pending
            label = s.label(rid, pending["started"] + p["ttft_seconds"], now, True, True)
            if label:
                labels.append(
                    dict(index=index, worker=w, reason="delayed_first_text_feedback", label=label)
                )
    return dict(rows=rows, labels=labels, fits=fitrows, clockgaps=clockgaps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Output already exists")
    args.out.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text())
    allrows = []
    receipts = []
    for capture in manifest["captures"]:
        for key in ("state", "router", "paired"):
            require(
                sha(capture[key + "_path"]) == capture[key + "_sha256"], "Manifest source changed"
            )
        state_path = Path(capture["state_path"])
        state = json.loads(state_path.read_text())
        root = Path(state["result_path"])
        require(state["results_valid"], "Invalid capture")
        for name, digest in state["result_files"].items():
            require(sha(root / name) == digest, "Canonical result changed")
        settings = state_path.parents[3] / "shared-settings.json"
        max_gap = float(json.loads(settings.read_text())["--estimated-wait-max-snapshot-age-secs"])
        events = read_events(Path(capture["router_path"]))
        result = replay(events, max_gap)
        require(result == replay(events, max_gap), "Nondeterministic history replay")
        paired = {
            r["opc_request_id"]: r
            for r in map(json.loads, Path(capture["paired_path"]).read_text().splitlines())
        }
        rows = []
        for row in result["rows"]:
            p = paired.get(row["request_id"])
            if p is None:
                continue
            truth = (
                None
                if p.get("controlled_rejection") or p.get("local_drop")
                else p.get("first_text_late")
            )
            require(type(truth) in (bool, type(None)), "Nonbinary truth")
            row.update(
                capture=capture["case"],
                model=capture["model"],
                role=capture["role"],
                phase=p["phase"],
                truth=truth,
                actual_ttft=p.get("smg_ttft_seconds"),
                arrival=p["started_elapsed_seconds"],
                actual_rejected=p.get("controlled_rejection", False),
            )
            rows.append(row)
        target = args.out / capture["case"]
        target.mkdir()
        for name, value in [
            ("admissions", rows),
            ("labels", result["labels"]),
            ("fits", result["fits"]),
            ("clockgaps", result["clockgaps"]),
        ]:
            (target / (name + ".jsonl")).write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in value)
            )
        cards = {
            arm: {
                phase: score([r for r in rows if phase == "all" or r["phase"] == phase], arm)[0]
                for phase in ["all"] + sorted({r["phase"] for r in rows})
            }
            for arm in ["live_n_scalar", *d.ARMS]
        }
        decisions = {arm: compare(rows, "live_n_scalar", arm) for arm in d.ARMS}
        receipt = dict(
            case=capture["case"],
            scores=cards,
            selection=decisions,
            settings_path=str(settings),
            settings_sha256=sha(settings),
            max_gap=max_gap,
            label_count=len(result["labels"]),
            label_kinds=dict(Counter(r["reason"] for r in result["labels"])),
            deterministic_repeat_equal=True,
            max_forecast_selected_gap_seconds=max(
                (r["gap_seconds"] for r in result["clockgaps"]), default=None
            ),
            source=capture,
        )
        (target / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        receipts.append(receipt)
        allrows += rows
        print(
            capture["case"],
            {arm: (cards[arm]["all"]["fp"], cards[arm]["all"]["fn"]) for arm in cards},
            flush=True,
        )
    summary = dict(
        method=d.METHOD,
        manifest_sha256=sha(args.manifest),
        historical_only=True,
        unseen_holdout=False,
        limitations=[
            "Forecast-clock counterfactual, not exact selected-admission product replay.",
            "Native progress unavailable historically; never imputed.",
            "Historical feature profile used as identity surrogate.",
            "Native cancellation after prior deadline bound retains existing bound without inventing later elapsed.",
            "No enforced-traffic counterfactual claims.",
        ],
        selection={arm: compare(allrows, "live_n_scalar", arm) for arm in d.ARMS},
        cases=receipts,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
