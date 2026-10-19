#!/usr/bin/env python3
"""Causal calibration ablations and offered-request admission scorecards.

Replay preserves recorded eligibility, epochs, raw forecasts and feedback.
It estimates predictions on a fixed trace, never closed-loop policy outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

try:
    from .analyze_wait_calibration import (
        applied_scale,
        audit,
        close,
        events,
        median_scale,
        mse_scale,
        TWO_WAVE_METHOD,
        TWO_WAVE_RATE,
        two_wave_trace,
    )
except ImportError:
    from analyze_wait_calibration import (
        applied_scale,
        audit,
        close,
        events,
        median_scale,
        mse_scale,
        TWO_WAVE_METHOD,
        TWO_WAVE_RATE,
        two_wave_trace,
    )

METHODS = ("median_feedback", "median_poll", "mse_feedback", "mse_poll")


def quantile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(fraction * len(values)) - 1)] if values else None


def errors(rows, method):
    values = [
        r[method] - r["smg_ttft_seconds"]
        for r in rows
        if r.get(method) is not None and r.get("smg_ttft_seconds") is not None
    ]
    return dict(
        n=len(values),
        mae=statistics.mean(map(abs, values)) if values else None,
        bias=statistics.mean(values) if values else None,
        **{f"p{q}_under": quantile([max(-v, 0) for v in values], q / 100) for q in (95, 99)},
        **{f"p{q}_absolute": quantile(list(map(abs, values)), q / 100) for q in (95, 99)},
    )


def state_key(event):
    return (
        event["worker"],
        event.get("current_calibration_epoch", event["calibration_epoch"]),
        event["calibration_state"],
    )


def replay(log, rows):
    checked = audit(log)
    if checked["ok"] and checked["calibration_method"] == TWO_WAVE_METHOD:
        return replay_two_wave(log, rows, checked)
    if not checked["ok"] or checked["calibration_method"] != "mse_poll":
        raise ValueError(f"Requires a passing current poll-fit trace: {checked}")
    changes = defaultdict(dict)
    forecasts, gates = {}, {}
    for event in log:
        kind = event.get("event")
        if kind == "calibration_forecast":
            rid = event["request_id"]
            if rid in forecasts:
                raise ValueError(f"Duplicate forecast {rid}")
            forecasts[rid] = event
        if kind == "forecast" and event.get("calibration_enabled"):
            rid = event["request_id"]
            if rid in gates:
                raise ValueError("This experiment requires one exact gate per request")
            gates[rid] = event
        if (
            kind == "calibration_update"
            or kind == "calibration_feedback"
            and event["reason"] == "accepted"
        ):
            changes[state_key(event)][(event["sequence_after"], event["fit_sequence_after"])] = (
                event
            )
    versions = {}
    diagnostics = []
    for key, updates in changes.items():
        history = deque(maxlen=32)
        trajectories = {method: (1.0, 1.0, 0.0, 0) for method in METHODS}
        versions[key + (0, 0)] = trajectories.copy()
        ages = deque(maxlen=32)
        for sequence, event in sorted(updates.items()):
            now = event["calibration_time_seconds"]
            is_poll = event["event"] == "calibration_update"
            if not is_poll:
                x = event["raw_excess_seconds"]
                history.append((max(event["ttft_seconds"] - event["baseline_seconds"], 0) / x, x))
                ages.append((now, event["feedback_age_seconds"]))
            for method in METHODS:
                if method.endswith("poll") == is_poll:
                    target = (median_scale if method.startswith("median") else mse_scale)(history)
                    trajectories[method] = (
                        applied_scale(trajectories[method], now),
                        target,
                        now,
                        len(history),
                    )
            versions[key + sequence] = trajectories.copy()
            diagnostics.append(
                dict(
                    worker=key[0],
                    epoch=key[1],
                    state=key[2],
                    time=now,
                    event=event["event"],
                    feedback_age=event.get("feedback_age_seconds"),
                    history_admission_age_p50=quantile([age + now - at for at, age in ages], 0.5),
                    scales={m: applied_scale(v, now) for m, v in trajectories.items()},
                    targets={m: v[1] for m, v in trajectories.items()},
                )
            )

    def predictions(f):
        values = {method: None for method in ("raw",) + METHODS}
        if f:
            key = state_key(f)
            seq = (f["calibration_observation_sequence"], f["calibration_fit_sequence"])
            previous = versions.get(key + seq)
            if previous is None and seq == (0, 0):
                previous = {m: (1.0, 1.0, 0.0, 0) for m in METHODS}
            if previous is None:
                raise ValueError(f"Missing causal version {key + seq}")
            now = f["calibration_time_seconds"]
            for method, version in previous.items():
                if now < version[2]:
                    raise ValueError("Prediction used future feedback")
                value = (
                    f["baseline_seconds"]
                    + applied_scale(version, now) * f["predicted_excess_seconds"]
                )
                if method == "mse_poll" and not close(value, f["calibrated_ttft_seconds"]):
                    raise ValueError("Replay does not reproduce live prediction")
                if f.get("baseline_available", True):
                    values[method] = value
            if f.get("baseline_available", True):
                values["raw"] = f["predicted_ttft_seconds"]
        return values

    result = []
    offsets = defaultdict(list)
    for row in rows:
        row = dict(row)
        f = forecasts.get(row["opc_request_id"])
        gate = gates.get(row["opc_request_id"])
        row.update(predictions(f))
        row["gate_predictions"] = predictions(gate)
        row["gate_ready"] = gate["ready"] if gate else None
        if f:
            row["baseline_seconds"] = f["baseline_seconds"]
            row["calibration_epoch"] = f["calibration_epoch"]
            offsets[state_key(f)].append(
                row["started_elapsed_seconds"] - f["calibration_time_seconds"]
            )
        result.append(row)
    for item in diagnostics:
        values = offsets.get((item["worker"], item["epoch"], item["state"]))
        item["elapsed_seconds"] = item["time"] + statistics.median(values) if values else None
    return result, diagnostics, checked


def replay_two_wave(log, rows, checked):
    """All ablations retain recorded two-wave eligibility/lifecycle and 0.5/s gain."""
    _, records = two_wave_trace(log, keep_records=True)
    forecasts = {e["request_id"]: e for e in log if e.get("event") == "calibration_forecast"}
    gates = {e["request_id"]: e for e in log if e.get("event") == "forecast" and e.get("calibration_enabled")}
    versions, trajectories, diagnostic = {}, {}, []
    initial = lambda: {m: (1.0, 1.0, 0.0, 0) for m in METHODS}
    for record in records:
        event, key = record["event"], record["key"]
        kind = event["event"]
        if kind == "calibration_feedback" and event["reason"] != "accepted":
            continue
        now = event["calibration_time_seconds"]
        history = record["samples"]
        current = trajectories.setdefault(key, initial())
        if kind == "calibration_rotation" and record["removed"] and len(history) < 4:
            current.update({m: (1.0, 1.0, now, len(history)) for m in METHODS})
        elif kind != "calibration_rotation":
            for method in METHODS:
                if method.endswith("poll") == (kind == "calibration_update"):
                    target = (median_scale if method.startswith("median") else mse_scale)([(ratio, x) for ratio, x, _, _ in history])
                    current[method] = (applied_scale(current[method], now, TWO_WAVE_RATE), target, now, len(history))
        versions[key + (record["history_revision"], record["fitted_revision"])] = current.copy()
        diagnostic.append(dict(worker=key[0], epoch=key[1], state=key[2], time=now, event=kind,
            feedback_age=event.get("feedback_age_seconds"),
            history_admission_age_p50=quantile([age + now - at for _, _, at, age in history], 0.5),
            scales={m: applied_scale(v, now, TWO_WAVE_RATE) for m, v in current.items()}, targets={m: v[1] for m, v in current.items()},
            bank_index=record["bank_index"], previous_sample_count=record["previous_sample_count"], current_sample_count=record["current_sample_count"],
            history_revision=record["history_revision"], fitted_revision=record["fitted_revision"],
            removed_samples=record["removed"], gain_rate=TWO_WAVE_RATE))

    def predictions(forecast):
        values = {method: None for method in ("raw",) + METHODS + (TWO_WAVE_METHOD,)}
        if forecast is None:
            return values
        key = state_key(forecast)
        revision = (forecast["calibration_history_revision"], forecast["calibration_fitted_revision"])
        previous = versions.get(key + revision, initial() if revision == (0, 0) else None)
        if previous is None:
            raise ValueError(f"Missing causal revision {key + revision}")
        now = forecast["calibration_time_seconds"]
        for method, version in previous.items():
            if now < version[2]:
                raise ValueError("Prediction used future feedback")
            value = forecast["baseline_seconds"] + applied_scale(version, now, TWO_WAVE_RATE) * forecast["predicted_excess_seconds"]
            if method == "mse_poll" and not close(value, forecast["calibrated_ttft_seconds"]):
                raise ValueError("Two-wave replay does not reproduce live prediction")
            if forecast.get("baseline_available", True):
                values[method] = value
        if forecast.get("baseline_available", True):
            values["raw"] = forecast["predicted_ttft_seconds"]
        values[TWO_WAVE_METHOD] = values["mse_poll"]
        return values

    result, offsets = [], defaultdict(list)
    for row in rows:
        row = dict(row)
        forecast, gate = forecasts.get(row["opc_request_id"]), gates.get(row["opc_request_id"])
        row.update(predictions(forecast))
        row["gate_predictions"] = predictions(gate)
        row["gate_ready"] = gate["ready"] if gate else None
        row["pre_admission_gate_flags"] = {name: gate.get(field) if gate else None for name, field in (("raw", "would_reject"), ("calibrated", "calibrated_would_reject"))}
        row.pop("calibration_epoch", None)
        if forecast:
            row.update(baseline_seconds=forecast["baseline_seconds"], calibration_epoch=forecast["calibration_epoch"],
                calibration_bank_index=forecast["calibration_bank_index"], calibration_history_revision=forecast["calibration_history_revision"],
                calibration_fitted_revision=forecast["calibration_fitted_revision"])
            if row.get("started_elapsed_seconds") is not None:
                offsets[state_key(forecast)].append(row["started_elapsed_seconds"] - forecast["calibration_time_seconds"])
        result.append(row)
    for item in diagnostic:
        values = offsets.get((item["worker"], item["epoch"], item["state"]))
        item["elapsed_seconds"] = item["time"] + statistics.median(values) if values else None
    return result, diagnostic, checked


def scorecard(rows, deadline):
    n = len(rows)
    complete = [r for r in rows if r.get("complete") and r.get("e2e_seconds", deadline) < deadline]
    answers = [r for r in complete if r.get("first_answer_seconds") is not None]
    return dict(
        offered=n,
        completed_before_deadline=len(complete),
        completed_per_offered=len(complete) / n if n else None,
        completed_with_content=len(answers),
        content_completions_per_offered=len(answers) / n if n else None,
        natural_finish_with_content=sum(r.get("finish_reason") == "stop" for r in answers),
        output_limit_completions=sum(r.get("finish_reason") == "length" for r in complete),
        completed_late=sum(
            bool(r.get("complete")) and r.get("e2e_seconds", 0) >= deadline for r in rows
        ),
        timeouts=sum(bool(r.get("timed_out")) for r in rows),
        rejected=sum(bool(r.get("controlled_rejection")) for r in rows),
        other=sum(
            not (r.get("complete") or r.get("timed_out") or r.get("controlled_rejection"))
            for r in rows
        ),
        observed_completed_latency_p95=quantile([r["e2e_seconds"] for r in complete], 0.95),
        latency_note="Completed-only percentile; rejected and timeout latencies are not imputed.",
    )


def sweep(rows, deadline, reserves):
    """A diagnostic total-TTFT + fixed completion reserve rule, not raw extra gate."""
    result = []
    for method in ("raw",) + METHODS:
        for reserve in reserves:
            counts = dict(tp=0, fp=0, tn=0, fn=0, unknown=0, unready=0)
            for r in rows:
                prediction = r.get("gate_predictions", {}).get(method)
                if (
                    prediction is None
                    or r.get("completion_late") is None
                    or r.get("controlled_rejection")
                    or r.get("local_drop")
                ):
                    counts["unknown"] += 1
                    continue
                if not r.get("gate_ready"):
                    counts["unready"] += 1
                reject = bool(r.get("gate_ready")) and prediction + reserve >= deadline
                late = r["completion_late"]
                counts["tp" if reject and late else "fp" if reject else "fn" if late else "tn"] += 1
            result.append(
                dict(
                    method=method,
                    reserve_seconds=reserve,
                    **counts,
                    late_recall=counts["tp"] / (counts["tp"] + counts["fn"])
                    if counts["tp"] + counts["fn"]
                    else None,
                    healthy_reject_rate=counts["fp"] / (counts["fp"] + counts["tn"])
                    if counts["fp"] + counts["tn"]
                    else None,
                )
            )
    return result


def reset_timeline(log, rows):
    started = {r["opc_request_id"]: r.get("started_elapsed_seconds") for r in rows}
    offsets = [
        datetime.fromisoformat(e["timestamp"]).timestamp() - started[e["request_id"]]
        for e in log
        if e.get("event") == "calibration_forecast" and started.get(e.get("request_id")) is not None
    ]
    if not offsets:
        return []
    origin = statistics.median(offsets)
    return [
        dict(
            elapsed_seconds=datetime.fromisoformat(e["timestamp"]).timestamp() - origin,
            reason=e["reason"],
            epoch=e["calibration_epoch"],
            time_precision="approximately 1 second",
        )
        for e in log
        if e.get("event") == "calibration_reset"
        and datetime.fromisoformat(e["timestamp"]).timestamp() >= origin
    ]


def plot(rows, diagnostic, stages, resets, output, two_wave=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6 if two_wave else 4, 1, figsize=(14, 19 if two_wave else 13), sharex=True)
    for method in METHODS:
        # Separate epochs/banks so lifecycle discontinuities are not interpolated.
        groups = defaultdict(list)
        for d in diagnostic:
            if d["elapsed_seconds"] is not None:
                groups[(d["worker"], d["epoch"], d["state"], d.get("bank_index"))].append(d)
        for i, values in enumerate(groups.values()):
            color = dict(zip(METHODS, ("C0", "C1", "C2", "C3")))[method]
            axes[0].plot(
                [d["elapsed_seconds"] for d in values],
                [d["scales"][method] for d in values],
                color=color,
                label=(TWO_WAVE_METHOD + " (live)" if two_wave and method == "mse_poll" else method) if i == 0 else None,
                linewidth=0.8,
            )
            if method == "mse_poll":
                axes[0].plot(
                    [d["elapsed_seconds"] for d in values],
                    [d["targets"][method] for d in values],
                    color=color,
                    linestyle=":",
                    alpha=0.5,
                    label="MSE poll target" if i == 0 else None,
                )
    observed = [
        d for d in diagnostic if d["elapsed_seconds"] is not None and d["feedback_age"] is not None
    ]
    axes[1].scatter(
        [d["elapsed_seconds"] for d in observed],
        [d["feedback_age"] for d in observed],
        s=2,
        label="Age at feedback arrival",
    )
    fitted = defaultdict(list)
    for d in diagnostic:
        if d["elapsed_seconds"] is not None and d["event"] == "calibration_update":
            fitted[(d["worker"], d["epoch"], d["state"], d.get("bank_index"))].append(d)
    for i, values in enumerate(fitted.values()):
        axes[1].plot(
            [d["elapsed_seconds"] for d in values],
            [d["history_admission_age_p50"] for d in values],
            color="C3",
            label="Median age of admissions in fitting window" if i == 0 else None,
        )
    for method in ("raw", "median_feedback", "mse_poll"):
        exact = [
            r for r in rows if r.get(method) is not None and r.get("smg_ttft_seconds") is not None
        ]
        axes[2].scatter(
            [r["started_elapsed_seconds"] for r in exact],
            [r[method] - r["smg_ttft_seconds"] for r in exact],
            s=2,
            alpha=0.35,
            label=method,
        )
    axes[2].axhline(0, color="black", linewidth=0.5)
    for phase in dict.fromkeys(r["phase"] for r in rows):
        rr = [
            r for r in rows
            if r["phase"] == phase and r.get("started_elapsed_seconds") is not None
            and r.get("calibration_epoch") is not None
        ]
        times = [r["started_elapsed_seconds"] for r in rr]
        if times:
            axes[3].scatter(
                times,
                [int(r["calibration_epoch"]) for r in rr],
                s=2,
                label=phase,
            )
    labels = ["Calibration gain", "Feedback age (seconds)", "Predicted − actual TTFT (s)", "Selected/admitted epoch (known only)"]
    if two_wave:
        labels += ["Retained samples (max 64)", "History / fitted revision"]
        groups = defaultdict(list)
        for d in diagnostic:
            if d["elapsed_seconds"] is not None:
                groups[(d["worker"], d["epoch"], d["state"])].append(d)
        bank_axis = axes[4].twinx()
        bank_axis.set_ylabel("Bank index")
        for i, values in enumerate(groups.values()):
            times = [d["elapsed_seconds"] for d in values]
            for name, color in (("previous_sample_count", "C0"), ("current_sample_count", "C1")):
                axes[4].step(times, [d[name] for d in values], where="post", color=color, label=name if i == 0 else None)
            bank_axis.step(times, [d["bank_index"] for d in values], where="post", color="gray", linestyle=":", label="bank index" if i == 0 else None)
            for name, color in (("history_revision", "C2"), ("fitted_revision", "C3")):
                axes[5].step(times, [d[name] for d in values], where="post", color=color, label=name if i == 0 else None)
        rotations = sorted({d["elapsed_seconds"] for d in diagnostic if d["event"] == "calibration_rotation" and d["elapsed_seconds"] is not None})
        for i, at in enumerate(rotations):
            axes[4].axvline(at, color="teal", alpha=0.25, linestyle="--", label="Bank rotation; same epoch" if i == 0 else None)
        bank_axis.set_yscale("linear")
        bank_axis.legend(fontsize=7, loc="upper right")
    for ax, label in zip(
        axes,
        labels,
    ):
        ax.set_ylabel(label)
        ax.set_yscale("linear")
        ax.set_xscale("linear")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="upper left", ncol=3)
        for stage in stages:
            ax.axvline(stage["elapsed_seconds"], color="gray", alpha=0.2)
    for reset in resets:
        axes[3].axvline(
            reset["elapsed_seconds"],
            color="red" if reset["reason"] == "history_expired" else "gray",
            alpha=0.8 if reset["reason"] == "history_expired" else 0.15,
            linestyle=":",
        )
        if reset["reason"] == "history_expired":
            axes[3].annotate(
                "600s expiry (±1s)",
                (reset["elapsed_seconds"], axes[3].get_ylim()[1]),
                xytext=(3, -15),
                textcoords="offset points",
                fontsize=8,
                rotation=90,
                va="top",
            )
    axes[-1].set_xlabel(
        "Elapsed client time (s); feedback panels use observation time, errors use admission time"
    )
    lifecycle = " — all ablations: recorded two-wave banks, 0.5 gain/s" if two_wave else ""
    fig.suptitle(output.parent.name + " — fixed-trace causal ablations; no policy intervention" + lifecycle)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def analyze(source, output, plots=False):
    summary = json.loads((source / "summary.json").read_text())
    if summary["execution_status"] != "PASS":
        raise ValueError("Invalid execution excluded from policy conclusions")
    rows = [json.loads(line) for line in (source / "paired-production.jsonl").open()]
    log = list(events(source / "router-events.jsonl"))
    rows, diagnostics, checked = replay(log, rows)
    profile = json.loads((source / "profile.json").read_text())
    deadline = profile["deadline_seconds"]
    two_wave = checked["calibration_method"] == TWO_WAVE_METHOD
    methods = ("raw",) + METHODS + ((TWO_WAVE_METHOD,) if two_wave else ())
    groups = {"all": rows}
    groups.update(
        {
            "phase:" + phase: [r for r in rows if r["phase"] == phase]
            for phase in dict.fromkeys(r["phase"] for r in rows)
        }
    )
    groups.update(
        {
            "cohort:" + cohort: [r for r in rows if r["cohort"] == cohort]
            for cohort in dict.fromkeys(r["cohort"] for r in rows)
        }
    )
    resets = reset_timeline(log, rows)
    result = dict(
        source=str(source),
        audit=checked,
        resets=resets,
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation=(
            "All new-trace ablations use recorded two-wave acceptance-time banks (300s, previous/current32 each), fixed eligibility and0.5/s gain. "
            "mse_poll_two_wave_v1 is the explicit live alias of mse_poll. No ablation represents the old binary or old32-sample/600s-epoch lifecycle."
        ) if two_wave else (
            "Fixed eligibility, raw state and 600s epochs; historical median algorithm with current lifecycle. "
            "Not a historical binary replay or closed-loop policy simulation."
        ),
        errors={group: {m: errors(rr, m) for m in methods} for group, rr in groups.items()},
        scorecards={group: scorecard(rr, deadline) for group, rr in groups.items()},
        reserve_rule=(
            "Exact pre-admission gate: ready AND predicted TTFT + reserve >= client deadline; unknown remains unknown. "
            "This differs from the existing extra-TTFT gate."
        ),
        reserve_sweep=sweep(rows, deadline, (0, 30, 60, 90, 120, 150)),
        reserve_warning=(
            "Classification only: no claim that hypothetical rejects prevent timeouts or increase completion. "
            "Select on training runs and validate on independent closed-loop repeats."
        ),
        input_sha256={
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in (
                "router-events.jsonl",
                "paired-production.jsonl",
                "profile.json",
                "summary.json",
            )
        },
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, values in (("replay.jsonl", rows), ("transitions.jsonl", diagnostics)):
        with (output / name).open("w") as stream:
            for row in values:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
    if plots:
        plot(
            rows,
            diagnostics,
            json.loads((source / "stages.json").read_text()),
            resets,
            output / "transitions.png",
            two_wave=two_wave,
        )
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for method in ("raw", "median_feedback", "mse_poll"):
            points = [
                r
                for r in result["reserve_sweep"]
                if r["method"] == method
                and r["late_recall"] is not None
                and r["healthy_reject_rate"] is not None
            ]
            if points:
                ax.plot(
                    [r["healthy_reject_rate"] * 100 for r in points],
                    [r["late_recall"] * 100 for r in points],
                    "o-",
                    label=method,
                )
                if method == "mse_poll":
                    for r in points:
                        ax.annotate(
                            str(r["reserve_seconds"]) + "s",
                            (r["healthy_reject_rate"] * 100, r["late_recall"] * 100),
                            xytext=(4, 3),
                            textcoords="offset points",
                            fontsize=8,
                        )
        if ax.lines:
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "No observed late completions: recall is unavailable",
                transform=ax.transAxes,
                ha="center",
            )
        ax.set(
            xlabel="Known healthy requests flagged (%)",
            ylabel="Known late requests flagged (%)",
            title="Fixed-trace completion-reserve sweep\nClassification only; no simulated admission benefit",
        )
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output / "reserve-tradeoff.png", dpi=160)
        plt.close(fig)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    result = analyze(args.input, args.output, args.plots)
    print(
        json.dumps(
            dict(scorecard=result["scorecards"]["all"], errors=result["errors"]["all"]), indent=2
        )
    )


if __name__ == "__main__":
    main()
