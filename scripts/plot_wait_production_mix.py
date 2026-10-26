#!/usr/bin/env python3
"""Plot the production-mix plan or measured run; never invent timeout latency."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path):
    with path.open() as stream:
        return [
            {k: v for k, v in json.loads(line).items() if not k.endswith("_prometheus")}
            for line in stream
        ]


def values(rows, key):
    return [r.get(key) if r.get(key) is not None else math.nan for r in rows]


def render(path, output=None, linear=False):
    output = output or path / "graphs"
    time_scale = "linear" if linear else "symlog"
    output.mkdir(parents=True, exist_ok=True)
    profile = json.loads((path / "profile.json").read_text())
    ttft_only = profile.get("measurement_target") == "ttft"
    plan = read_rows(path / "plan.jsonl")
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in [c["name"] for c in profile["cohorts"]]:
        rows = [r for r in plan if r["cohort"] == name]
        axes[0].scatter(
            values(rows, "input_words"),
            values(rows, "max_output_tokens"),
            s=4,
            alpha=0.25,
            label=name,
        )
    axes[0].set(xlabel="Input words (not tokens)", ylabel="Output token limit (not forced output)")
    axes[0].legend(fontsize=7)
    start = 0
    for phase in profile["phases"]:
        axes[1].plot([start, start + phase["seconds"]], [phase["rps"]] * 2, linewidth=3)
        axes[1].text(
            start + phase["seconds"] / 2,
            phase["rps"],
            phase["name"],
            rotation=45,
            fontsize=7,
            va="bottom",
        )
        start += phase["seconds"]
    recovery = next((phase for phase in profile["phases"] if phase["name"] == "recovery"), None)
    arrival_label = "Nominal arrival time (s)"
    if recovery:
        arrival_label += (
            "; recovery starts after drain"
            if recovery.get("drain_before")
            else "; recovery during ongoing traffic"
        )
    axes[1].set(xlabel=arrival_label, ylabel="Offered requests/s")
    fig.suptitle(
        "Synthetic production-style plan — proportions are not measured production traffic"
    )
    fig.tight_layout()
    name = "production-mix-plan.png"
    fig.savefig(output / name, dpi=160)
    plt.close(fig)
    figures.append(name)
    inputs = ["profile.json", "plan.jsonl"]
    if (path / "summary.json").exists():
        summary = json.loads((path / "summary.json").read_text())
        config = json.loads((path / "run-config.json").read_text())
        rows = read_rows(path / "paired-production.jsonl")
        rows = [r for r in rows if r.get("started_elapsed_seconds") is not None]
        metrics = read_rows(path / "metrics.jsonl")
        stages = json.loads((path / "stages.json").read_text())
        deadline = profile["deadline_seconds"]
        title = (
            "SMOKE / MOCK — not production evidence"
            if summary["smoke"]
            else "Synthetic production mix"
        )
        title += (
            f" | execution {summary['execution_status']}; coverage {summary['production_coverage']}"
        )
        title += f"\n{config['engine']} · {config['run_id']}"
        t = values(metrics, "sample_completed_elapsed_seconds")
        fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
        comparisons = [
            ("raw", "tab:orange", "Raw floor-inclusive total"),
            ("queue_only_raw", "tab:green", "Queue-only uncalibrated B + N"),
            ("calibrated", "tab:cyan", "Queue-only calibrated B + gN"),
        ]
        if ttft_only:
            # The old raw gauge can exist without a measured baseline. Only
            # independently available exact totals belong on a total-TTFT plot.
            for key, color, label in comparisons:
                known = [
                    r
                    for r in rows
                    if r.get("pre_admission_total_ttft_predictions", {}).get(key) is not None
                ]
                axes[0].scatter(
                    values(known, "started_elapsed_seconds"),
                    [r["pre_admission_total_ttft_predictions"][key] for r in known],
                    s=10,
                    marker="x",
                    color=color,
                    label=label + f", exact pre-admission (n={len(known)})",
                )
        else:
            axes[0].plot(
                t, values(metrics, "raw_ttft"), color="tab:orange", label="Raw TTFT, sampled gauge"
            )
            axes[0].plot(
                t,
                values(metrics, "calibrated_ttft"),
                color="tab:cyan",
                label="Current TTFT, sampled gauge",
            )
        axes[0].scatter(
            values(rows, "started_elapsed_seconds"),
            values(rows, "client_ttft_seconds"),
            s=12,
            color="navy",
            label="Client first text, measured",
        )
        if not ttft_only:
            paired = [r for r in rows if r.get("raw_ttft_seconds") is not None]
            axes[0].scatter(
                values(paired, "started_elapsed_seconds"),
                values(paired, "raw_ttft_seconds"),
                s=10,
                marker="x",
                color="tab:orange",
                label="Raw, exact request snapshot",
            )
            axes[0].scatter(
                values(paired, "started_elapsed_seconds"),
                values(paired, "calibrated_ttft_seconds"),
                s=10,
                marker="x",
                color="tab:cyan",
                label="Current, exact request snapshot",
            )
        censored = [
            r
            for r in rows
            if r.get("timed_out")
            and r.get("first_text_late") is True
            and r.get("client_ttft_seconds") is None
        ]
        axes[0].scatter(
            values(censored, "started_elapsed_seconds"),
            [r.get("observation_horizon_seconds", deadline) for r in censored],
            marker="^",
            color="red",
            s=15,
            label="No first text by observation horizon: lower bound only",
        )
        axes[0].set(ylabel="First-text TTFT (s)", yscale=time_scale, ylim=(0, None))
        axes[0].legend(fontsize=7, ncol=2)
        complete = [r for r in rows if r.get("complete")]
        timed_out = [r for r in rows if r.get("timed_out") and r.get("completion_late") is True]
        if ttft_only:
            observed = [r for r in rows if r.get("client_ttft_seconds") is not None]
            axes[1].scatter(
                values(observed, "started_elapsed_seconds"),
                values(observed, "e2e_seconds"),
                s=10,
                label="Observation ended after first text; generation not the target",
            )
            axes[1].scatter(
                values(censored, "started_elapsed_seconds"),
                values(censored, "observation_horizon_seconds"),
                marker="^",
                color="red",
                s=15,
                label="Horizon reached without first text",
            )
            axes[1].set(ylabel="Observation duration (s)", yscale=time_scale, ylim=(0, None))
        else:
            axes[1].scatter(
                values(complete, "started_elapsed_seconds"),
                values(complete, "e2e_seconds"),
                s=10,
                label="Full completion: finish reason + DONE",
            )
            axes[1].scatter(
                values(timed_out, "started_elapsed_seconds"),
                [deadline] * len(timed_out),
                marker="^",
                color="red",
                s=15,
                label="Client canceled at deadline; exact completion unknown",
            )
            axes[1].set(ylabel="Client completion (s)", yscale=time_scale, ylim=(0, None))
        axes[1].legend(fontsize=8)
        axes[2].plot(t, values(metrics, "native_waiting"), label="Native waiting")
        axes[2].plot(t, values(metrics, "native_running"), label="Native running")
        axes[2].plot(t, values(metrics, "client_active"), label="Client active")
        kv = axes[2].twinx()
        kv.plot(t, values(metrics, "native_kv"), color="purple", alpha=0.5, label="KV usage")
        kv.set(ylabel="KV fraction", ylim=(0, 1.05))
        axes[2].set(ylabel="Requests")
        axes[2].legend(fontsize=8, loc="upper left")
        kv.legend(fontsize=8, loc="upper right")
        axes[3].scatter(
            values(rows, "scheduled_elapsed_seconds"), values(rows, "dispatch_lag_seconds"), s=8
        )
        axes[3].axhline(
            profile["maximum_p99_dispatch_lag_seconds"],
            color="red",
            linestyle="--",
            label="p99 validity limit",
        )
        axes[3].set(xlabel="Elapsed client time (s)", ylabel="Producer dispatch lag (s)")
        axes[3].legend(fontsize=8)
        for index, ax in enumerate(axes[:2]):
            ax.axhline(
                deadline,
                color="red",
                linestyle="--",
                linewidth=0.8,
                label=f"{deadline:g}-second client deadline",
            )
            ax.set_ylim(0, max(deadline * 1.1, ax.get_ylim()[1]))
            ax.legend(fontsize=7 if index == 0 else 8, ncol=2)
        for ax in axes:
            ax.grid(alpha=0.15)
            for stage in stages:
                ax.axvline(stage["elapsed_seconds"], color="gray", alpha=0.2, linewidth=0.7)
        fig.suptitle(
            title + "\nPer-request deadlines, continuous arrivals, no global pressure unload"
        )
        fig.tight_layout()
        name = "production-mix-timeline.png"
        fig.savefig(output / name, dpi=160)
        plt.close(fig)
        figures.append(name)
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        if ttft_only:
            exact = [r for r in rows if r.get("client_ttft_seconds") is not None]
            predictions = []
            for key, color, label in comparisons:
                known = [
                    r
                    for r in exact
                    if r.get("pre_admission_total_ttft_predictions", {}).get(key) is not None
                ]
                y = [r["pre_admission_total_ttft_predictions"][key] for r in known]
                predictions += y
                axes[0].scatter(
                    values(known, "client_ttft_seconds"),
                    y,
                    s=10,
                    alpha=0.6,
                    color=color,
                    label=f"{label} (n={len(known)})",
                )
            maximum = max([deadline] + values(exact, "client_ttft_seconds") + predictions)
            axes[0].axhline(deadline, color="red", linestyle=":", linewidth=0.8)
            axes[0].axvline(deadline, color="red", linestyle=":", linewidth=0.8)
        else:
            exact = [
                r
                for r in rows
                if r.get("smg_ttft_seconds") is not None and r.get("raw_ttft_seconds") is not None
            ]
            for key, color, label in [
                ("raw_ttft_seconds", "tab:orange", "Raw"),
                ("calibrated_ttft_seconds", "tab:cyan", "Current"),
            ]:
                axes[0].scatter(
                    values(exact, "smg_ttft_seconds"),
                    values(exact, key),
                    s=10,
                    alpha=0.6,
                    color=color,
                    label=label,
                )
            maximum = max(
                [deadline]
                + [
                    r[k]
                    for r in exact
                    for k in ["smg_ttft_seconds", "raw_ttft_seconds", "calibrated_ttft_seconds"]
                    if r.get(k) is not None
                ]
            )
        axes[0].plot([0, maximum], [0, maximum], "k--", linewidth=0.8)
        axes[0].set(
            xlabel="Actual client first text (s); numeric observations only"
            if ttft_only
            else "Actual SMG first text (s); exact pairs only",
            ylabel="Pre-admission total estimate (s)" if ttft_only else "Selected estimate (s)",
            xscale=time_scale,
            yscale=time_scale,
            xlim=(0, None),
            ylim=(0, None),
        )
        axes[0].text(
            0.03,
            0.97,
            (
                f"{'Observed TTFTs' if ttft_only else 'Exact pairs'}: {len(exact)}\nCensored "
                f"actual TTFTs are omitted, not invented"
            ),
            transform=axes[0].transAxes,
            va="top",
            fontsize=8,
        )
        axes[0].legend(fontsize=8)
        for cohort in profile["cohorts"]:
            admitted = [
                r
                for r in rows
                if r["cohort"] == cohort["name"]
                and (
                    r.get("first_text_late") is not None
                    if ttft_only
                    else r.get("complete")
                    or (r.get("timed_out") and r.get("completion_late") is True)
                )
            ]
            observed = (
                sorted(
                    r["client_ttft_seconds"]
                    for r in admitted
                    if r.get("client_ttft_seconds") is not None
                )
                if ttft_only
                else sorted(r["e2e_seconds"] for r in admitted if r.get("complete"))
            )
            if admitted:
                end = max([deadline] + observed)
                axes[1].step(
                    [0] + observed + [end],
                    [0]
                    + [(i + 1) / len(admitted) for i in range(len(observed))]
                    + [len(observed) / len(admitted)],
                    where="post",
                    label=cohort["name"],
                )
        axes[1].axvline(deadline, color="red", linestyle="--")
        axes[1].set(
            xlabel="Observed first-text TTFT (s)"
            if ttft_only
            else "Observed full completion time (s)",
            ylabel="Observed / known first-text outcomes"
            if ttft_only
            else "Completed / all completed-or-timeout requests",
            ylim=(0, 1.05),
        )
        axes[1].text(
            0.03,
            0.03,
            "Unfinished timeouts remain in the denominator.\nPlateaus below 1 show the censored tail.",
            transform=axes[1].transAxes,
            fontsize=8,
        )
        axes[1].legend(fontsize=7)
        fig.suptitle(
            title
            + (
                "\nCurrent = queue-only calibrated diagnostic; first-text distribution"
                if ttft_only
                else "\nExact forecast accuracy and deadline-aware completion distribution"
            )
        )
        fig.tight_layout()
        name = "production-mix-accuracy.png"
        fig.savefig(output / name, dpi=160)
        plt.close(fig)
        figures.append(name)
        if ttft_only:
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            snapshots = [
                dict(elapsed=r["started_elapsed_seconds"], **r.get("pre_admission_components", {}))
                for r in rows
            ]
            for field, label in [
                ("calibration_baseline_seconds", "Calibration baseline B"),
                ("queue_excess_seconds", "Native queue input N"),
                ("observed_delay_floor", "Empirical floor F (raw comparator only)"),
                ("predicted_excess_seconds", "Raw extra delay"),
            ]:
                points = [
                    r for r in snapshots if isinstance(r.get(field), (int, float)) and r[field] >= 0
                ]
                axes[0].scatter(values(points, "elapsed"), values(points, field), s=3, label=label)
            axes[0].set(ylabel="Input decomposition (seconds)", ylim=(0, None))
            axes[0].legend(fontsize=8)
            axes[1].scatter(
                values(snapshots, "elapsed"),
                values(snapshots, "calibration_scale"),
                s=3,
                label="Gain g; no F in fit or forecast",
            )
            axes[1].set(ylabel="Gain", ylim=(0, None))
            axes[1].legend(fontsize=8)
            axes[2].scatter(
                values(snapshots, "elapsed"),
                values(snapshots, "calibration_epoch"),
                s=3,
                label="Epoch",
            )
            axes[2].scatter(
                values(snapshots, "elapsed"),
                values(snapshots, "calibration_bank_index"),
                s=3,
                label="Bank index",
            )
            axes[2].set(xlabel="Admission time (s), linear", ylabel="Epoch / bank")
            axes[2].legend(fontsize=8)
            for ax in axes:
                ax.grid(alpha=0.15)
            fig.suptitle(
                title + "\nExact pre-admission decomposition: calibrated total = B + g × N"
            )
            fig.tight_layout()
            name = "production-mix-components.png"
            fig.savefig(output / name, dpi=160)
            plt.close(fig)
            figures.append(name)
        inputs += [
            "summary.json",
            "run-config.json",
            "paired-production.jsonl",
            "metrics.jsonl",
            "stages.json",
        ]
    manifest = dict(
        plot_schema_version=4 if ttft_only else 3,
        measurement_target=profile.get("measurement_target", "completion"),
        time_axis_scale=time_scale,
        plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs={n: hashlib.sha256((path / n).read_bytes()).hexdigest() for n in inputs},
        figures=figures,
    )
    (output / "production-plot-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use linear time axes for the request timeline and exact-pair comparison.",
    )
    args = parser.parse_args()
    print(json.dumps(render(args.input, args.output, args.linear), indent=2))
