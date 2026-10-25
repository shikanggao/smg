#!/usr/bin/env python3
"""Render retained pressure with explicit pending-at-cutoff cohort markers."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"actual": "#172f44", "raw": "#dc762b", "calibrated": "#087e8b"}


def values(rows, key):
    return np.array([float(r.get(key) or "nan") for r in rows])


def line(ax, rows, key, label, color=None, **kwargs):
    y = values(rows, key)
    y[y < 0] = np.nan
    xkey = (
        "sample_completed_elapsed_seconds"
        if rows and "sample_completed_elapsed_seconds" in rows[0]
        else "elapsed_seconds"
    )
    ax.plot(values(rows, xkey), y, label=label, color=color, **kwargs)


def plot(path, title):
    rows = json.loads((path / "paired-deadline-stratified.json").read_text())
    probes = [r for r in rows if not r["case"].startswith("background")]
    with (path / "metrics.tsv").open() as file:
        metrics = list(csv.DictReader(file, delimiter="\t"))
    stages = json.loads((path / "stages.json").read_text())
    report = json.loads((path / "deadline-analysis.json").read_text())
    strata = json.loads((path / "deadline-strata.json").read_text())
    unload = strata["unload_elapsed_seconds"]
    cutoff = strata["cutoff_elapsed_seconds"]
    reason = strata["cutoff_reason"].replace("_", " ")
    output = path / "graphs"
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(17, 15),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [2, 1.8, 0.8, 1.2, 1.1]},
    )
    ax = axes[0]
    line(ax, metrics, "estimated_wait", "Raw TTFT, sampled", COLORS["raw"], lw=1.1, alpha=0.7)
    line(
        ax,
        metrics,
        "calibrated_ttft",
        "Current TTFT, sampled",
        COLORS["calibrated"],
        lw=1.1,
        alpha=0.8,
    )
    for key, label, color in [
        ("raw_ttft", "Raw, exact probe snapshot", COLORS["raw"]),
        ("calibrated_ttft", "Current, exact probe snapshot", COLORS["calibrated"]),
        ("actual_ttft", "Actual SMG first text, probes", COLORS["actual"]),
    ]:
        valid = [r for r in probes if r[key] is not None]
        for affected, marker in [(False, "o"), (True, "x")]:
            group = [r for r in valid if r["first_text_pending_at_cutoff"] == affected]
            ax.scatter(
                [r["started_elapsed_seconds"] for r in group],
                [r[key] for r in group],
                s=19,
                alpha=0.65,
                color=color,
                marker=marker,
                label=label + ("; pending at cutoff ×" if affected else "; not pending ○"),
                rasterized=True,
            )
    ax.axhline(300, color="#a63039", ls="--", lw=1, label="300 seconds")
    ax.set_ylabel("First-text TTFT (s)")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8, ncols=2)
    ax.set_title(
        title
        + (
            "\nCurves: metric samples. ○: not pending at cutoff. ×: pending "
            "at cutoff; later latency can change. Points use request "
            "start."
        ),
        loc="left",
    )
    ax = axes[1]
    for key, label, marker, color in [
        ("client_ttft", "Client first text", "o", "#315c9c"),
        ("e2e_seconds", "Full completion: DONE + finish_reason", "D", "#71569e"),
    ]:
        valid = [
            r
            for r in probes
            if r[key] is not None and (r["complete"] if key == "e2e_seconds" else True)
        ]
        field = (
            "first_text_pending_at_cutoff"
            if key == "client_ttft"
            else "completion_pending_at_cutoff"
        )
        for affected in [False, True]:
            group = [r for r in valid if r[field] == affected]
            ax.scatter(
                [r["started_elapsed_seconds"] for r in group],
                [r[key] for r in group],
                s=19,
                alpha=0.65,
                marker="x" if affected else marker,
                color=color,
                label=label + ("; pending at cutoff ×" if affected else ""),
                rasterized=True,
            )
    censored = [r for r in probes if not r["complete"] and r["completion_late"]]
    if censored:
        ax.scatter(
            [r["started_elapsed_seconds"] for r in censored],
            [300] * len(censored),
            marker="^",
            s=30,
            color="#a63039",
            label="Known deadline miss; exact completion censored",
        )
    ax.axhline(300, color="#a63039", ls="--", lw=1, label="Production completion deadline")
    ax.set_ylabel("Client latency (s)")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8, ncols=2)
    ax = axes[2]
    line(ax, metrics, "calibration_scale", "Applied g", COLORS["calibrated"], lw=1.2)
    ax.axhline(1, color="#888888", ls="--", lw=0.7)
    ax.set_ylabel("Applied scale g")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[3]
    for key, label, color in [
        ("waiting", "Engine waiting", "#bf5639"),
        ("running", "Engine running", "#315c9c"),
        ("pending_first_text", "SMG pending first text", "#91713c"),
        ("client_background_active", "Client background in flight", "#687d71"),
    ]:
        line(ax, metrics, key, label, color, lw=1)
    ax.set_ylabel("Requests")
    ax.set_ylim(bottom=0)
    right = ax.twinx()
    line(right, metrics, "kv_usage", "KV usage", "#7057a3", lw=0.9, alpha=0.7)
    right.set_ylim(0, 1)
    right.set_ylabel("KV fraction", color="#7057a3")
    ax.legend(loc="upper left", fontsize=8, ncols=2)
    ax = axes[4]
    line(ax, metrics, "baseline_seconds", "Baseline B", "#53616e", lw=1)
    line(ax, metrics, "observed_delay_floor", "Delay floor F", "#bf8b34", lw=1)
    ax.set_ylabel("Delay (s)")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    right = ax.twinx()
    line(right, metrics, "effective_capacity", "Capacity C", "#315c9c", lw=1)
    right.set_ylabel("C (tokens/s)")
    right.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8, ncols=2)
    ax.set_xlabel("Elapsed time since test start (seconds)")
    for axis in axes:
        axis.grid(alpha=0.16)
        axis.axvline(
            cutoff,
            color="#172f44",
            lw=1.5,
            ls="--",
            label=("Pressure cutoff: " + reason) if axis is axes[0] else "_nolegend_",
        )
        if cutoff < unload:
            axis.axvline(
                unload,
                color="#64707b",
                lw=1.2,
                ls=":",
                label="Planned background unload" if axis is axes[0] else "_nolegend_",
            )
        for stage in stages:
            start = stage.get("elapsed_seconds", stage.get("start_elapsed_seconds"))
            if start is not None:
                axis.axvline(start, color="#c4c9cf", lw=0.6)
    for stage in stages:
        start = stage.get("elapsed_seconds", stage.get("start_elapsed_seconds"))
        if start is not None:
            axes[0].text(
                start,
                0.97,
                stage["name"],
                rotation=90,
                transform=axes[0].get_xaxis_transform(),
                va="top",
                fontsize=6.5,
                color="#586675",
            )
    axes[0].legend(loc="upper left", fontsize=8, ncols=2)
    timeline = output / "deadline-pressure-stratified-timeline.png"
    fig.savefig(timeline, dpi=135)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
    exact = [r for r in probes if r["actual_ttft"] is not None]
    for key, label in [("raw", "Raw TTFT"), ("calibrated", "Current TTFT")]:
        for affected, marker in [(False, "o"), (True, "x")]:
            group = [r for r in exact if r["first_text_pending_at_cutoff"] == affected]
            ax.scatter(
                [r["actual_ttft"] for r in group],
                [r[key + "_ttft"] for r in group],
                color=COLORS[key],
                label=label + ("; pending at cutoff ×" if affected else "; not pending ○"),
                marker=marker,
                s=25,
                alpha=0.6,
            )
    maximum = (
        max([300] + [r[k] for r in exact for k in ["actual_ttft", "raw_ttft", "calibrated_ttft"]])
        * 1.05
    )
    ax.plot([0, maximum], [0, maximum], color="#64707b", ls="--", lw=1, label="Perfect agreement")
    ax.axvline(300, color="#a63039", ls=":", lw=1)
    ax.axhline(300, color="#a63039", ls=":", lw=1)
    ax.set(
        xlim=(0, maximum),
        ylim=(0, maximum),
        xlabel="Actual SMG first-text TTFT (s)",
        ylabel="Predicted TTFT (s)",
        title=title + "\nExact same-request probes; all outliers retained",
    )
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.16)
    ax.legend(loc="lower right")
    ax.text(
        0.04,
        0.96,
        (
            f"Measured SMG TTFT >300s: {report['measured_smg_sentinel_over_300']}\nCompleted "
            f"paired proof before cutoff: {strata['completed_paired_proof_before_cutoff']}\nRejected "
            f"counterfactual outcomes excluded"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    accuracy = output / "deadline-pressure-stratified-accuracy.png"
    fig.savefig(accuracy, dpi=150)
    plt.close(fig)
    manifest = dict(
        schema_version=2,
        cutoff_elapsed_seconds=cutoff,
        cutoff_reason=strata["cutoff_reason"],
        plotted_probe_count=len(probes),
        plotted_actual_probe_count=len(exact),
        maximum_probe_value_seconds=maximum / 1.05,
        plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs={
            name: hashlib.sha256((path / name).read_bytes()).hexdigest()
            for name in [
                "paired-deadline-stratified.json",
                "metrics.tsv",
                "stages.json",
                "deadline-analysis.json",
                "deadline-strata.json",
            ]
        },
        figures=[timeline.name, accuracy.name],
    )
    (output / "deadline-stratified-plot-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--title", required=True)
    args = p.parse_args()
    plot(args.input, args.title)
