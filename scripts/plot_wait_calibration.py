#!/usr/bin/env python3
"""Reproduce online TTFT calibration validation figures from retained harness data.

Run analyze_wait_calibration.py for each suite first, then:
  python plot_wait_calibration.py --input results/llama --title 'vLLM Llama 3.1 8B'
No calibration is fitted here: applied forecasts use the recorded live history at selection.
Unrestricted targets remain in the analysis data and are omitted from these figures.
"""
from __future__ import annotations
import argparse
import csv
import json
import hashlib
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'actual': '#172f44', 'raw': '#dc762b', 'calibrated': '#087e8b'}
SUITES = ['train', 'holdout', 'cold', 'hard']


def values(rows, key):
    return np.array([float(r.get(key) or 'nan') for r in rows])


def series(ax, metrics, key, label, color=None, **kwargs):
    y = values(metrics, key)
    y[y < 0] = np.nan
    ax.plot(values(metrics, 'sample_completed_elapsed_seconds'), y, label=label, color=color, **kwargs)


def timeseries(path, title):
    requests = json.loads((path / 'paired-calibration.json').read_text())
    metrics = list(csv.DictReader((path / 'metrics.tsv').open(), delimiter='\t'))
    summary = json.loads((path / 'summary.json').read_text())
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True, layout='constrained', gridspec_kw={'height_ratios': [2.3, 1, 1.3, 1.3]})
    ax = axes[0]
    series(ax, metrics, 'estimated_wait', 'Raw TTFT, sampled gauge', COLORS['raw'], lw=1.5, alpha=.8)
    series(ax, metrics, 'calibrated_ttft', 'Current estimate, sampled gauge', COLORS['calibrated'], lw=1.5, alpha=.9)
    x = [r['started_elapsed_seconds'] for r in requests]
    ax.scatter(x, [r['actual_ttft'] for r in requests], s=8, c=COLORS['actual'], alpha=.35, label='Actual first text, at request start', rasterized=True)
    ax.set_yscale('symlog', linthresh=1)
    ax.set_ylabel('TTFT (s; log above 1s)')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', ncols=2, fontsize=8)
    ax.set_title(f'{title} — {summary["suite"]}\nCurves are sampled estimates; dots are individual observed TTFTs. Exact same-request accuracy is scored separately.', loc='left', fontsize=12)
    ax = axes[1]
    if 'calibration_scale' in metrics[0]:
        series(ax, metrics, 'calibration_scale', 'Applied g, ≤0.1 units/s in each state', COLORS['calibrated'], lw=1.3, alpha=.85)
    ready = [r for r in requests if r['calibration_state'] == 'capacity_ready']
    unready = [r for r in requests if r['calibration_state'] == 'capacity_unready']
    for rows, label, color in [(ready, 'Applied g: native capacity ready', '#087e8b'), (unready, 'Applied g: native capacity unready', '#78828c')]:
        ax.scatter([r['started_elapsed_seconds'] for r in rows], [r['scale'] for r in rows], s=6, color=color, alpha=.65, label=label, rasterized=True)
    ax.axhline(1, ls='--', color='#999999', lw=.8)
    ax.set_yscale('symlog', linthresh=1)
    ax.set_ylabel('g (log above 1)')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', ncols=2, fontsize=8)
    ax = axes[2]
    series(ax, metrics, 'waiting', 'Waiting', '#bf5639', lw=1)
    series(ax, metrics, 'running', 'Running', '#315c9c', lw=1)
    series(ax, metrics, 'pending_first_text', 'Pending first text', '#91713c', lw=1, alpha=.7)
    ax.set_ylabel('Requests')
    ax.set_ylim(bottom=0)
    right = ax.twinx()
    series(right, metrics, 'kv_usage', 'KV usage', '#7057a3', lw=.9, alpha=.7)
    right.set_ylabel('KV fraction', color='#7057a3')
    right.set_ylim(0, 1)
    ax.legend(loc='upper left', ncols=3, fontsize=8)
    ax = axes[3]
    series(ax, metrics, 'baseline_seconds', 'Baseline B', '#53616e', lw=1)
    series(ax, metrics, 'observed_delay_floor', 'Observed extra-delay floor F', '#bf8b34', lw=1)
    ax.set_ylabel('Delay (seconds)')
    ax.set_ylim(bottom=0)
    right = ax.twinx()
    series(right, metrics, 'effective_capacity', 'Capacity C', '#315c9c', lw=1, alpha=.7)
    right.set_ylabel('C (tokens/second)', color='#315c9c')
    right.set_ylim(bottom=0)
    ax.legend(loc='upper left', ncols=2, fontsize=8)
    ax.set_xlabel('Elapsed time since suite start (seconds)')
    for axis in axes:
        axis.grid(alpha=.16)
        for phase in summary['phases']:
            axis.axvline(phase['start_elapsed_seconds'], color='#c4c9cf', lw=.5)
    for phase in summary['phases']:
        axes[0].text(phase['start_elapsed_seconds'], .97, phase['name'], rotation=90,
                     transform=axes[0].get_xaxis_transform(), va='top', fontsize=6.5, color='#586675', alpha=.85)
    output = path.parent / 'graphs'
    output.mkdir(exist_ok=True)
    dest = output / (summary['suite'] + '-calibration-timeseries.png')
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return str(dest)


def accuracy(model_path, title):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    result = []
    for ax, suite in zip(axes, ['train', 'holdout', 'cold']):
        path = model_path / suite
        rows = [r for r in json.loads((path / 'paired-calibration.json').read_text()) if not r['case'].startswith('baseline-')]
        actual = [r['actual_ttft'] for r in rows]
        maximum = max(actual + [r[key + '_ttft'] for r in rows for key in ['raw', 'calibrated']]) * 1.03
        for key in ['raw', 'calibrated']:
            ax.scatter(actual, [r[key + '_ttft'] for r in rows], s=7, alpha=.25, color=COLORS[key], label={'raw':'Raw','calibrated':'Current (time-limited)'}[key], rasterized=True)
        ax.plot([0, maximum], [0, maximum], color='#64707b', lw=1, ls='--', label='Perfect agreement')
        ax.set(xlim=(0, maximum), ylim=(0, maximum), xlabel='Actual first-text TTFT (seconds)', ylabel='Predicted TTFT (seconds)', title=suite)
        ax.set_xscale('symlog', linthresh=1)
        ax.set_yscale('symlog', linthresh=1)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.16)
        report = json.loads((path / 'calibration-analysis.json').read_text())
        raw = report['all']['raw']['mae']
        calibrated = report['all']['calibrated']['mae']
        ax.text(.04, .96, f'MAE: raw {raw:.3f}s → current {calibrated:.3f}s\nn={len(rows):,}', transform=ax.transAxes, va='top', fontsize=9)
    axes[0].legend(loc='lower right', fontsize=8)
    fig.suptitle(title + ' — exact same-request accuracy (baseline probes excluded)\nAbove diagonal = overprediction; below = underprediction. Axes linear 0–1s, logarithmic above 1s; all outliers retained.', fontsize=12)
    output = model_path / 'graphs'
    output.mkdir(exist_ok=True)
    dest = output / 'calibration-accuracy.png'
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return str(dest)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--title', required=True)
    args = p.parse_args()
    outputs = [timeseries(args.input / suite, args.title) for suite in SUITES]
    outputs.append(accuracy(args.input, args.title))
    manifest = dict(figures=outputs, plotter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), inputs={str(path.relative_to(args.input)): hashlib.sha256(path.read_bytes()).hexdigest() for suite in SUITES for name in ['paired-calibration.json', 'metrics.tsv', 'summary.json', 'calibration-analysis.json'] for path in [args.input / suite / name]})
    (args.input / 'graphs/calibration-plot-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))
