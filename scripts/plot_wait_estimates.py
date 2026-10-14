#!/usr/bin/env python3
"""Plot same-request estimated/actual first-text TTFT from a run directory or CSV.

See README.wait-estimate-plots.md for the input schema and future-test workflow.
Importing this module does not read data or generate output.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import json
import math
import os
import re
import statistics

COLORS = {'actual': '#2563eb', 'estimate': '#d97706', 'unknown': '#6b7280'}

def stats(rows):
    errors = [r['error_seconds'] for r in rows]
    complete = [r for r in rows if r['forecast_ready'] and r['baseline_available']]
    return dict(paired_requests=len(rows),
                estimated_ttft_mean_seconds=statistics.mean(r['estimated_ttft_seconds'] for r in rows) if rows else None,
                actual_ttft_mean_seconds=statistics.mean(r['actual_ttft_seconds'] for r in rows) if rows else None,
                bias_seconds=statistics.mean(errors) if errors else None,
                mae_seconds=statistics.mean(abs(x) for x in errors) if errors else None,
                complete_ready_requests=len(complete),
                complete_ready_mae_seconds=statistics.mean(abs(r['error_seconds']) for r in complete) if complete else None)

def timing_panel(ax, rows, key='request_start_seconds'):
    successful = [r for r in rows if r['status'] == 200]
    x = [r[key] for r in successful]
    actual = [r['actual_ttft_seconds'] for r in successful]
    predicted = [r['estimated_ttft_seconds'] for r in successful]
    ax.vlines(x, [min(a, p) for a, p in zip(actual, predicted)],
              [max(a, p) for a, p in zip(actual, predicted)],
              color='#64748b', alpha=.12, linewidth=.55, zorder=1)
    ax.scatter(x, actual, s=11, color=COLORS['actual'], alpha=.65, label='Actual first text', zorder=2)
    for state, marker, label in [('ready', '^', 'Estimate: ready'),
                                 ('unready', '^', 'Estimate: unready'),
                                 ('missing', 'x', 'Estimate: baseline missing')]:
        group = [r for r in successful if (
            'missing' if not r['baseline_available'] else 'ready' if r['forecast_ready'] else 'unready') == state]
        if not group:
            continue
        xx = [r[key] for r in group]
        yy = [r['estimated_ttft_seconds'] for r in group]
        if state == 'missing':
            ax.scatter(xx, yy, s=14, marker=marker, color=COLORS['unknown'], linewidths=.8, alpha=.55, label=label, zorder=3)
        else:
            ax.scatter(xx, yy, s=14, marker=marker,
                       facecolors=COLORS['estimate'] if state == 'ready' else 'none',
                       edgecolors=COLORS['estimate'], linewidths=.65, alpha=.7, label=label, zorder=3)
    rejected = [r for r in rows if r['status'] != 200]
    if rejected:
        ax.plot([r[key] for r in rejected], [.97]*len(rejected), '|',
                transform=ax.get_xaxis_transform(), color='#b91c1c', markersize=5,
                label='No actual TTFT (rejected/error)', alpha=.4)
    if successful:
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylim(0, 1)
        ax.text(.5, .5, 'No actual first-text observations', ha='center', transform=ax.transAxes)
    ax.set_ylabel('Time to first text (seconds)')
    ax.grid(alpha=.2)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.008), fontsize=8,
              ncol=2, framealpha=.92, borderaxespad=0)

def plot(rows, title, focus_case, output_path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.cm import ScalarMappable
    successful = [r for r in rows if r['status'] == 200]
    focus = [r for r in rows if r['case'] == focus_case]
    focus_ok = [r for r in focus if r['status'] == 200]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11.5), layout='constrained')
    top, zoom, error, parity = axes.flat
    timing_panel(top, rows)
    top.set(title='All scenarios: each pair at the same request start', xlabel='Request start (seconds since first test request)')
    top.set_title(top.get_title(), pad=47)
    phases = sorted({r['case'] for r in rows}, key=lambda c: min(r['request_start_seconds'] for r in rows if r['case'] == c))
    phase_map = []
    for i, case in enumerate(phases):
        group = [r for r in rows if r['case'] == case]
        start = min(r['request_start_seconds'] for r in group)
        end = max(r['request_start_seconds'] for r in group)
        phase_map.append(dict(index=i+1, case=case, start_seconds=start, last_start_seconds=end))
        for ax in [top, error]:
            ax.axvspan(start, end, color='#e2e8f0', alpha=.5 if i % 2 == 0 else .15, zorder=0)
        top.text((start+end)/2, .99, str(i+1), ha='center', va='top',
                 transform=top.get_xaxis_transform(), fontsize=8, bbox=dict(facecolor='white', edgecolor='none', alpha=.7))
    timing_panel(zoom, focus, key='phase_request_start_seconds')
    zoom.set(title=focus_case+' · paired-request zoom', xlabel='Request start (seconds since this phase began)')
    zoom.set_title(zoom.get_title(), pad=47)
    for missing in [False, True]:
        group = [r for r in successful if (not r['baseline_available']) == missing]
        if not group:
            continue
        color = [COLORS['unknown'] if missing else '#b91c1c' if r['error_seconds'] >= 0 else '#2563eb' for r in group]
        error.scatter([r['request_start_seconds'] for r in group], [r['error_seconds'] for r in group],
                      s=13, c=color, marker='x' if missing else 'o', alpha=.55,
                      linewidths=.65 if missing else 0)
    error.axhline(0, color='#111827', linewidth=1)
    error.set(title='Prediction error: estimate − actual',
              xlabel='Request start (seconds since first test request)', ylabel='Signed error (seconds)')
    error.grid(alpha=.2)
    summary = stats(successful)
    complete_mae = summary['complete_ready_mae_seconds']
    error.set_title(error.get_title(), pad=47)
    error_label = (f"All-pair bias: {summary['bias_seconds']:+.3f} s; MAE: {summary['mae_seconds']:.3f} s\n"
                   + (f"Ready + baseline available: n={summary['complete_ready_requests']}, MAE={complete_mae:.3f} s"
                      if complete_mae is not None else 'No ready predictions with a known baseline')) if successful else 'No accepted first-text observations'
    error.text(.01, 1.01, error_label, transform=error.transAxes, va='bottom', fontsize=9,
               bbox=dict(facecolor='white', edgecolor='#cbd5e1', alpha=.92))
    norm = Normalize(vmin=0, vmax=max((r['phase_request_start_seconds'] for r in focus), default=0) or 1)
    cmap = plt.get_cmap('viridis')
    for state, marker in [('ready', 'o'), ('unready', 'o'), ('missing', 'x')]:
        group = [r for r in focus_ok if (
            'missing' if not r['baseline_available'] else 'ready' if r['forecast_ready'] else 'unready') == state]
        colors = [cmap(norm(r['phase_request_start_seconds'])) for r in group]
        if not group:
            continue
        kwargs = dict(c=colors) if state == 'missing' else dict(facecolors=colors if state == 'ready' else 'none', edgecolors=colors)
        parity.scatter([r['actual_ttft_seconds'] for r in group], [r['estimated_ttft_seconds'] for r in group],
                       marker=marker, s=20, alpha=.7, linewidths=.8, **kwargs)
    limit = max(1, max((max(r['actual_ttft_seconds'], r['estimated_ttft_seconds']) for r in focus_ok), default=0)) * 1.06
    parity.plot([0, limit], [0, limit], color='#475569', linewidth=1, label='Estimate = actual')
    parity.set(xlim=(0, limit), ylim=(0, limit), aspect='equal',
               title='Direct comparison · color adds request start time',
               xlabel='Actual first-text time (seconds)', ylabel='Estimated first-text time (seconds)')
    parity.grid(alpha=.2)
    parity.set_title(parity.get_title(), pad=47)
    parity.legend(handles=[Line2D([], [], color='#475569', label='Perfect match'),
                           Line2D([], [], marker='o', linestyle='none', color='#64748b', label='Ready, known baseline'),
                           Line2D([], [], marker='o', linestyle='none', color='#64748b', markerfacecolor='none', label='Unready'),
                           Line2D([], [], marker='x', linestyle='none', color='#64748b', label='Baseline missing')],
                  fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.008), ncol=2, borderaxespad=0)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=parity, fraction=.045, pad=.02,
                 label='Request start in focused phase (seconds)')
    fig.suptitle(title+'\nSame request ID, estimate before admission, actual first text on SMG clock', fontsize=16)
    if not focus_ok:
        parity.text(.5, .5, 'No actual first-text observations', ha='center', transform=parity.transAxes)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return dict(chart=output_path.name, phase_map=phase_map,
                successful_pairs=len(successful), unobserved_requests=sum(r['status'] != 200 for r in rows),
                summary=summary, focus_case=focus_case, focus_summary=stats(focus_ok))


def boolean(value):
    if value in (True, False):
        return bool(value)
    if str(value).lower() in ('true', '1'):
        return True
    if str(value).lower() in ('false', '0'):
        return False
    raise ValueError(f'Invalid boolean: {value!r}')


def read_run(path):
    """Join exact forecasts/first-text durations by request ID, never by time."""
    meta = json.loads((path / 'summary.json').read_text())
    requests = list(csv.DictReader((path / 'requests.tsv').open(), delimiter='\t'))
    pairs = json.loads((path / 'paired-requests.json').read_text())
    by_id = {r['request_id']: r for r in pairs}
    if len(by_id) != len(pairs):
        raise ValueError('Duplicate paired request ID')
    forecasts = {}
    for line in (path / 'router-events.jsonl').read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        event = event.get('fields', {}) | event
        if event.get('event') == 'forecast':
            rid = event['request_id']
            if rid in forecasts:
                raise ValueError(f'Duplicate forecast: {rid}')
            forecasts[rid] = event
    rows = []
    for request in requests:
        if request['case'].startswith('baseline-'):
            continue
        rid = request['opc_request_id']
        if rid not in forecasts:
            raise ValueError(f'Missing exact pre-admission forecast: {rid}')
        forecast = forecasts[rid]
        pair = by_id.get(rid)
        status = int(request['status'])
        if (pair is not None) != (status == 200):
            raise ValueError(f'Missing or unexpected actual first-text duration: {rid}')
        rows.append(dict(
            model=path.parent.name, suite=meta.get('suite', path.name),
            engine=meta.get('engine', ''), case=request['case'], request_id=rid, status=status,
            request_start_seconds=float(request['started_elapsed_seconds']),
            harness_started_elapsed_seconds=float(request['started_elapsed_seconds']),
            estimated_ttft_seconds=forecast['predicted_ttft_seconds'],
            actual_ttft_seconds=pair['actual_ttft_seconds'] if pair else None,
            forecast_ready=forecast['ready'], baseline_available=forecast['baseline_available'],
            would_reject=forecast['would_reject'],
            client_ttft_seconds=float(request['ttft_seconds']) if pair else None,
            client_completion_seconds=float(request['e2e_seconds']),
            prompt_words=int(request['prompt_words']), output_token_limit=int(request['output_tokens']),
        ))
        if pair and pair['predicted_ttft_seconds'] != forecast['predicted_ttft_seconds']:
            raise ValueError(f'Paired estimate differs from forecast: {rid}')
    return rows, meta.get('image')


def read_csv(path):
    rows = list(csv.DictReader(path.open()))
    required = {'model', 'suite', 'case', 'request_id', 'status', 'request_start_seconds',
                'estimated_ttft_seconds', 'actual_ttft_seconds', 'forecast_ready', 'baseline_available'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('CSV must contain: ' + ', '.join(sorted(required)))
    return [row for row in rows if not row['case'].startswith('baseline-')]


def prepare(rows):
    """Validate measurements and recompute time origins/error for each run."""
    if not rows:
        raise ValueError('No non-baseline requests to plot')
    ids = set()
    for row in rows:
        rid = row['request_id']
        if not rid:
            raise ValueError('Request ID must not be empty')
        if rid in ids:
            raise ValueError(f'Duplicate request ID: {rid}')
        ids.add(rid)
        row['status'] = int(row['status'])
        for key in ['request_start_seconds', 'estimated_ttft_seconds']:
            row[key] = float(row[key])
            if not math.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f'Invalid {key}: {rid}')
        for key in ['forecast_ready', 'baseline_available']:
            row[key] = boolean(row[key])
        value = row['actual_ttft_seconds']
        if row['status'] == 200:
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid actual first-text duration: {rid}')
            row['actual_ttft_seconds'] = value
            row['error_seconds'] = row['estimated_ttft_seconds'] - value
            row['estimate_actual_ratio'] = row['estimated_ttft_seconds'] / value
        else:
            if value not in (None, ''):
                raise ValueError(f'Rejected/error request cannot have actual TTFT: {rid}')
            row['actual_ttft_seconds'] = None
            row['error_seconds'] = None
            row['estimate_actual_ratio'] = None
    origin = min(row['request_start_seconds'] for row in rows)
    phase_origins = {case: min(row['request_start_seconds'] for row in rows if row['case'] == case)
                     for case in {row['case'] for row in rows}}
    for row in rows:
        start = row['request_start_seconds']
        row['phase_request_start_seconds'] = start - phase_origins[row['case']]
        row['request_start_seconds'] = start - origin
    return sorted(rows, key=lambda row: (row['request_start_seconds'], row['request_id']))


def choose_focus(rows):
    counts = Counter(row['case'] for row in rows)
    mixed = [case for case in counts if 'mixed' in case.lower() or 'mixture' in case.lower()]
    return max(mixed or list(counts), key=lambda case: counts[case])


def slug(value):
    return re.sub(r'[^a-zA-Z0-9._-]+', '-', str(value)).strip('.-') or 'run'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Analyzed run directory or paired CSV')
    parser.add_argument('--output-dir', type=Path, required=True, help='Directory for PNG, CSV and summary JSON')
    parser.add_argument('--title', help='Display title override, for a single model/suite')
    parser.add_argument('--focus-case', help='Phase for the zoom/parity panels; default: a mixed phase or largest phase')
    parser.add_argument('--source-revision', help='Optional tested source revision for the summary')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing generated outputs')
    args = parser.parse_args()
    try:
        rows, image = read_run(args.input) if args.input.is_dir() else (read_csv(args.input), None)
        groups = defaultdict(list)
        for row in rows:
            groups[(row['model'], row['suite'])].append(row)
        if args.title and len(groups) != 1:
            raise ValueError('--title requires one model/suite group')
        prepared = []
        for (model, suite), group in groups.items():
            group = prepare(group)
            focus = args.focus_case or choose_focus(group)
            if not any(row['case'] == focus for row in group):
                raise ValueError(f'Focus case {focus!r} absent from {model}/{suite}')
            prefix = f'ttft-{slug(model)}-{slug(suite)}'
            if any(p[0] == prefix for p in prepared):
                raise ValueError(f'Output filename collision: {prefix}')
            for suffix in ['.png', '-pairs.csv', '-summary.json']:
                path = args.output_dir / (prefix + suffix)
                if path.exists() and not args.overwrite:
                    raise ValueError(f'Output exists: {path}; use another output directory or --overwrite')
            engine = group[0].get('engine', '')
            title = args.title or ' · '.join(str(v) for v in [engine, model, suite] if v)
            prepared.append((prefix, model, suite, title, focus, group))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('MPLCONFIGDIR', str(args.output_dir.resolve() / '.matplotlib'))
        os.environ.setdefault('XDG_CACHE_HOME', str(args.output_dir.resolve() / '.cache'))
        os.environ.setdefault('MPLBACKEND', 'Agg')
        for prefix, model, suite, title, focus, group in prepared:
            result = plot(group, title, focus, args.output_dir / (prefix + '.png'))
            result.update(model=model, suite=suite, source_revision=args.source_revision, image=image,
                          input=str(args.input.resolve()), status_counts=dict(Counter(r['status'] for r in group)),
                          definitions={'error_seconds': 'estimate minus actual SMG first-text duration',
                                       'estimate_actual_ratio': 'estimate divided by actual; 1 means equality'},
                          caveats=['Actual and estimate share the original request-start coordinate.',
                                   'Unready/baseline-missing totals are initialization diagnostics.',
                                   'Rejected/error requests have no actual TTFT; error statistics are accepted-only.',
                                   'Client completion time is separate from first-text time.'])
            (args.output_dir / (prefix + '-summary.json')).write_text(json.dumps(result, indent=2))
            fields = list(dict.fromkeys(key for row in group for key in row))
            with (args.output_dir / (prefix + '-pairs.csv')).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(group)
            print(json.dumps({'model': model, 'suite': suite, 'chart': str(args.output_dir / (prefix + '.png')),
                              'paired_requests': result['successful_pairs'], 'unobserved_requests': result['unobserved_requests']}))
    except (ValueError, TypeError, KeyError, OSError, ImportError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
