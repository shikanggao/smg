#!/usr/bin/env python3
"""Plot recorded engine inputs, SMG formula components and same-request TTFT.

This replays recorded admission traces; it never queries or changes a deployment.
See README.wait-metric-plots.md for the required files and clock boundaries.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import os
import re
import statistics


def number(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return math.nan
    return value if math.isfinite(value) and value >= 0 else math.nan


def delta(rows, key):
    """Nonnegative counter increments; gaps and resets stay missing, not zero."""
    values = [number(row.get(key)) for row in rows]
    return [math.nan] + [b-a if math.isfinite(a) and math.isfinite(b) and b >= a
                        else math.nan for a, b in zip(values, values[1:])]


def rate(rows, key):
    increments = delta(rows, key)
    times = [float(row['elapsed_seconds']) for row in rows]
    return [math.nan] + [d/(b-a) if b > a else math.nan
                        for d, a, b in zip(increments[1:], times, times[1:])]


def interval_mean(rows, sum_key, count_key):
    sums, counts = delta(rows, sum_key), delta(rows, count_key)
    return [s/c if c > 0 and math.isfinite(s) else math.nan
            for s, c in zip(sums, counts)]


def recalculate(forecast):
    """Replay b=0, alpha=1 with the uncalibrated-capacity observed-delay branch.

    Older traces have no explicit floor-available flag. For fresh, queue-proxy
    schema it is equivalent to waiting>0 and at least four observed samples.
    Stale snapshots/direct-token queues need a richer trace, not this inference.
    """
    if 'observed_delay_available' in forecast and 'capacity_learned_ready' in forecast:
        floor_only = not forecast['capacity_learned_ready'] and forecast['observed_delay_available']
    else:
        floor_only = (not forecast['native_ready'] and forecast['native_waiting'] > 0
                      and forecast['observed_samples'] >= 4)
    delay = forecast['observed_delay_floor']
    if not floor_only:
        delay = max(delay, forecast['token_work']/forecast['effective_capacity'])
    return forecast['baseline_seconds'] + delay, floor_only


def load_run(path):
    meta = json.loads((path/'summary.json').read_text())
    with (path/'metrics.tsv').open() as handle:
        metrics = list(csv.DictReader(handle, delimiter='\t'))
    if len(metrics) < 2:
        raise ValueError(f'{path}: at least two metric samples required')
    times = [float(row['elapsed_seconds']) for row in metrics]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f'{path}: metric time must increase')
    # The replay branch inference below is valid only for fresh proxy queues.
    if any(number(row.get('snapshot_age')) > 30 for row in metrics):
        raise ValueError(f'{path}: stale snapshots cannot be replayed from this trace schema')
    if any(number(row.get('queue_proxy')) != 1 for row in metrics):
        raise ValueError(f'{path}: replay requires recorded queue-proxy inputs')
    forecasts = {}
    for line in (path/'router-events.jsonl').read_text().splitlines():
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
    pairs = json.loads((path/'paired-requests.json').read_text())
    actuals = {row['request_id']: row for row in pairs}
    if len(actuals) != len(pairs):
        raise ValueError('Duplicate paired request ID')
    requests, errors = [], []
    with (path/'requests.tsv').open() as handle:
        for request in csv.DictReader(handle, delimiter='\t'):
            rid = request['opc_request_id']
            event = forecasts[rid]
            actual = actuals.get(rid)
            status = int(request['status'])
            if (actual is not None) != (status == 200):
                raise ValueError(f'Unexpected or missing actual TTFT: {rid}')
            calculated, floor_only = recalculate(event)
            error = abs(calculated-event['predicted_ttft_seconds'])
            errors.append(error)
            if error > 1e-8:
                raise ValueError(f'Formula replay mismatch {rid}: {error}')
            if actual and actual['predicted_ttft_seconds'] != event['predicted_ttft_seconds']:
                raise ValueError(f'Paired forecast mismatch: {rid}')
            duration = number(actual['actual_ttft_seconds']) if actual else math.nan
            if actual and (not math.isfinite(duration) or duration <= 0):
                raise ValueError(f'Invalid actual TTFT: {rid}')
            requests.append(event | dict(
                x=float(request['started_elapsed_seconds']), case=request['case'],
                status=status, actual_ttft_seconds=duration,
                recalculated_ttft_seconds=calculated, floor_only=floor_only))
    requests.sort(key=lambda row: (row['x'], row['request_id']))
    intervals = [b-a for a, b in zip(times, times[1:])]
    missing = [key for key in metrics[0] if key not in ('timestamp', 'phase')
               and not any(math.isfinite(number(row.get(key))) for row in metrics)]
    info = dict(model=path.parent.name, suite=meta.get('suite', path.name),
                engine=meta['engine'], image=meta['image'], input_path=str(path),
                requests=len(requests), accepted=len(pairs),
                max_recalculation_error_seconds=max(errors),
                metric_samples=len(metrics), missing_metrics=missing,
                scrape_interval_mean_seconds=statistics.mean(intervals),
                scrape_interval_max_seconds=max(intervals),
                max_snapshot_age_seconds=max(number(row['snapshot_age']) for row in metrics),
                phases=[dict(name=name, start_seconds=min(r['x'] for r in requests if r['case']==name),
                             last_start_seconds=max(r['x'] for r in requests if r['case']==name))
                        for name in dict.fromkeys(r['case'] for r in requests)])
    return metrics, requests, info


def draw(path, metrics, requests, info, view, focus, output, title, floor):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    row_count = {'formula': 7, 'learning': 8, 'diagnostics': 10}[view]
    fig, axes = plt.subplots(row_count, 1, figsize=(15, 2.25*row_count+1),
                             sharex=True, layout='constrained')
    selected = [row for row in requests if row['case']==focus] if focus else requests
    if focus:
        lo = min(row['x'] for row in selected)
        # Include the tail of the last request so metrics show the queue draining.
        hi = max(row['x'] + (row['actual_ttft_seconds'] if row['status']==200 else 0)
                 for row in selected)
        padding = max(1, (hi-lo)*.02)
        limits = (lo-padding, hi+padding)
    else:
        limits = (0, max(float(metrics[-1]['elapsed_seconds']), requests[-1]['x'])+1)
    xx = [float(row['elapsed_seconds']) for row in metrics]
    rx = [row['x'] for row in selected]

    def sampled(ax, key, label, color=None, values=None):
        ys = values if values is not None else [number(row.get(key)) for row in metrics]
        if any(math.isfinite(y) for y in ys):
            ax.step(xx, ys, where='post', label=label, color=color, linewidth=1.1, alpha=.85)
        else:
            ax.plot([], [], color='#94a3b8', label=label+' (not recorded/available)')

    def traced(ax, key, label, color=None):
        ax.scatter(rx, [number(row[key]) for row in selected], s=7, alpha=.45,
                   color=color, label=label)

    def panel(ax, name, unit):
        ax.set_title(name, loc='left', fontsize=11, pad=26)
        ax.set_ylabel(unit, fontsize=9)

    def flags(ax, entries):
        # Separate lanes keep constant/overlapping boolean signals visible.
        for index, (key, label, source) in enumerate(entries):
            base = index*1.5
            ax.axhline(base, color='#cbd5e1', linewidth=.5)
            if source == 'trace':
                ax.scatter(rx, [base+.8*float(row[key]) for row in selected],
                           s=7, alpha=.55, label=label)
            else:
                ax.step(xx, [base+.8*number(row.get(key)) for row in metrics],
                        where='post', linewidth=1.1, label=label)
        ax.set_yticks([i*1.5+.4 for i in range(len(entries))],
                     [label for _, label, _ in entries], fontsize=8)
        ax.set_ylim(-.15, (len(entries)-1)*1.5+1)

    top = axes[0]
    successful = [row for row in selected if row['status']==200]
    top.scatter([r['x'] for r in successful], [r['actual_ttft_seconds'] for r in successful],
                color='#2563eb', label='Actual SMG first-text TTFT', s=10, alpha=.65)
    for state, fill in [(True, '#d97706'), (False, 'none')]:
        group = [r for r in selected if r['ready']==state and r['baseline_available']]
        top.scatter([r['x'] for r in group], [r['recalculated_ttft_seconds'] for r in group],
                    facecolors=fill, edgecolors='#d97706', marker='^', s=11, linewidth=.6,
                    label='Recalculated estimate: '+('ready' if state else 'unready'), alpha=.6)
    missing_baseline = [r for r in selected if not r['baseline_available']]
    if missing_baseline:
        top.scatter([r['x'] for r in missing_baseline], [r['recalculated_ttft_seconds'] for r in missing_baseline],
                    color='#6b7280', marker='x', s=11, linewidth=.7,
                    label='Estimate: baseline missing (incomplete total)', alpha=.6)
    sampled(top, 'estimated_wait', 'Scraped SMG estimate', '#6b7280')
    rejected = [r for r in selected if r['status']!=200]
    if rejected:
        top.plot([r['x'] for r in rejected], [.96]*len(rejected), '|',
                 transform=top.get_xaxis_transform(), color='#b91c1c',
                 label='Rejected: no actual TTFT', alpha=.4)
    panel(top, 'Estimate and actual: both placed at that request’s start', 'Seconds')
    top.set_title('Estimate and actual: both placed at that request’s start',
                  loc='left', fontsize=11, pad=42)
    if view == 'formula':
        traced(axes[1], 'predicted_excess_seconds', 'Exact estimated extra delay', '#d97706')
        axes[1].scatter([r['x'] for r in successful if r['baseline_available']],
                         [max(0, r['actual_ttft_seconds']-r['baseline_seconds'])
                          for r in successful if r['baseline_available']],
                         label='Actual TTFT − B at admission', s=8, color='#2563eb', alpha=.5)
        sampled(axes[1], 'excess_seconds', 'Scraped extra delay', '#64748b')
        sampled(axes[1], 'threshold', 'Admission budget', '#b91c1c')
        panel(axes[1], 'Extra delay compared with the shared budget', 'Seconds')
        traced(axes[2], 'token_work', 'Exact W = Q + 0.05D', '#7c3aed')
        if all('queued_tokens' in row for row in selected):
            traced(axes[2], 'queued_tokens', 'Exact Q: queued work', '#2563eb')
            axes[2].scatter(rx, [.05*row['dispatched_tokens_since_poll'] for row in selected],
                            s=7, alpha=.45, color='#059669', label='Exact 0.05D: dispatch contribution')
            panel(axes[2], 'Work numerator at admission: recorded queue and dispatch contributions', 'Tokens')
        else:
            panel(axes[2], 'Work numerator at admission (Q and D not separately logged)', 'Tokens')
        traced(axes[3], 'effective_capacity', 'C at admission', '#059669')
        sampled(axes[3], 'effective_capacity', 'Scraped effective C', '#047857')
        sampled(axes[3], 'learned_capacity', 'Learned prefill capacity', '#a855f7')
        axes[3].axhline(floor, color='#b91c1c', linestyle='--', label=f'{floor:g} tokens/s floor')
        panel(axes[3], 'Capacity denominator (generation throughput is not this divisor)', 'Tokens/second')
        sampled(axes[4], 'baseline_seconds', 'B: low-load baseline', '#2563eb')
        sampled(axes[4], 'observed_delay_floor', 'F: adjusted observed delay floor', '#d97706')
        sampled(axes[4], 'native_queue_delay', 'Native recent queue delay', '#7c3aed')
        traced(axes[4], 'observed_delay_floor', 'F at admission', '#b45309')
        panel(axes[4], 'Delay components; missing baseline is a gap, not measured zero', 'Seconds')
        flags(axes[5], [('ready', 'Admission ready', 'trace'),
                        ('native_ready', 'Native capacity ready', 'trace'),
                        ('baseline_available', 'Baseline available', 'trace'),
                        ('data_usable', 'Usable snapshot', 'sample')])
        panel(axes[5], 'Readiness: each lane moves upward when true', 'Low=0; high=1')
        sampled(axes[6], 'snapshot_age', 'Engine snapshot age', '#2563eb')
        sampled(axes[6], '', 'Metric fetch duration', '#d97706', values=[
            float(r['sample_completed_elapsed_seconds'])-float(r['elapsed_seconds']) for r in metrics])
        panel(axes[6], 'Freshness and collection delay', 'Seconds')
    elif view == 'learning':
        sampled(axes[1], 'vllm_waiting', 'Native waiting (alias shared by both engines)', '#7c3aed')
        sampled(axes[1], 'waiting', 'SMG waiting snapshot', '#d97706')
        traced(axes[1], 'native_waiting', 'Waiting at admission', '#059669')
        if all('observed_backlog' in row for row in selected):
            traced(axes[1], 'observed_backlog', 'Backlog used to scale measured F', '#64748b')
        panel(axes[1], 'Queue count: native scrape and the snapshot used by SMG', 'Requests')
        sampled(axes[2], 'vllm_running', 'Native running', '#7c3aed')
        sampled(axes[2], 'running', 'SMG running snapshot', '#d97706')
        panel(axes[2], 'Running count: baseline eligibility and queue interpretation', 'Requests')
        for key, label in [('mean_prefill_tokens', 'Rolling uncached prompt mean'),
                           ('median_prefill_tokens', 'Histogram prompt median'),
                           ('blended_prompt_tokens', 'Queue proxy tokens/request')]:
            sampled(axes[3], key, label)
        panel(axes[3], 'Prompt size used to convert queued requests to token work', 'Tokens/request')
        for key, label in [('prefill_tps', 'Observed prefill rate'), ('learned_capacity', 'Learned saturated rate'),
                           ('effective_capacity', 'Effective capacity C')]:
            sampled(axes[4], key, label)
        axes[4].axhline(floor, color='#b91c1c', linestyle='--', label=f'{floor:g} tokens/s floor')
        panel(axes[4], 'Prefill capacity learning; active request rate differs by engine', 'Tokens/second')
        sampled(axes[5], 'prefill_size_samples', 'Prompt-size sample count')
        panel(axes[5], 'Evidence weighting the 1,024-token / 32-sample prompt prior', 'Samples')
        for key, label in [('capacity_samples', 'Saturated capacity samples'),
                           ('baseline_samples', 'SMG isolated TTFT samples'),
                           ('observed_delay_samples', 'Observed busy-delay samples')]:
            sampled(axes[6], key, label)
        panel(axes[6], 'Learning windows (capacity min 8; baseline / delay min 4)', 'Samples')
        flags(axes[7], [(key, label, 'sample') for key, label in [
            ('capacity_learning_active', 'Saturated learning active'),
            ('capacity_ready', 'Admission capacity/delay ready'),
            ('throughput_fallback', 'Capacity floor selected'), ('queue_proxy', 'Queue uses prompt proxy')]])
        panel(axes[7], 'Learning and formula branch flags: each lane moves upward when true', 'Low=0; high=1')
    else:
        sampled(axes[1], 'pending_engine', 'Pending engine (bounded diagnostic)')
        sampled(axes[1], 'pending_first_text', 'Streams pending first text')
        if all('pending_first_text_requests' in row for row in selected):
            traced(axes[1], 'pending_first_text_requests', 'Pending first text at admission', '#059669')
        panel(axes[1], 'Pending streams are not counted as full remaining prompt work', 'Requests')
        sampled(axes[2], 'generation_tps', 'SMG generation rate')
        sampled(axes[2], '', 'Native generation counter Δ / scrape Δt', values=rate(metrics, 'generation_tokens'))
        panel(axes[2], 'Generation throughput: diagnostic, not C', 'Tokens/second')
        sampled(axes[3], 'kv_usage', 'SMG KV usage')
        sampled(axes[3], 'vllm_kv_usage', 'Native KV usage')
        panel(axes[3], 'KV cache occupancy (fraction; no beta / KV penalty in this formula)', 'Fraction')
        sampled(axes[4], 'visibility_seconds', 'SMG busy first-SSE-to-text correction (diagnostic)')
        sampled(axes[4], '', 'Engine queue histogram interval mean', values=interval_mean(metrics, 'queue_sum', 'queue_count'))
        sampled(axes[4], '', 'Engine first-token histogram interval mean', values=interval_mean(metrics, 'ttft_sum', 'ttft_count'))
        sampled(axes[4], '', 'Engine prefill-time interval mean', values=interval_mean(metrics, 'prefill_time_sum', 'prefill_time_count'))
        panel(axes[4], 'Native stage observations have different reporting / text boundaries', 'Seconds')
        for key, label in [('queue_count', 'Native queue observations'), ('ttft_count', 'Native TTFT observations'),
                           ('prefill_time_count', 'Native prefill-time observations')]:
            sampled(axes[5], '', label+' Δ', values=delta(metrics, key))
        panel(axes[5], 'Observation counts gained per scrape interval', 'Observations/interval')
        for key, label in [('queue_sum', 'Queue time'), ('ttft_sum', 'Engine first-token time'),
                           ('prefill_time_sum', 'Prefill time')]:
            sampled(axes[6], '', label+' sum Δ', values=delta(metrics, key))
        panel(axes[6], 'Histogram time sums gained per interval (not request latency)', 'Seconds/interval')
        for key, label in [('prompt_compute_tokens', 'Native prompt/local-compute counter'),
                           ('completed_prefill_tokens', 'Completed uncached prefill counter'),
                           ('generation_tokens', 'Generation counter')]:
            sampled(axes[7], '', label+' Δ', values=delta(metrics, key))
        panel(axes[7], 'Token counters gained per interval; SGLang prompt counter includes cached tokens', 'Tokens/interval')
        for key, label in [('rejections', 'Hard rejections'), ('shadow_rejections', 'Shadow would-reject'),
                           ('unknown', 'Unready decisions')]:
            sampled(axes[8], key, label)
        panel(axes[8], 'SMG decision counters for this router run; absent series stays missing', 'Cumulative decisions')
        # A compact strip retains request-level baseline availability beside diagnostic inputs.
        flags(axes[9], [('baseline_available', 'Baseline available', 'trace'), ('ready', 'Admission ready', 'trace')])
        panel(axes[9], 'Request-level decision context: each lane moves upward when true', 'Low=0; high=1')
    for ax in axes:
        for index, phase in enumerate(info['phases']):
            if focus and phase['name'] != focus:
                continue
            ax.axvspan(phase['start_seconds'], phase['last_start_seconds'],
                       color='#cbd5e1', alpha=.16 if index % 2 == 0 else .04, zorder=0)
        ax.set_xlim(limits)
        ax.set_ylim(bottom=0 if ax.get_ylim()[0] >= -.01 else ax.get_ylim()[0])
        if isinstance(ax.yaxis.get_major_formatter(), ScalarFormatter):
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.grid(alpha=.18)
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.006), fontsize=8,
                  ncol=4, framealpha=.9, borderaxespad=0)
        ax.tick_params(axis='x', labelbottom=True)
    # Numbered phases refer to the accompanying manifest/table, avoiding overlapping long labels.
    for index, phase in enumerate(info['phases']):
        if focus and phase['name'] != focus:
            continue
        top.text((phase['start_seconds']+phase['last_start_seconds'])/2, .98, str(index+1),
                  ha='center', va='top', transform=top.get_xaxis_transform(), fontsize=8,
                  bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
    axes[-1].set_xlabel('Seconds since harness start (baseline, load and recovery retained)')
    scope = 'Full run' if not focus else 'Zoom: '+focus
    fig.suptitle(f'{title} · {view.title()} · {scope}\n'
                 'First text includes reasoning/content; estimates are before admission; missing metrics are gaps',
                 fontsize=14)
    fig.savefig(output, dpi=130)
    plt.close(fig)
    return dict(filename=output.name, view=view, focus_case=focus, xlim_seconds=list(limits))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='One run or root containing model/suite run directories')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--title', help='Display title for a single input run')
    parser.add_argument('--zoom-suite', action='append', default=[], help='Also plot the busiest mixed phase for this suite (repeatable)')
    parser.add_argument('--capacity-floor', type=float, default=2000, help='Reference line only; replay uses logged capacity')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ.setdefault('MPLCONFIGDIR', str(args.output_dir/'.matplotlib'))
    os.environ.setdefault('XDG_CACHE_HOME', str(args.output_dir/'.cache'))
    runs = [args.input] if (args.input/'metrics.tsv').exists() else sorted(
        p.parent for p in args.input.rglob('metrics.tsv') if (p.parent/'paired-requests.json').exists())
    if not runs:
        parser.error('No analyzed run directories found')
    if args.title and len(runs) != 1:
        parser.error('--title requires a single run')
    manifest = []
    for path in runs:
        metrics, requests, info = load_run(path)
        label = args.title or f"{info['engine']} / {info['model']} / "+(
            'Calibration (inference)' if info['suite']=='train' else info['suite'])
        scopes = [None]
        if info['suite'] in args.zoom_suite:
            mixed = {r['case'] for r in requests if 'mixed' in r['case'] or 'mixture' in r['case']}
            if mixed:
                scopes.append(max(sorted(mixed), key=lambda case: sum(r['case']==case for r in requests)))
        info['charts'] = []
        prefix = re.sub(r'[^a-zA-Z0-9._-]+', '-', f"{info['model']}-{info['suite']}")
        for focus in scopes:
            for view in ['formula', 'learning', 'diagnostics']:
                output = args.output_dir/f"{prefix}-{view}{'-mixed-zoom' if focus else ''}.png"
                if output.exists() and not args.overwrite:
                    raise FileExistsError(f'{output}: use --overwrite to replace')
                info['charts'].append(draw(path, metrics, requests, info, view, focus, output, label, args.capacity_floor))
        manifest.append(info)
        print(f"{prefix}: {len(info['charts'])} charts, {info['requests']} exact formula checks", flush=True)
    (args.output_dir/'metric-plot-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
