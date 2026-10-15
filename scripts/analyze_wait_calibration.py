#!/usr/bin/env python3
"""Audit causal online TTFT calibration and compare exact same-request forecasts.

Input is one harness suite directory containing requests.tsv, summary.json and
router-events.jsonl. Logs may include previous suites from the same router.
No offline fit is substituted for the live calibration forecasts.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics


def median_scale(history):
    if len(history) < 4:
        return 1.0
    # Stable ratio-only sort matches Rust; tied ratios retain arrival order.
    values = sorted(history, key=lambda entry: entry[0])
    largest = max(weight for _, weight in values)
    # Match Rust f64 iterator summation, rather than Python >=3.12's
    # compensated sum: an exact half-weight tie can choose another endpoint.
    total = 0.0
    for _, weight in values:
        total += weight / largest
    target = total / 2
    cumulative = 0
    for ratio, weight in values:
        cumulative += weight / largest
        if cumulative >= target:
            return ratio
    return 1.0


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=2e-8, abs_tol=2e-8)


def events(path):
    for line in path.read_text().splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        yield obj.get('fields', {}) | obj


def audit(log):
    """Reconstruct each state version using accepted sequence, not log ordering."""
    updates = defaultdict(dict)
    selected = []
    feedback = []
    errors = []
    for event in log:
        if event.get('event') == 'calibration_forecast':
            selected.append(event)
        if event.get('event') != 'calibration_feedback':
            continue
        feedback.append(event)
        if event['reason'] == 'accepted':
            key = (event['worker'], event['current_calibration_epoch'], event['calibration_state'])
            sequence = event['sequence_after']
            if sequence in updates[key]:
                errors.append(['duplicate_accepted_sequence', key, sequence])
            updates[key][sequence] = event
    versions = {}
    feedback_times = {}
    for key, changes in updates.items():
        history = deque(maxlen=32)
        versions[key + (0,)] = (1.0, 0)
        for sequence, event in sorted(changes.items()):
            if sequence != event['sequence_before'] + 1 or sequence != max((k[-1] for k in versions if k[:-1] == key), default=0) + 1:
                errors.append(['missing_history_sequence', key, sequence])
            if event['calibration_epoch'] != event['current_calibration_epoch']:
                errors.append(['accepted_old_epoch', event['request_id']])
            if not close(event['scale_before'], median_scale(history)) or event['sample_count_before'] != len(history):
                errors.append(['incorrect_feedback_before', event['request_id']])
            ratio = max(event['ttft_seconds'] - event['baseline_seconds'], 0) / event['raw_excess_seconds']
            history.append((ratio, event['raw_excess_seconds']))
            value = median_scale(history)
            if not close(value, event['scale_after']) or len(history) != event['sample_count_after']:
                errors.append(['incorrect_feedback_after', event['request_id']])
            versions[key + (sequence,)] = (value, len(history))
            if event.get('timestamp'):
                feedback_times[key + (sequence,)] = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
            if event['admission_observation_sequence'] > event['sequence_before']:
                errors.append(['feedback_uses_own_or_future_observation', event['request_id']])
    for event in selected:
        seq = event['calibration_observation_sequence']
        key = (event['worker'], event['calibration_epoch'], event['calibration_state'], seq)
        expected = versions.get(key, (1.0, 0) if seq == 0 else None)
        if expected is None or not close(expected[0], event['calibration_scale']) or expected[1] != event['calibration_sample_count']:
            errors.append(['incorrect_selected_history', event['request_id'], key])
        if key in feedback_times and event.get('timestamp'):
            if feedback_times[key] > datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00')):
                errors.append(['forecast_uses_future_feedback', event['request_id']])
        base = event['baseline_seconds']
        extra = event['predicted_excess_seconds']
        if not close(event['predicted_ttft_seconds'], base + extra) or not close(event['calibrated_ttft_seconds'], base + event['calibration_scale'] * extra):
            errors.append(['incorrect_forecast_arithmetic', event['request_id']])
        expected_reject = event['ready'] and math.isfinite(event['calibrated_excess_seconds']) and event['calibrated_excess_seconds'] >= event['threshold_seconds']
        if expected_reject != event['calibrated_would_reject']:
            errors.append(['incorrect_calibrated_readiness_gate', event['request_id']])
    selected_by_id = {event['request_id']: event for event in selected}
    for event in feedback:
        forecast = selected_by_id.get(event['request_id'])
        if not forecast:
            errors.append(['feedback_missing_selected_snapshot', event['request_id']])
            continue
        for source, target in [('baseline_seconds', 'baseline_seconds'), ('predicted_excess_seconds', 'raw_excess_seconds'), ('calibration_scale', 'admission_scale'), ('calibration_observation_sequence', 'admission_observation_sequence'), ('calibration_epoch', 'calibration_epoch')]:
            if not close(forecast[source], event[target]):
                errors.append(['feedback_changed_saved_input', event['request_id'], source])
    return dict(ok=not errors, errors=errors[:100], error_count=len(errors),
                selected_forecasts=len(selected), feedback=len(feedback),
                accepted=sum(len(v) for v in updates.values()),
                feedback_reasons=dict(Counter(event['reason'] for event in feedback)),
                history_versions=len(versions), states=len(updates))


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def score(rows):
    result = {'n': len(rows)}
    if not rows:
        return result
    result['actual_mean'] = statistics.mean(r['actual_ttft'] for r in rows)
    result['actual_p95'] = percentile([r['actual_ttft'] for r in rows], .95)
    result['client_minus_smg_p50_p95'] = [percentile([r['client_ttft'] - r['actual_ttft'] for r in rows], q) for q in [.5, .95]]
    for name in ['raw', 'calibrated']:
        errors = [r[name + '_ttft'] - r['actual_ttft'] for r in rows]
        counts = Counter()
        for r in rows:
            late = r['actual_extra'] >= 5
            reject = r[name + '_would_reject']
            counts['tp' if late and reject else 'fn' if late else 'fp' if reject else 'tn'] += 1
        positives = counts['tp'] + counts['fn']
        negatives = counts['fp'] + counts['tn']
        result[name] = dict(mean=statistics.mean(r[name + '_ttft'] for r in rows),
                            bias=statistics.mean(errors), mae=statistics.mean(abs(e) for e in errors),
                            p95_abs_error=percentile([abs(e) for e in errors], .95),
                            **{key: counts[key] for key in ['tp', 'fn', 'fp', 'tn']},
                            positives=positives, negatives=negatives,
                            fnr=counts['fn'] / positives if positives else None,
                            fpr=counts['fp'] / negatives if negatives else None,
                            ready_fn=sum(r['ready'] and r['actual_extra'] >= 5 and not r[name + '_would_reject'] for r in rows),
                            unready_fn=sum(not r['ready'] and r['actual_extra'] >= 5 for r in rows))
    result['unready'] = sum(not r['ready'] for r in rows)
    result['calibration_mature'] = sum(r['calibration_count'] >= 4 for r in rows)
    result['baseline_missing'] = sum(not r['baseline_available'] for r in rows)
    result['scale_p10_p50_p90'] = [percentile([r['scale'] for r in rows], q) for q in [.1, .5, .9]]
    return result


def analyze(path):
    requests = list(csv.DictReader((path / 'requests.tsv').open(), delimiter='\t'))
    metrics = list(csv.DictReader((path / 'metrics.tsv').open(), delimiter='\t'))
    def clean_metric(key):
        return [v for row in metrics if math.isfinite(v := float(row.get(key) or 'nan')) and v >= 0]
    sample_times = clean_metric('elapsed_seconds')
    loads = {key: max(clean_metric(key), default=None) for key in ['waiting', 'running', 'kv_usage', 'pending_first_text', 'effective_capacity']}
    loads['samples'] = len(metrics)
    loads['max_sample_gap_seconds'] = max((b - a for a, b in zip(sample_times, sample_times[1:])), default=None)
    summary = json.loads((path / 'summary.json').read_text())
    log = list(events(path / 'router-events.jsonl'))
    causality = audit(log)
    prefix = 'classifier-' + str(summary['seed']) + '-'
    by_event = defaultdict(dict)
    duplicates = Counter()
    for event in log:
        request_id = event.get('request_id', '')
        name = event.get('event')
        if not request_id.startswith(prefix) or name not in ['forecast', 'calibration_forecast', 'first_text', 'calibration_feedback']:
            continue
        if request_id in by_event[name]:
            duplicates[name] += 1
        by_event[name][request_id] = event
    baselines = defaultdict(list)
    def shape(row):
        return 'cached' if row['case'] in ['baseline-cached', 'cached-burst', 'cached-burst-warm'] else row['prompt_words'] + '-out' + row['output_tokens']
    for row in requests:
        text = by_event['first_text'].get(row['opc_request_id'])
        if row['case'].startswith('baseline-') and text:
            baselines[shape(row)].append(text['ttft_seconds'])
    matched = {key: statistics.median(value) for key, value in baselines.items()}
    rows = []
    rejected = []
    missing = []
    for row in requests:
        rid = row['opc_request_id']
        gate = by_event['forecast'].get(rid)
        if row['status'] == '503':
            rejected.append(dict(case=row['case'], request_id=rid, gate=gate, error=row['error']))
            continue
        if row['status'] != '200':
            missing.append([rid, 'HTTP ' + row['status']])
            continue
        forecast = by_event['calibration_forecast'].get(rid)
        text = by_event['first_text'].get(rid)
        feedback = by_event['calibration_feedback'].get(rid)
        if not all([forecast, text, feedback, gate]):
            missing.append([rid, 'missing joined event'])
            continue
        baseline = matched[shape(row)]
        rows.append(dict(case=row['case'], request_id=rid, started_elapsed_seconds=float(row['started_elapsed_seconds']),
                         prompt_words=int(row['prompt_words']), actual_ttft=text['ttft_seconds'],
                         client_ttft=float(row['ttft_seconds']), actual_extra=max(text['ttft_seconds'] - baseline, 0),
                         matched_baseline=baseline, admission_baseline=forecast['baseline_seconds'],
                         baseline_available=forecast['baseline_available'], raw_ttft=forecast['predicted_ttft_seconds'],
                         raw_extra=forecast['predicted_excess_seconds'], calibrated_ttft=forecast['calibrated_ttft_seconds'],
                         calibrated_extra=forecast['calibrated_excess_seconds'], scale=forecast['calibration_scale'],
                         calibration_count=forecast['calibration_sample_count'], calibration_sequence=forecast['calibration_observation_sequence'],
                         calibration_state=forecast['calibration_state'], calibration_epoch=forecast['calibration_epoch'],
                         ready=forecast['ready'], raw_would_reject=forecast['would_reject'],
                         calibrated_would_reject=forecast['calibrated_would_reject'],
                         gate_would_reject=gate['would_reject'], gate_excess=gate['predicted_excess_seconds'],
                         feedback_reason=feedback['reason'], effective_capacity=gate['effective_capacity'],
                         queue_tokens=gate['queued_tokens'], dispatch_tokens=gate['dispatched_tokens_since_poll'],
                         observed_delay_floor=gate['observed_delay_floor'], native_queue_delay=gate['native_queue_delay']))
    checks = dict(causal_replay=causality['ok'], no_duplicate_events=not duplicates,
                  all_successes_joined=not missing,
                  all_selected_receive_feedback=len(rows) == len(by_event['calibration_feedback']),
                  all_shadow_requests_200=all(row['status'] == '200' for row in requests) if summary['suite'] != 'hard' else True,
                  all_503_from_raw_gate=all(r['gate'] and r['gate']['would_reject'] and r['gate']['ready'] and r['gate']['predicted_excess_seconds'] >= r['gate']['threshold_seconds'] and 'estimated wait budgets' in r['error'] for r in rejected),
                  hard_healthy_recovery_no_rejections=not any(r['case'] in ['hard-low', 'hard-recovery'] for r in rejected),
                  raw_gate_budget_arithmetic=all(e['would_reject'] == bool(e['ready'] and e['predicted_excess_seconds'] >= e['threshold_seconds']) for e in by_event['forecast'].values()))
    nonbaseline = [row for row in rows if not row['case'].startswith('baseline-')]
    low_names = ['sparse-short', 'sparse-long', 'cached-burst', 'cached-burst-warm', 'recovery-sparse', 'heldout-low', 'heldout-recovery', 'cold-recovery-sparse', 'hard-low', 'hard-recovery']
    low = [row for row in nonbaseline if row['case'] in low_names]
    high = [row for row in nonbaseline if row['case'] not in low_names]
    result = dict(suite=summary['suite'], engine=summary['engine'], image=summary['image'],
                  requests=len(requests), pairs=len(rows), missing=missing, duplicates=dict(duplicates), checks=checks, sampled_load=loads,
                  causal_audit=causality, baselines=matched, baseline_counts={k: len(v) for k, v in baselines.items()},
                  feedback_reasons=dict(Counter(r['feedback_reason'] for r in rows)),
                  rejections=len(rejected), rejections_by_phase=dict(Counter(r['case'] for r in rejected)),
                  raw_rejections_calibrated_gate_would_permit=sum(not r['gate'].get('calibrated_would_reject', False) for r in rejected if r['gate']),
                  all=score(nonbaseline), low=score(low), high=score(high),
                  phases={case: score([r for r in rows if r['case'] == case]) for case in dict.fromkeys(r['case'] for r in rows)},
                  states={state: score([r for r in nonbaseline if r['calibration_state'] == state]) for state in ['capacity_unready', 'capacity_ready']},
                  notes=['Raw/calibrated accuracy uses the same selected snapshot. Original pre-admission gate is audited separately.',
                         'Truth is prompt-shape-matched median low-load first-text TTFT plus the shared five-second extra-delay budget.',
                         'Rejected hard-mode requests have no observed counterfactual TTFT. Calibrated gate remains hypothetical.',
                         'The holdout is later unseen traffic with online updates enabled; it is not a frozen fit.'])
    (path / 'paired-calibration.json').write_text(json.dumps(rows, indent=2) + '\n')
    (path / 'calibration-analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(path=str(path), checks=checks, pairs=len(rows), rejections=len(rejected), low=result['low'], high=result['high']), indent=2))
    if not all(checks.values()):
        raise RuntimeError('Correlation, causality or operational audit failed; preserve evidence')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    analyze(parser.parse_args().input)
