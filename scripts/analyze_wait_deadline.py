#!/usr/bin/env python3
"""Audit live 300-second pressure traces; do not invent rejected outcomes."""
from pathlib import Path
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
audit_path = ROOT / 'analyze_wait_calibration.py'
if not audit_path.exists():
    audit_path = ROOT / 'analyze.py'
spec = importlib.util.spec_from_file_location('calibration_audit', audit_path)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
DEADLINE = 300.0


def number(row, key):
    try:
        value = float(row.get(key, ''))
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def truth(value):
    return str(value).lower() in ['true', '1']


def outcomes(request):
    client_ttft = number(request, 'ttft_seconds')
    e2e = number(request, 'e2e_seconds')
    complete = truth(request.get('complete'))
    crossed = truth(request.get('completion_deadline_crossed_without_finish')) or e2e is not None and e2e >= DEADLINE
    completion = e2e >= DEADLINE if complete and e2e is not None else True if crossed or client_ttft is not None and client_ttft >= DEADLINE else None
    first_text = client_ttft >= DEADLINE if client_ttft is not None else True if crossed else None
    return completion, first_text


def score(rows, outcome):
    result = {'n': len(rows)}
    for prediction in ['raw', 'calibrated', 'gate']:
        counts = Counter()
        ready_misses = unready_misses = 0
        for row in rows:
            late = row[outcome]
            if late is None:
                counts['unknown'] += 1
                continue
            reject = row[prediction + '_would_reject']
            counts['tp' if late and reject else 'fn' if late else 'fp' if reject else 'tn'] += 1
            if late and not reject:
                ready = row.get('gate_ready', row['ready']) if prediction == 'gate' else row['ready']
                if ready:
                    ready_misses += 1
                else:
                    unready_misses += 1
        positives = counts['tp'] + counts['fn']
        negatives = counts['fp'] + counts['tn']
        result[prediction] = dict(**{k: counts[k] for k in ['tp', 'fn', 'fp', 'tn', 'unknown']},
            positives=positives, negatives=negatives,
            fnr=counts['fn'] / positives if positives else None,
            fpr=counts['fp'] / negatives if negatives else None,
            ready_fn=ready_misses, unready_fn=unready_misses)
    return result


def analyze(path):
    requests = list(csv.DictReader((path / 'requests.tsv').open(), delimiter='\t'))
    metrics = list(csv.DictReader((path / 'metrics.tsv').open(), delimiter='\t'))
    summary = json.loads((path / 'summary.json').read_text())
    log = list(a.events(path / 'router-events.jsonl'))
    audit = a.audit(log)
    by_event = {name: {} for name in ['forecast', 'calibration_forecast', 'first_text', 'calibration_feedback']}
    duplicates = Counter()
    for event in log:
        name = event.get('event')
        if name not in by_event:
            continue
        rid = event.get('request_id')
        if rid in by_event[name]:
            duplicates[name] += 1
        by_event[name][rid] = event
    rows, rejected, missing, canceled_before_selection = [], [], [], []
    for request in requests:
        rid = request['opc_request_id']
        gate = by_event['forecast'].get(rid)
        status = str(request['status'])
        if status == '503':
            rejected.append(dict(request_id=rid, case=request['case'], gate=gate, error=request.get('error')))
            continue
        forecast = by_event['calibration_forecast'].get(rid)
        first = by_event['first_text'].get(rid)
        feedback = by_event['calibration_feedback'].get(rid)
        # Canceled queued background requests have no first text or feedback.
        if not forecast or not gate:
            if truth(request.get('cancelled')):
                canceled_before_selection.append(rid)
            else:
                missing.append([rid, 'selected or gate forecast absent'])
            continue
        client_ttft = number(request, 'ttft_seconds')
        e2e = number(request, 'e2e_seconds')
        complete = truth(request.get('complete'))
        completion_late, first_late = outcomes(request)
        # A still-open request aged beyond the deadline is a known miss but
        # its exact eventual TTFT is censored, never a fabricated 300s point.
        smg_ttft = first['ttft_seconds'] if first else None
        if complete and not first:
            missing.append([rid, 'complete request lacks SMG first text'])
        if first and not feedback:
            missing.append([rid, 'first text lacks feedback'])
        if first and (not math.isfinite(first['ttft_seconds']) or first['ttft_seconds'] <= 0):
            missing.append([rid, 'invalid SMG first-text time'])
        if first and feedback and not a.close(first['ttft_seconds'], feedback['ttft_seconds']):
            missing.append([rid, 'first-text and feedback time differ'])
        row = dict(request_id=rid, case=request['case'], phase=request.get('phase_at_start'),
            started_elapsed_seconds=number(request, 'started_elapsed_seconds'),
            completed_elapsed_seconds=number(request, 'completed_elapsed_seconds'),
            client_ttft=client_ttft, actual_ttft=smg_ttft, e2e_seconds=e2e,
            complete=complete, done_seen=truth(request.get('done_seen')),
            finish_reason_seen=truth(request.get('finish_reason_seen')),
            cancelled=truth(request.get('cancelled')), error=request.get('error'),
            completion_late=completion_late, first_text_late=first_late,
            first_text_censored=client_ttft is None,
            measured_smg_first_text_over_300=smg_ttft is not None and smg_ttft > DEADLINE,
            raw_ttft=forecast['predicted_ttft_seconds'], calibrated_ttft=forecast['calibrated_ttft_seconds'],
            raw_extra=forecast['predicted_excess_seconds'], calibrated_extra=forecast['calibrated_excess_seconds'],
            raw_would_reject=forecast['would_reject'], calibrated_would_reject=forecast['calibrated_would_reject'],
            ready=forecast['ready'], native_ready=forecast['native_ready'],
            native_waiting=forecast.get('native_waiting'), native_queue_visible=forecast.get('native_queue_visible'),
            baseline=forecast['baseline_seconds'], scale=forecast['calibration_scale'],
            target_scale=forecast.get('calibration_target_scale'),
            calibration_count=forecast['calibration_sample_count'],
            feedback_reason=feedback['reason'] if feedback else None,
            gate_would_reject=gate['would_reject'], gate_ready=gate['ready'], threshold=forecast['threshold_seconds'])
        rows.append(row)
    probes = [r for r in rows if not r['case'].startswith('background')]
    sentinel = [r for r in probes if r['case'].startswith('sentinel')]
    controls = [r for r in probes if r['case'].startswith(('low', 'recovery'))]
    measured = [r for r in sentinel if r['actual_ttft'] is not None]
    late = [r for r in measured if r['actual_ttft'] > DEADLINE]
    finite_metric = lambda key: [v for r in metrics if (v := number(r, key)) is not None and v >= 0]
    times = finite_metric('sample_completed_elapsed_seconds') or finite_metric('elapsed_seconds')
    load = {k: max(finite_metric(k), default=None) for k in ['waiting', 'running', 'kv_usage', 'pending_first_text', 'effective_capacity']}
    load.update(samples=len(metrics), longest_sample_interval_seconds=max((b - c for c, b in zip(times, times[1:])), default=None))
    # Integrate only adjacent valid samples; scrape failures do not prove
    # sustained occupancy across the missing interval.
    high_seconds = 0.0
    for previous, current in zip(metrics, metrics[1:]):
        start, stop = number(previous, 'sample_completed_elapsed_seconds'), number(current, 'sample_completed_elapsed_seconds')
        before, after = number(previous, 'vllm_kv_usage'), number(current, 'vllm_kv_usage')
        if start is not None and stop is not None and before is not None and after is not None and 0 <= stop - start <= 10 and before >= .9 and after >= .9:
            high_seconds += stop - start
    load['native_kv_at_least_90_percent_observed_seconds'] = high_seconds
    exact = [r for r in probes if r['actual_ttft'] is not None]
    accuracy = {k: dict(n=len(exact), mae=statistics.mean(abs(r[k + '_ttft'] - r['actual_ttft']) for r in exact) if exact else None) for k in ['raw', 'calibrated']}
    mode = summary.get('mode', path.name)
    stages = json.loads((path / 'stages.json').read_text())
    stage_times = {s['name']: s.get('elapsed_seconds') for s in stages}
    pressure_start, pressure_end = stage_times.get('adaptive-pressure'), stage_times.get('unload-background')
    pressure_duration = pressure_end - pressure_start if pressure_start is not None and pressure_end is not None else None
    minimum_hard = summary.get('protocol', {}).get('minimum_hard_pressure_seconds', 0)
    checks = dict(causal_replay=audit['ok'], no_duplicate_events=not duplicates,
        harness_completed=summary.get('result') == 'PASS' and (summary.get('client_coverage_met') is True if 'shadow' in mode else True),
        all_dispatched_joined=not missing,
        nonbackground_responses_valid=all(str(q['status']) == '503' or str(q['status']) == '200' and truth(q.get('complete')) and not q.get('error') for q in requests if not q['case'].startswith('background')),
        all_complete_have_done_and_finish=all(r['done_seen'] and r['finish_reason_seen'] for r in rows if r['complete']),
        no_success_before_first_text=all(r['client_ttft'] is not None for r in rows if r['complete']),
        completed_latencies_valid=all(r['e2e_seconds'] is not None and r['client_ttft'] is not None and r['client_ttft'] > 0 and r['e2e_seconds'] + 2e-8 >= r['client_ttft'] for r in rows if r['complete']),
        threshold_shared_300=all(a.close(r['threshold'], DEADLINE) for r in rows),
        all_rejections_from_raw_gate=all(r['gate'] and r['gate']['ready'] and r['gate']['would_reject'] and r['gate']['predicted_excess_seconds'] >= DEADLINE and 'estimated wait budgets' in (r['error'] or '') for r in rejected),
        low_recovery_no_rejections=not any(r['case'].startswith(('low', 'recovery')) for r in rejected),
        hard_pressure_covered=len([r for r in rejected if r['case'].startswith('sentinel')]) >= 3 or len(late) >= 3 if 'hard' in mode else True,
        minimum_hard_pressure_duration=not minimum_hard or pressure_duration is not None and pressure_duration >= minimum_hard if 'hard' in mode else True,
        observed_smg_sentinel_beyond_300=len(late) >= summary.get('protocol', {}).get('minimum_completed_late_sentinels', 3) if 'shadow' in mode else True)
    result = dict(mode=mode, deadline_seconds=DEADLINE, pressure_duration_seconds=pressure_duration, requests=len(requests), statuses=dict(Counter(r['status'] for r in requests)),
        pairs=len(rows), probes=len(probes), sentinel=len(sentinel), rejected=len(rejected),
        rejections_by_case=dict(Counter(r['case'] for r in rejected)),
        missing=missing[:100], canceled_before_selection=len(canceled_before_selection), duplicates=dict(duplicates), checks=checks, causal_audit=audit, sampled_load=load,
        measured_smg_sentinel_over_300=len(late), measured_client_sentinel_over_300=sum(r['client_ttft'] is not None and r['client_ttft'] > DEADLINE for r in sentinel),
        maximum_sentinel_smg_ttft=max((r['actual_ttft'] for r in measured), default=None),
        maximum_sentinel_client_ttft=max((r['client_ttft'] for r in sentinel if r['client_ttft'] is not None), default=None),
        complete_sentinel=sum(r['complete'] for r in sentinel),
        complete_sentinel_over_300=sum(r['complete'] and r['e2e_seconds'] > DEADLINE for r in sentinel),
        censored_sentinel=sum(r['first_text_censored'] for r in sentinel),
        healthy_controls=score(controls, 'completion_late'),
        healthy_boundary_240_300_seconds=score([r for r in sentinel if r['complete'] and 240 <= r['e2e_seconds'] < 300], 'completion_late'),
        late_boundary_300_360_seconds=score([r for r in sentinel if r['complete'] and 300 <= r['e2e_seconds'] < 360], 'completion_late'),
        sentinel_completion_deadline=score(sentinel, 'completion_late'),
        sentinel_first_text_deadline=score(sentinel, 'first_text_late'),
        probes_completion_deadline=score(probes, 'completion_late'),
        same_request_smg_ttft_accuracy=accuracy,
        notes=['The raw and calibrated gates predict extra TTFT above saved baseline, not full response completion.',
               'The client completion deadline is 300 seconds from dispatch; TTFT and completion outcomes are reported separately.',
               'Rejected requests have unknown counterfactual completion times and are excluded from FP/FN denominators.',
               'Open or canceled requests crossing the deadline are known misses with censored exact latency; no exact latency is invented.',
               'Healthy controls are separate low/recovery phases. Pressure sentinels use short outputs so late TTFT alone proves a late completion.'])
    (path / 'paired-deadline.json').write_text(json.dumps(rows, indent=2) + '\n')
    (path / 'deadline-analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['mode', 'checks', 'requests', 'sentinel', 'measured_smg_sentinel_over_300', 'maximum_sentinel_smg_ttft', 'sentinel_completion_deadline']}, indent=2))
    if not all(checks.values()):
        raise RuntimeError('Deadline evidence audit failed; preserve the attempt and investigate')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    analyze(p.parse_args().input)
