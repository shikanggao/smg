#!/usr/bin/env python3
"""Meaningful intervention/censoring fixtures; no engine or network access."""
from contextlib import redirect_stdout
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('stratifier', ROOT / 'stratify_wait_deadline.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def fixture(path, intervention=True):
    raw, paired = [], []

    def request(rid, case, start, end, ttft=None, *, complete=True,
                status=200, error='', cancelled=False, reject=False):
        r = dict(opc_request_id=rid, case=case, status=str(status),
                 started_elapsed_seconds=start, completed_elapsed_seconds=end,
                 ttft_seconds='' if ttft is None else ttft, e2e_seconds=end-start,
                 complete=complete, done_seen=complete, finish_reason_seen=complete,
                 cancelled=cancelled, error=error, original_field='retained-' + rid)
        raw.append(r)
        if status == 503:
            return
        paired.append(dict(request_id=rid, case=case, started_elapsed_seconds=start,
            completed_elapsed_seconds=end, client_ttft=ttft,
            actual_ttft=ttft-.1 if ttft is not None else None,
            e2e_seconds=end-start, complete=complete, cancelled=cancelled, error=error,
            completion_late=(end-start >= 300) if complete else None,
            first_text_late=ttft >= 300 if ttft is not None else None,
            raw_ttft=351. if reject else 101., calibrated_ttft=351. if rid == 'late-pending' or reject else 101.,
            raw_would_reject=reject, calibrated_would_reject=rid == 'late-pending' or reject,
            gate_would_reject=reject, ready=True))

    request('natural', 'background-natural', 0, 100, 1)
    if intervention:
        request('timeout', 'background-timeout', 0, 400, complete=False, error='TimeoutError()')
    request('late-pending', 'sentinel-pending', 0, 460, 450)
    request('accelerated', 'sentinel-accelerated', 200, 460, 250)
    request('cancel-before-deadline', 'sentinel-cancelled', 0, 100,
            complete=False, cancelled=True, error='planned cancellation')
    request('healthy-boundary', 'sentinel-healthy', 0, 260, 250, reject=True)
    request('paired-proof', 'sentinel-proof', 0, 340, 335)
    request('post-cutoff', 'sentinel-later', 450, 451, .1)
    request('rejected', 'sentinel-rejected', 0, .1, complete=False,
            status=503, error='estimated wait budgets exceeded')
    fields = list(dict.fromkeys(k for r in raw for k in r))
    with (path / 'requests.tsv').open('w') as f:
        writer=csv.DictWriter(f, fieldnames=fields, delimiter='\t'); writer.writeheader(); writer.writerows(raw)
    (path / 'paired-deadline.json').write_text(json.dumps(paired))
    (path / 'stages.json').write_text(json.dumps([dict(name='pressure', elapsed_seconds=0),
                                                dict(name='unload-background', elapsed_seconds=800)]))
    return raw, paired


class DeadlineStrataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.raw, self.paired = fixture(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def run_stratifier(self):
        with redirect_stdout(io.StringIO()):
            result = s.stratify(self.path)
        rows = json.loads((self.path / 'paired-deadline-stratified.json').read_text())
        return result, {r['request_id']: r for r in rows}

    def test_natural_done_completion_is_not_intervention(self):
        fixture(self.path, intervention=False)
        result, _ = self.run_stratifier()
        self.assertEqual((result['cutoff_elapsed_seconds'], result['cutoff_reason']), (800, 'planned_unload'))

    def test_earlier_timeout_freezes_truth_and_keeps_legacy_unload_literal(self):
        result, rows = self.run_stratifier()
        self.assertEqual((result['cutoff_elapsed_seconds'], result['cutoff_reason']), (400, 'background_client_timeout'))
        self.assertIsNone(rows['accelerated']['completion_deadline_at_cutoff'])
        self.assertTrue(rows['accelerated']['completed_before_unload'])
        self.assertFalse(rows['accelerated']['completed_before_cutoff'])

    def test_cancellation_before_deadline_stays_unknown_after_clock_passes(self):
        _, rows = self.run_stratifier()
        row = rows['cancel-before-deadline']
        self.assertIsNone(row['completion_deadline_at_cutoff'])
        self.assertIsNone(row['first_text_deadline_at_cutoff'])
        self.assertFalse(row['completion_pending_at_cutoff'])

    def test_already_crossed_deadline_known_even_when_exact_finish_later(self):
        result, rows = self.run_stratifier()
        self.assertTrue(rows['late-pending']['completion_deadline_at_cutoff'])
        self.assertTrue(rows['late-pending']['first_text_deadline_at_cutoff'])
        self.assertTrue(rows['late-pending']['first_text_pending_at_cutoff'])
        self.assertEqual(rows['late-pending']['actual_ttft'], 449.9)
        score = result['completion_deadline_at_cutoff']
        self.assertEqual((score['raw']['fn'], score['calibrated']['fn']), (2, 1))
        self.assertEqual((score['raw']['negatives'], score['raw']['unknown']), (1, 3))

    def test_completed_proof_and_healthy_boundary_require_pre_cutoff_finish(self):
        result, _ = self.run_stratifier()
        self.assertEqual(result['completed_paired_proof_before_cutoff'], 1)
        self.assertEqual(result['paired_proof_before_cutoff_ids'], ['paired-proof'])
        self.assertEqual(result['healthy_boundary_completed_before_cutoff_ids'], ['healthy-boundary'])

    def test_rejected_unknown_retained_without_entering_latency_denominators(self):
        result, _ = self.run_stratifier()
        all_rows = json.loads((self.path / 'requests-deadline-stratified.json').read_text())
        self.assertEqual(len(all_rows), len(self.raw))
        row = next(r for r in all_rows if r['request_id'] == 'rejected')
        self.assertTrue(row['counterfactual_unknown'])
        self.assertIsNone(row['completion_deadline_at_cutoff'])
        self.assertEqual(row['raw_request']['original_field'], 'retained-rejected')
        self.assertEqual(result['sentinel_count'], 6)

    def test_earlier_background_cancellation_is_cutoff(self):
        r = dict(self.raw[0], opc_request_id='cancel-bg', complete=False, done_seen=False,
                 finish_reason_seen=False, cancelled=True, error='cancelled', completed_elapsed_seconds=200)
        with (self.path / 'requests.tsv').open('a') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.raw[0]), delimiter='\t'); writer.writerow(r)
        result, _ = self.run_stratifier()
        self.assertEqual((result['cutoff_elapsed_seconds'], result['cutoff_reason']), (200, 'background_cancellation'))

    def test_controlled_background_rejection_does_not_cut_pressure_short(self):
        fixture(self.path, intervention=False)
        row = dict(self.raw[0], opc_request_id='rejected-bg', status='503', complete=False,
                   done_seen=False, finish_reason_seen=False, error='estimated wait budgets exceeded',
                   completed_elapsed_seconds=50)
        with (self.path / 'requests.tsv').open('a') as f:
            csv.DictWriter(f, fieldnames=list(self.raw[0]), delimiter='\t').writerow(row)
        result, _ = self.run_stratifier()
        self.assertEqual(result['cutoff_reason'], 'planned_unload')

    def test_background_error_and_premature_eof_are_interventions(self):
        for error, reason in [('ConnectionError()', 'background_error'), ('', 'background_incomplete_stream')]:
            with self.subTest(reason=reason):
                fixture(self.path)
                row = dict(self.raw[0], opc_request_id='error-bg', complete=False,
                           done_seen=False, finish_reason_seen=False, error=error, completed_elapsed_seconds=300)
                with (self.path / 'requests.tsv').open('a') as f:
                    csv.DictWriter(f, fieldnames=list(self.raw[0]), delimiter='\t').writerow(row)
                result, _ = self.run_stratifier()
                self.assertEqual((result['cutoff_elapsed_seconds'], result['cutoff_reason']), (300, reason))

    def test_provenance_matches_untouched_inputs(self):
        original = {n: (self.path / n).read_bytes() for n in ['requests.tsv','paired-deadline.json','stages.json']}
        result, _ = self.run_stratifier()
        self.assertEqual(result['stratifier_sha256'], hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest())
        for name, contents in original.items():
            self.assertEqual((self.path / name).read_bytes(), contents)
            self.assertEqual(result['input_sha256'][name], hashlib.sha256(contents).hexdigest())

    def test_render_retains_outlier_and_early_cutoff(self):
        rows = json.loads((self.path / 'paired-deadline.json').read_text())
        next(r for r in rows if r['request_id'] == 'late-pending')['calibrated_ttft'] = 100000.
        (self.path / 'paired-deadline.json').write_text(json.dumps(rows))
        self.run_stratifier()
        keys=['elapsed_seconds','sample_completed_elapsed_seconds','estimated_wait','calibrated_ttft',
              'calibration_scale','waiting','running','pending_first_text','client_background_active',
              'kv_usage','baseline_seconds','observed_delay_floor','effective_capacity']
        with (self.path / 'metrics.tsv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=keys,delimiter='\t');writer.writeheader()
            writer.writerows([dict.fromkeys(keys, t) for t in [0,.5,1]])
        (self.path / 'deadline-analysis.json').write_text(json.dumps({'measured_smg_sentinel_over_300':2}))
        env=os.environ | {'MPLCONFIGDIR':str(self.path/'mpl'),'XDG_CACHE_HOME':str(self.path/'cache')}
        subprocess.run([sys.executable,str(ROOT/'plot_wait_deadline_stratified.py'),'--input',str(self.path),
                        '--title','Synthetic intervention fixture'],env=env,check=True,stdout=subprocess.DEVNULL)
        manifest=json.loads((self.path/'graphs/deadline-stratified-plot-manifest.json').read_text())
        self.assertEqual(manifest['cutoff_reason'], 'background_client_timeout')
        self.assertEqual(manifest['maximum_probe_value_seconds'], 100000.)
        for name in manifest['figures']:
            self.assertGreater((self.path/'graphs'/name).stat().st_size, 10000)


if __name__ == '__main__':
    unittest.main()
