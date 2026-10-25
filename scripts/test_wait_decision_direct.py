import json
import unittest
from copy import deepcopy
import wait_decision_calibration as legacy
import wait_decision_direct as direct
from replay_wait_decision import audit_schema, replay, verify_snapshot
from replay_wait_decision_direct import replay_opportunities
from test_wait_decision_replay import event_fixture, snapshot


class DirectTests(unittest.TestCase):
    def supported(self):
        state = direct.State(); state.poll(0.,7,True,100.)
        for i in range(16):
            s = direct.snapshot_from_v1(snapshot()); s['scalar_gain'] = 11.
            state.register(i, 0., s); state.label(i, 202., 202., True, True)
        state.poll(203., 7, True, 300.)
        return state

    def test_unsupported_falls_back_to_scalar_but_training_basis_is_raw(self):
        s=snapshot(); s['scalar_gain']=7.; s['x']=700.
        raw=direct.snapshot_from_v1(s); self.assertEqual(raw['x'],100.)
        values=direct.State().predictions(raw,0.,True)
        for p in values:
            self.assertEqual(p['extra'],700.)
            self.assertTrue(p['scalar_fallback']);self.assertEqual(p['source_arm'],'live_n_scalar')
            self.assertEqual(p['coefficients'],[7.,0.,0.,0.])

    def test_supported_training_and_predictions_are_independent_of_legacy_gain(self):
        state=self.supported();s=direct.snapshot_from_v1(snapshot(210.))
        a=state.predictions(s,210.,True);s['scalar_gain']=19.;b=state.predictions(s,210.,True)
        self.assertEqual(a,b);self.assertFalse(a[0]['scalar_fallback'])
        self.assertAlmostEqual(a[0]['coefficients'][0],2.1/1.1)
        self.assertAlmostEqual(a[0]['extra'],100.*2.1/1.1)

    def test_nonfinite_fallback_never_becomes_available(self):
        s=direct.snapshot_from_v1(snapshot());s['scalar_gain']=float('nan')
        p=direct.State().predictions(s,0.,True)[0]
        self.assertFalse(p['available']);self.assertEqual(p['contributions'][0],p['extra'])
        s['scalar_gain']=-1.;p=direct.State().predictions(s,0.,True)[0]
        self.assertEqual(p['contributions'],[0.,0.,0.,0.]);self.assertEqual(p['unbounded_extra'],-100.)

    def test_unavailable_uses_scalar_diagnostics_but_keeps_unavailable(self):
        state=self.supported();s=direct.snapshot_from_v1(snapshot());s['scalar_gain']=9.
        p=state.predictions(s,220.,False)[0]
        self.assertEqual(p['extra'],900.);self.assertTrue(p['scalar_fallback']);self.assertFalse(p['available'])

    def test_live_and_original_zero_and_tiny_eligibility(self):
        s=snapshot();s.update(live_queue_extra=0.,original_queue_extra=10.,scalar_gain=10.,x=0.)
        raw=direct.snapshot_from_v1(s);self.assertEqual(raw['x'],0.);self.assertFalse(raw['eligible'])
        s.update(live_queue_extra=None,original_queue_extra=.002,scalar_gain=.1,x=.0002,eligible=False)
        raw=direct.snapshot_from_v1(s);self.assertEqual(raw['source'],'original_n_fallback');self.assertTrue(raw['eligible'])
        s.update(original_queue_extra=.0002,scalar_gain=10.,x=.002,eligible=True)
        self.assertFalse(direct.snapshot_from_v1(s)['eligible'])

    def test_direct_snapshot_rejects_scalar_contamination(self):
        s=direct.snapshot_from_v1(snapshot());state=direct.State();state.poll(0.,7,True,5.)
        verify_snapshot(s,state,5.,method=direct.METHOD);s['scalar_gain']=2.;s['x']=200.
        with self.assertRaisesRegex(ValueError,'original X'):verify_snapshot(s,state,5.,method=direct.METHOD)

    def test_v1_opportunity_parity(self):
        source=event_fixture();audit=replay(source);actual=replay_opportunities(source,direct_basis=False)
        self.assertEqual(audit['labels'][0]['label'],actual['labels'][0]['label'])
        self.assertEqual(audit['admissions'][0]['predictions'],actual['admissions'][0]['predictions'])

    def test_newly_eligible_terminal_is_reconstructed_without_old_accepted_label(self):
        events=event_fixture();admission=events[4];p=json.loads(admission['decision_json'])
        p['snapshot'].update(x=.0002,scalar_gain=.1,original_queue_extra=.002,live_queue_extra=.002,eligible=False)
        admission['decision_json']=json.dumps(p)
        events=[e for e in events if e['event']!='decision_label']
        actual=replay_opportunities(events,direct_basis=True)
        self.assertEqual(len(actual['labels']),1)
        self.assertEqual(actual['labels'][0]['label']['snapshot']['x'],.002)
        self.assertEqual(actual['labels'][0]['label']['at'],5.2)
        self.assertAlmostEqual(actual['labels'][0]['label']['elapsed'],4.)
        self.assertTrue(actual['admissions'][0]['predictions'][0]['scalar_fallback'])

    def test_previously_eligible_terminal_is_not_reused_when_direct_input_too_small(self):
        events=event_fixture();p=json.loads(events[4]['decision_json'])
        p['snapshot'].update(x=.002,scalar_gain=10.,original_queue_extra=.0002,live_queue_extra=.0002,eligible=True)
        events[4]['decision_json']=json.dumps(p)
        actual=replay_opportunities(events,direct_basis=True)
        self.assertEqual(actual['labels'],[])

    def test_poll_bound_opportunity_then_exact_upgrade_keeps_first_bank(self):
        events=event_fixture();admission=json.loads(events[4]['decision_json'])
        admission['snapshot'].update(x=.0002,scalar_gain=.1,original_queue_extra=.002,live_queue_extra=.002,eligible=False)
        events[4]['decision_json']=json.dumps(admission)
        events=events[:5]
        events.append(dict(event='feature_deadline_resolution',worker='w',calibration_epoch=0,request_id='r',
            diagnostic_time_seconds=301.2,observed_time_seconds=301.2,request_started_time_seconds=1.1,resolution_source='poll'))
        events.append(dict(event='decision_terminal',worker='w',calibration_epoch=0,diagnostic_time_seconds=401.2,
            decision_method=legacy.METHOD,decision_json=json.dumps(dict(id=1,observed_at=401.1,outcome='first_text'))))
        events.append(dict(event='feature_outcome',worker='w',calibration_epoch=0,request_id='r',
            diagnostic_time_seconds=401.2,outcome='first_text'))
        events.append(dict(event='first_text',request_id='r',ttft_seconds=400.))
        value=replay_opportunities(events,direct_basis=True)
        self.assertEqual(len(value['labels']),2)
        bound,exact=[r['label'] for r in value['labels']]
        self.assertEqual(bound['elapsed'],300.);self.assertFalse(bound['exact'])
        self.assertEqual(exact['first_label_at'],301.2);self.assertTrue(exact['exact'])
        self.assertEqual(bound['snapshot'],exact['snapshot'])

    def test_direct_schema_requires_fallback_and_rejects_mixed_methods(self):
        events=event_fixture();poll=deepcopy(events[1]);poll['decision_method']=direct.METHOD
        self.assertFalse(audit_schema([poll],direct.METHOD)['ok'])
        p=json.loads(poll['decision_json']);s=direct.snapshot_from_v1(p['snapshot']);state=direct.State();state.poll(0.,7,True,5.)
        p.update(snapshot=s,predictions=state.predictions(s,0.,True));poll['decision_json']=json.dumps(p)
        self.assertTrue(audit_schema([poll],direct.METHOD)['ok'])
        self.assertFalse(audit_schema([poll,events[1]],direct.METHOD)['ok'])

    def test_full_v2_audit_accepts_direct_schema_without_changing_v1(self):
        events=event_fixture();state=direct.State();state.poll(0.,7,True,5.)
        for e in events:
            if 'decision_json' not in e:continue
            e['decision_method']=direct.METHOD;p=json.loads(e['decision_json'])
            if p.get('snapshot') is not None:
                p['snapshot']=direct.snapshot_from_v1(p['snapshot'])
                if 'predictions' in p:
                    p['predictions']=state.predictions(p['snapshot'],p['snapshot']['at'],True)
            if 'label' in p:p['label']['snapshot']=direct.snapshot_from_v1(p['label']['snapshot'])
            e['decision_json']=json.dumps(p)
        result=replay(events,direct.METHOD)
        self.assertTrue(result['ok']);self.assertEqual(result['method'],direct.METHOD)
        self.assertTrue(result['admissions'][0]['predictions'][0]['scalar_fallback'])
        self.assertTrue(replay(event_fixture())['ok'])

    def test_current_profiles_require_v3(self):
        from pathlib import Path
        for name in ('wait-ttft-deadline300.json','wait-ttft-observe420.json'):
            self.assertEqual(json.loads(Path(__file__).with_name(name).read_text())['expected_decision_method'],'decision_live_queue_slope_ridge_hinge_v3')


if __name__=='__main__':unittest.main()
