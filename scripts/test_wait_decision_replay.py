import unittest
from copy import deepcopy
from wait_decision_calibration import State, METHOD
from replay_wait_decision import audit_schema, verify_snapshot, verify_native, replay
from wait_ttft_readiness import Readiness
import json


def snapshot(now=0.):
    return dict(at=now,profile=7,baseline=2.,x=100.,source='live_n_scalar',original_queue_extra=100.,
                live_queue_extra=100.,scalar_gain=1.,fresh=True,native_ready=True,native_waiting=1.,
                poll_started=now,eligible=True,context=State().context(now))


class LifecycleTests(unittest.TestCase):
    def state(self):
        s=State();s.poll(0.,7,True,5.);return s
    def test_bound_upgrade_keeps_original_bank_position(self):
        s=self.state();s.register(1,0.,snapshot());s.label(1,300.,300.,False,False,300.)
        s.register(2,10.,snapshot(10.));s.label(2,310.,310.,False,False,300.)
        first=deepcopy(s.rows[0]);value=s.label(1,400.,400.,True,True)
        self.assertEqual([r['id'] for r in s.rows],[1,2]);self.assertEqual(value['first_label_at'],300.)
        self.assertEqual(value['snapshot'],first['snapshot']);self.assertEqual(len(s.pending),1)
    def test_evicted_bound_cannot_resurrect(self):
        s=self.state()
        for i in range(33):s.register(i,0.,snapshot());s.label(i,300.,300.,False,False,300.)
        self.assertEqual(len(s.rows),32);self.assertIsNone(s.label(0,400.,400.,True,True))
    def test_age_600_expires_even_when_feedback_is_recent(self):
        s=self.state();s.register(1,0.,snapshot());s.label(1,599.,599.,False,False)
        s.poll(600.,7,True,5.);self.assertFalse(s.rows)
    def test_growth_requires_continuity_and_complete_registry(self):
        s=self.state();s.poll(30.,7,True,40.);s.register(1,31.,snapshot(31.))
        self.assertEqual(s.context(31.)['values'][0],1.)
        s.poll(40.,7,True,5.);self.assertIsNone(s.context(40.)['values'][0])
        s.mark_incomplete();self.assertIsNone(s.context(40.)['oldest_seconds'])
    def test_zero_native_progress_is_valid(self):
        s=self.state();s.register(1,0.,snapshot())
        self.assertEqual(s.context(1.,0.,100.,True)['values'][2],1.)
        self.assertIsNone(s.context(1.,None,100.,True)['values'][2])
    def test_profile_reset_keeps_pending_but_rejects_old_training(self):
        s=self.state();s.register(1,0.,snapshot());s.poll(1.,8,True,5.)
        self.assertEqual(len(s.pending),1);self.assertIsNone(s.label(1,2.,2.,True,True))
    def test_snapshot_mutations_fail(self):
        s=self.state();v=snapshot();verify_snapshot(v,s,5.)
        for key,value in [('x',101.),('fresh',False),('source','original_n_scalar_fallback')]:
            broken=deepcopy(v);broken[key]=value
            with self.assertRaises(ValueError):verify_snapshot(broken,s,5.)
    def test_method_readiness_fails_closed(self):
        e=dict(event='decision_poll',worker='w',calibration_epoch=1,decision_method=METHOD,
               decision_json=json.dumps(dict(snapshot=None,predictions=None,state=State().diagnostics(),fresh=False,profile=None,max_gap=5.,available=False)))
        self.assertTrue(audit_schema([e])['ok']);self.assertFalse(audit_schema([])['ok'])
        e['decision_method']='old';self.assertFalse(audit_schema([e])['ok'])
        r=Readiness('x',expected_decision_method=METHOD);r.observe(e);self.assertTrue(r.errors)

if __name__=='__main__':unittest.main()


def event_fixture():
    def event(kind,now,p):
        return dict(event=kind,worker='w',calibration_epoch=0,diagnostic_time_seconds=now,
                    decision_method=METHOD,decision_json=json.dumps(p))
    native=dict(expected_rank_count=1,ranks=[dict(dp_rank=0,native_request_counts_available=True,
        native_request_counts_profile=1,timely_prefill_progress=0.,timely_prefill_progress_source='local_compute_tokens',
        timely_prefill_progress_interval=1.,learned_prefill_capacity=100.,prefill_capacity_sample_count=4,
        prefill_capacity_learning_active=True,num_waiting_reqs=1,num_running_reqs=0,
        num_waiting_uncached_tokens=100,num_waiting_uncached_tokens_available=True)])
    n=event('decision_native_poll',0.,native);n.update(poll_started_time_seconds=0.,outcome='success')
    state=State();state.poll(0.,7,True,5.);snap=snapshot();snap['context']=state.context(0.,0.,100.,True)
    poll=event('decision_poll',0.,dict(snapshot=snap,predictions=state.predictions(snap,0.,True),state=state.diagnostics(),
        profile=7,fresh=True,max_gap=5.,available=True))
    snap=snapshot(1.);snap['poll_started']=0.;snap['context']=state.context(1.,0.,100.,True)
    selected=dict(event='calibration_forecast',worker='w',request_id='r',calibration_epoch=0,diagnostic_time_seconds=1.,
        calibration_baseline_seconds=2.,queue_excess_seconds=100.,calibration_scale=1.,native_ready=True,native_waiting=1.)
    admission=event('decision_admission',1.2,dict(id=1,tracked=True,started=1.1,snapshot=snap,predictions=state.predictions(snap,1.,True)));admission['request_id']='r'
    state.register(1,1.1,snap);label=state.label(1,5.1,5.2,True,True)
    feedback=event('decision_label',5.2,dict(label=label));feedback['reason']='first_text'
    terminal=event('decision_terminal',5.2,dict(id=1,observed_at=5.1,outcome='first_text'))
    outcome=dict(event='feature_outcome',worker='w',request_id='r',calibration_epoch=0,diagnostic_time_seconds=5.2,outcome='first_text')
    return [n,poll,selected,admission,feedback,terminal,outcome,dict(event='first_text',request_id='r',ttft_seconds=4.)]


class ReplayTests(unittest.TestCase):
    def test_independent_causal_admission_and_terminal(self):
        result=replay(event_fixture());self.assertTrue(result['ok']);self.assertEqual(len(result['labels']),1)
    def test_saved_prediction_corruption(self):
        events=event_fixture();value=json.loads(events[3]['decision_json']);value['predictions'][0]['total']+=1
        events[3]['decision_json']=json.dumps(value)
        with self.assertRaises(ValueError):replay(events)
    def test_exact_label_requires_terminal_provenance(self):
        events=event_fixture();events[-1]['ttft_seconds']=300.
        with self.assertRaises(ValueError):replay(events)
    def test_native_source_and_rank_completeness_fail(self):
        for field,value in [('timely_prefill_progress_source','completed_prompt'),('timely_prefill_progress_interval',10.)]:
            events=event_fixture();native=json.loads(events[0]['decision_json']);native['ranks'][0][field]=value
            events[0]['decision_json']=json.dumps(native)
            with self.assertRaises(ValueError):replay(events)
    def test_missing_eligible_label_fails(self):
        events=event_fixture();del events[4]
        with self.assertRaises(ValueError):replay(events)


class ScoringTests(unittest.TestCase):
    def data(self):
        from replay_wait_decision import score_admissions
        admission=json.loads(event_fixture()[3]['decision_json'])
        admission.update(request_id='r',worker='w',epoch=0)
        paired=dict(opc_request_id='r',phase='pressure',first_text_late=False,started_elapsed_seconds=1.,
            feature_context_candidate=dict(snapshot=dict(threshold_seconds=300.)),
            pre_admission_recorded_extra_flags=dict(calibrated=False),pre_admission_total_ttft_predictions=dict(calibrated=102.))
        return score_admissions,admission,paired
    def test_missing_admission_cannot_shrink_denominator(self):
        fn,a,p=self.data()
        with self.assertRaises(ValueError):fn([], [p], 'capture','model')
    def test_extra_and_total_are_distinct(self):
        fn,a,p=self.data();a['snapshot']['x']=299.;a['snapshot']['baseline']=2.
        rows,report=fn([a],[p],'capture','model')
        self.assertEqual(report['extra']['scores']['live_n_scalar']['all']['fp'],0)
        self.assertEqual(report['total_ttft_300']['scores']['live_n_scalar']['all']['fp'],1)
    def test_rejected_owned_request_remains_unknown(self):
        fn,a,p=self.data();p['controlled_rejection']=True
        rows,report=fn([],[p],'capture','model')
        self.assertEqual(len(rows),1);self.assertIsNone(rows[0]['truth']);self.assertEqual(report['actual_rejected'],1)
