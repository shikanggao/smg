"""Focused source-schema fixtures for the independent deadline ledger audit."""
import copy
import json
import math
import unittest
try:
    from . import wait_feature_deadline as audit
except ImportError:
    import wait_feature_deadline as audit


class Trace:
    def __init__(self):
        self.events, self.rows, self.items, self.history = [], [], {}, []
        self.epoch, self.origin = 1, 0

    def context(self, now):
        counts = {k: 0 for k in audit.DEADLINE_COUNTS}
        for at, resolution in self.history:
            if math.floor(now-self.origin)-60 < math.floor(at-self.origin) <= math.floor(now-self.origin):
                for key in audit.RESOLUTIONS[resolution]:
                    counts[key] += 1
        pending = sum(not x['tracked'] and not x['done'] for x in self.items.values() if x['epoch'] == self.epoch)
        return dict(bucket_second=math.floor(now-self.origin), activity_origin_time_seconds=self.origin,
                    deadline_untracked_pending=pending, counts=counts, values=[None, None, None])

    def event(self, kind, rid, now, **extra):
        e = dict(event=kind, feature_method=audit.METHOD, worker='worker', request_id=rid,
                 calibration_epoch=self.epoch, diagnostic_time_seconds=now, **extra)
        c = self.context(now)
        p = dict(context=c, poll_started_time_seconds=now) if kind.startswith('feature_context_') else dict(activity=c, counts=c['counts'])
        e['feature_json'] = json.dumps(p)
        self.events.append(e)
        return e

    def admit(self, rid='a', start=0, now=.1, tracked=True, supported=True, rejected=False, register=True):
        self.rows.append(dict(opc_request_id=rid, first_text_late=None, client_ttft_seconds=None))
        self.event('feature_admission_decision',rid,now,decision='actual_rejected' if rejected else 'admitted',outcome_tracking_supported=supported)
        self.events.append(dict(event='calibration_forecast',worker='worker',request_id=rid,
                                calibration_epoch=self.epoch,diagnostic_time_seconds=now+.01))
        if register:
            self.items[rid] = dict(start=start, tracked=tracked, done=False, epoch=self.epoch)
            self.event('feature_deadline_started',rid,now+.02,request_started_time_seconds=start,
                       deadline_time_seconds=start+300,deadline_seconds=300,deadline_tracking_supported=tracked,
                       pending_deadline_count=sum(x['tracked'] and not x['done'] for x in self.items.values() if x['epoch']==self.epoch))
        return self

    def resolve(self,rid='a',observed=301,now=301,source='poll'):
        item=self.items[rid];item['done']=True
        elapsed=observed-item['start']
        resolution=('deadline_breach' if elapsed>=300 else 'timely_first_text' if source=='first_text'
                    else 'actual_cutoff' if source in ('upstream_timeout','pretext_downstream_disconnect') else 'unknown_pretext_end')
        self.history.append((now,resolution))
        self.event('feature_deadline_resolution',rid,now,resolution=resolution,resolution_source=source,
                   request_started_time_seconds=item['start'],deadline_time_seconds=item['start']+300,
                   deadline_seconds=300,observed_time_seconds=observed,elapsed_seconds=elapsed,
                   native_cancellation_acknowledgement='unavailable')
        return self

    def terminal(self,rid='a',now=301,outcome='first_text'):
        self.event('feature_outcome',rid,now,outcome=outcome,native_cancellation_acknowledgement='unavailable')
        return self

    def poll(self,now=301,started=None):
        e=self.event('feature_context_poll','',now)
        p=json.loads(e['feature_json']);p['poll_started_time_seconds']=now if started is None else started
        e['feature_json']=json.dumps(p)
        return self

    def reset(self,now=302):
        self.epoch+=1;self.origin=now;self.history=[]
        self.events.append(dict(event='calibration_reset',worker='worker',calibration_epoch=self.epoch,reason='queue_drained'))
        return self

    def ignored_terminal(self,rid='a',now=303,reason='generation_changed'):
        self.events.append(dict(event='feature_outcome_ignored',feature_method=audit.METHOD,worker='worker',request_id=rid,
            admission_epoch=1,calibration_epoch=self.epoch,diagnostic_time_seconds=now,outcome='first_text',reason=reason))
        return self

    def audit(self):
        return audit.audit_deadlines(self.events,self.rows)


def change_context(event, change):
    p=json.loads(event['feature_json']);c=p.get('activity',p.get('context'));change(c)
    if 'counts' in p:p['counts']=c['counts']
    event['feature_json']=json.dumps(p)


class DeadlineAuditTests(unittest.TestCase):
    def assert_bad(self,trace,reason=None):
        receipt=trace.audit();self.assertFalse(receipt['ok'],receipt)
        if reason:self.assertIn(reason,receipt['errors'][0]['reason'])

    def test_poll_once_then_late_text_and_real_cutoff_are_separate(self):
        t=Trace().admit().resolve().terminal(now=420).poll(now=421)
        result=t.audit();self.assertTrue(result['ok'],result)
        self.assertEqual(result['counts']['deadline_breach'],1)
        self.assertEqual(result['counts']['first_text_after_poll_breach'],1)
        self.assertEqual(result['poll_resolution_lag_seconds']['maximum'],1)
        # Client future labels never supply the recorded union classification.
        t.rows[0].update(first_text_late=False,client_ttft_seconds=1)
        self.assertTrue(t.audit()['ok'])
        t=Trace().admit().resolve().terminal(now=320,outcome='pretext_downstream_disconnect')
        self.assertTrue(t.audit()['ok'])
        t.resolve(now=420,observed=420,source='first_text')
        self.assert_bad(t,'duplicate union')

    def test_timely_claim_can_publish_after_deadline(self):
        t=Trace().admit().poll(now=300.1,started=300).resolve(observed=299.9,now=300.2,source='first_text').terminal(now=300.2)
        result=t.audit();self.assertTrue(result['ok'],result)
        self.assertEqual(result['counts']['timely_first_text'],1)
        self.assertEqual(result['counts']['poll_due_pending_with_later_terminal_observation'],1)

    def test_poll_start_is_conservative_not_later_publication_instant(self):
        t=Trace().admit().poll(now=300.01,started=299.99).resolve(now=301).terminal(now=302)
        self.assertTrue(t.audit()['ok'])

    def test_multiple_due_items_batch_has_no_resolution_pending_count(self):
        t=Trace().admit('a').admit('b',now=.2).resolve('b').resolve('a').poll(now=301.1)
        t.terminal('a',now=302).terminal('b',now=302)
        self.assertTrue(t.audit()['ok'])

    def test_early_poll_and_missing_resolution_fail(self):
        t=Trace().admit().resolve(now=299,observed=299).terminal(now=300)
        self.assert_bad(t,'Early poll')
        t=Trace().admit().resolve(source='first_text',observed=10,now=10).terminal(now=10)
        t.events=[e for e in t.events if e['event']!='feature_deadline_resolution']
        self.assert_bad(t,'once-only resolution')

    def test_metadata_only_scan_claim_can_defer_breach_to_a_later_poll(self):
        # A scan claim is not emitted; it can consume only metadata at t301,
        # release, then the next poll records the true overdue deadline.
        t=Trace().admit().poll(now=301).resolve(now=302,observed=302).terminal(now=303)
        result=t.audit();self.assertTrue(result['ok'],result)
        self.assertEqual(result['counts']['poll_due_pending_without_observed_terminal_unverified'],1)
        self.assertEqual(result['counts']['deadline_breach'],1)
        self.assertEqual(result['poll_resolution_lag_seconds']['maximum'],2)
        missing=Trace().admit().poll(now=301)
        self.assert_bad(missing,'Missing terminal')

    def test_diagnostic_not_observed_time_buckets_union(self):
        t=Trace().admit().resolve(observed=5,now=65.1,source='upstream_timeout').terminal(now=65.1,outcome='upstream_timeout').poll(now=70)
        self.assertTrue(t.audit()['ok'])
        e=next(e for e in t.events if e['event']=='feature_deadline_resolution')
        change_context(e,lambda c:c['counts'].update(deadline_or_cutoff_adverse=0))
        self.assert_bad(t,'union window mismatch')

    def test_signed_request_start_and_clock_mismatches(self):
        t=Trace().admit(start=-20).resolve(now=281,observed=281).terminal(now=282)
        self.assertTrue(t.audit()['ok'])
        for field,value in [('deadline_time_seconds',301),('elapsed_seconds',999),('observed_time_seconds',999),('deadline_seconds',301)]:
            changed=copy.deepcopy(t);e=next(e for e in changed.events if e['event']=='feature_deadline_resolution');e[field]=value
            self.assert_bad(changed)

    def test_reset_clears_pending_and_old_ignored_does_not_count(self):
        t=Trace().admit().reset(now=100).poll(now=101).ignored_terminal(now=102)
        result=t.audit();self.assertTrue(result['ok'],result)
        self.assertEqual(result['counts']['pending_cleared_reset'],1)
        self.assertEqual(result['counts'].get('resolved',0),0)
        # A poll-resolved request can also end in an ignored old-generation callback.
        finished=Trace().admit().resolve().reset(now=302).ignored_terminal(now=303)
        self.assertTrue(finished.audit()['ok'])
        bad=Trace().admit().reset(now=100).resolve(observed=301,now=301).terminal(now=302)
        self.assert_bad(bad,'reset generation')

    def test_registration_ignored_requires_selected_identity(self):
        t=Trace().admit(register=False).reset(now=1)
        t.events.append(dict(event='feature_deadline_ignored',feature_method=audit.METHOD,worker='worker',request_id='a',admission_epoch=1,calibration_epoch=2,reason='generation_changed_or_worker_missing',diagnostic_time_seconds=2))
        t.ignored_terminal(now=3)
        self.assertTrue(t.audit()['ok'])
        bad=copy.deepcopy(t);bad.events=[e for e in bad.events if e['event']!='calibration_forecast']
        self.assert_bad(bad,'selected calibration forecast')

    def test_selected_epoch_not_generic_preselection_epoch(self):
        t=Trace().admit().resolve(observed=2,now=2,source='first_text').terminal(now=2)
        t.events.insert(0,dict(event='forecast',request_id='a',worker='worker',calibration_epoch=0))
        self.assertTrue(t.audit()['ok'])
        next(e for e in t.events if e['event']=='calibration_forecast')['worker']='other'
        self.assert_bad(t,'worker mismatch')

    def test_untracked_registration_suppression_and_terminal_drain(self):
        t=Trace().admit(tracked=False).poll(now=301).resolve(now=302,observed=302,source='first_text').terminal(now=302)
        result=t.audit();self.assertTrue(result['ok'],result)
        self.assertEqual(result['counts']['untracked_started'],1)
        self.assertEqual(result['counts']['peak_untracked_pending'],1)
        bad=copy.deepcopy(t);e=next(e for e in bad.events if e['event']=='feature_deadline_started')
        change_context(e,lambda c:c.update(deadline_untracked_pending=0))
        self.assert_bad(bad,'Untracked pending ledger mismatch')
        bad=copy.deepcopy(t);e=next(e for e in bad.events if e['event']=='feature_context_poll')
        change_context(e,lambda c:c['values'].__setitem__(1,.5))
        self.assert_bad(bad,'suppress union rate')
        bad=Trace().admit(tracked=False).resolve().terminal()
        self.assert_bad(bad,'Poll cannot resolve untracked')

    def test_rejected_unsupported_and_noidentity_scope(self):
        for opts in [dict(rejected=True,register=False),dict(supported=False,register=False)]:
            self.assertTrue(Trace().admit(**opts).audit()['ok'])
        self.assert_bad(Trace().admit(rejected=True),'Rejected/unsupported')
        self.assert_bad(Trace().admit(supported=False),'Rejected/unsupported')
        bad=Trace().admit().resolve().terminal();next(e for e in bad.events if e['event']=='feature_deadline_started')['request_id']=''
        self.assert_bad(bad,'Unowned/no-identity')

    def test_missing_started_pending_count_duplicate_and_generation_guard(self):
        bad=Trace().admit().resolve().terminal();next(e for e in bad.events if e['event']=='feature_deadline_started')['pending_deadline_count']=2
        self.assert_bad(bad,'Tracked pending count')
        bad=Trace().admit().resolve().terminal();bad.events.insert(3,copy.deepcopy(bad.events[2]))
        self.assert_bad(bad,'Duplicate deadline registration')
        bad=Trace().admit(register=False)
        self.assert_bad(bad,'Missing deadline registration')
        bad=Trace().admit().resolve().terminal();next(e for e in bad.events if e['event']=='feature_deadline_resolution')['calibration_epoch']=2
        self.assert_bad(bad,'Epoch changed')

    def test_legacy_not_applicable_unknown_and_mixed_fail(self):
        legacy=[dict(event='feature_context_forecast',feature_method=audit.LEGACY_METHOD)]
        result=audit.audit_deadlines(legacy,[]);self.assertTrue(result['ok']);self.assertFalse(result['applicable'])
        self.assertFalse(audit.audit_deadlines([dict(event='feature_context_poll',feature_method='future')],[])['ok'])
        self.assertFalse(audit.audit_deadlines(legacy+[dict(event='feature_context_poll',feature_method=audit.METHOD)],[])['ok'])


if __name__=='__main__':unittest.main()
