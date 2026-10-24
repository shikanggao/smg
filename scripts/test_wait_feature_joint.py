import copy
import json
import unittest
try:
    from . import wait_feature_joint as j
    from . import wait_feature_context as f
    from . import replay_wait_joint_gains as replay
    from .test_wait_feature_context import fixture,event
except ImportError:
    import wait_feature_joint as j
    import wait_feature_context as f
    import replay_wait_joint_gains as replay
    from test_wait_feature_context import fixture,event


def samples():
    return [dict(n=100.,ratio=1.5+.8*(i%4)/3-.4*(i//4)/7,
                 context=[(i%4)/3,(i//4)/7,None]) for i in range(32)]


class JointGainTests(unittest.TestCase):
    def test_current_test_profiles_freeze_v3_method(self):
        import json
        from pathlib import Path
        for name in ("wait-ttft-deadline300.json", "wait-ttft-observe420.json"):
            value=json.loads((Path(__file__).parent/name).read_text())
            self.assertEqual(value["expected_feature_method"],j.METHOD)

    def test_matches_independently_emitted_rust_fixture(self):
        rows=[]
        for i in range(32):
            x=[float((i >> bit) & 1) for bit in range(3)]
            rows.append(dict(n=20.+5*i,ratio=1.4+.7*x[0]-.4*x[1]+.2*x[2],context=x))
        fit=j.learn(rows,1)
        expected=[1.3495767751164383,.5520135382249178,-.23088066119625672,.20156605729104365]
        for actual,want in zip(fit['coefficients'],expected):self.assertAlmostEqual(actual,want,places=12)
        m=j.Model();m.fit=fit;m.update([.2,.7,.4],1.,0.)
        p=m.prediction([.2,.7,.4],1.,20.,True,125.)
        self.assertAlmostEqual(p['gain'],1.3789894428404597,places=12)
        self.assertAlmostEqual(sum(p['contribution_seconds']),125*p['gain'],places=12)

    def test_joint_ridge_satisfies_normal_equations_with_neutral_prior(self):
        rows=samples();fit=j.learn(rows,7)
        self.assertEqual(fit['support'],[True,True,False])
        self.assertEqual(fit['revision'],7)
        self.assertGreater(fit['coefficients'][1],0)
        self.assertLess(fit['coefficients'][2],0)
        # Independent objective derivative, not a second linear solver.
        gradient=[.1*(w-p) for w,p in zip(fit['coefficients'],[1,0,0,0])]
        for r in rows:
            x=[1,*r['context'][:2],0]
            error=sum(w*z for w,z in zip(fit['coefficients'],x))-r['ratio']
            for k in range(4):gradient[k]+=error*x[k]/len(rows)
        for value in gradient:self.assertAlmostEqual(value,0,places=12)

    def test_missing_constant_or_insufficient_features_are_not_trained(self):
        for rows in [samples()[:11],[dict(n=100,ratio=2,context=[.2,0,0]) for _ in range(32)],
                     [dict(n=100,ratio=2,context=None) for _ in range(32)]]:
            self.assertEqual(j.learn(rows)['coefficients'],[1,0,0,0])
            self.assertFalse(any(j.learn(rows)['support']))

    def test_small_effective_sample_size_blocks_fit(self):
        rows=samples();rows[0]['n']=1e8
        self.assertFalse(any(j.learn(rows)['support']))

    def test_full_joint_subset_required_and_unknown_is_not_zero(self):
        m=j.Model();m.fit=j.learn(samples());m.update([.5,.5,None],1,10)
        self.assertTrue(m.prediction([.5,.5,None],1,11,True,100)['contextual'])
        for context in ([.5,None,None],[.5,1.1,None]):
            p=m.prediction(context,1.4,11,True,100)
            self.assertFalse(p['contextual']);self.assertEqual(p['gain'],1.4)
        m.update([.5,None,None],1.4,12)
        self.assertIsNone(m.trajectory)

    def test_l1_movement_bounds_combined_effect_not_four_times_rate(self):
        t=([1,0,0,0],[3,2,-2,1],10)
        now=j.applied(t,10.4)
        self.assertAlmostEqual(sum(abs(a-b) for a,b in zip(now,t[0])),.2)
        for context in ([0,0,0],[1,1,1],[.5,.7,.2]):
            before=j.gain(t[0],context);after=j.gain(now,context)
            self.assertLessEqual(abs(after-before),.2+1e-12)

    def test_signed_contributions_sum_after_positive_and_negative_caps(self):
        for coefficients in ([2,2,-.3,1],[.1,-2,.3,-1]):
            eff=j.effective(coefficients,[1,1,1])
            self.assertLessEqual(abs(sum(eff[1:])),.5+1e-12)
            self.assertGreaterEqual(sum(eff),-1e-12)
            self.assertAlmostEqual(sum(200*x for x in eff),200*j.gain(eff,[1,1,1]))

    def test_changed_support_reanchors_at_scalar(self):
        m=j.Model();m.fit=j.learn(samples());m.update([.5,.5,None],1,1)
        m.fit=j.learn([dict(r,context=[r['context'][0],None,None]) for r in samples()])
        m.update([.5,None,None],1.7,2)
        self.assertEqual(m.trajectory[0],[1.7,0,0,0])

    def test_v3_forecast_rejects_changed_weights_or_per_feature_contributions(self):
        value,gate=fixture();m=j.Model();m.fit=j.learn(samples());m.update([.5,0,0],2,10)
        value.update(method=j.METHOD,fit=m.fit,coefficient_order=["queue","kv","deadline_or_cutoff","rejection"],
                     neutral_prior=[1,0,0,0],coefficient_movement_l1_per_second=.5)
        value['context'].update(deadline_untracked_pending=0)
        value['context']['counts'].update(deadline_timely_first_text=16,deadline_breach=0,
            deadline_or_cutoff_adverse=0,deadline_or_cutoff_unknown=0)
        value['prediction']=m.prediction([.5,0,0],2,12,True,100)
        value.update(extra_seconds=100*value['prediction']['gain'],
                     total_ttft_seconds=1+100*value['prediction']['gain'])
        e=event(value,gate);e['feature_method']=j.METHOD
        self.assertEqual(f.forecast(e,gate=gate)['method'],j.METHOD)
        for key in ('applied_coefficients','effective_coefficients','contribution_seconds','target_coefficients'):
            bad=copy.deepcopy(value);bad['prediction'][key][1]+=.1
            e=event(bad,gate);e['feature_method']=j.METHOD
            with self.assertRaises(ValueError):f.forecast(e,gate=gate)

    def test_plot_preserves_unknown_gaps_and_actual_lower_bounds(self):
        try:
            from .plot_wait_joint_gains import series
        except ImportError:
            from plot_wait_joint_gains import series
        import math
        values=series([dict(p=1.),dict(p=None),dict(p=3.)],'p')
        self.assertTrue(math.isnan(values[1]));self.assertEqual(len(values),3)
        row=dict(client_ttft_seconds=None,cancellation_reason='observation_horizon',
            first_text_observation_horizon_seconds=420,started_elapsed_seconds=1.,cancellation_delivered_elapsed_seconds=311.)
        self.assertIsNone(replay.censor_bound(row,True))
        row['cancellation_delivered_elapsed_seconds']=421.1
        self.assertEqual(replay.censor_bound(row,True),420)

    def test_exact_metrics_do_not_fit_or_score_censor_bound_as_numeric(self):
        rows=[dict(phase='pressure',truth=True,actual_ttft_seconds=None,joint_v3=200),
              dict(phase='recovery',truth=False,actual_ttft_seconds=100,joint_v3=400),
              dict(phase='recovery',truth=None,actual_ttft_seconds=None,joint_v3=None)]
        score=replay.score(rows,'joint_v3')
        self.assertEqual((score['fn'],score['fp'],score['unknown']),(1,1,1))
        self.assertEqual(score['exact_n'],1);self.assertEqual(score['mae_seconds'],300)
        self.assertEqual(score['censored_known_late'],1)


class CurrentReplayTests(unittest.TestCase):
    def events(self):
        worker='w'; events=[]; model=j.Model(); profile=1
        def context(now,kv=.5):
            return dict(values=[kv,None,None],bucket_second=int(now),activity_origin_time_seconds=0.,
                deadline_untracked_pending=0,counts={name:0 for name in f.COUNTS+f.DEADLINE_COUNTS})
        def poll(now):
            events.append(dict(event='calibration_poll',worker=worker,calibration_epoch=0,
                outcome='success',diagnostic_time_seconds=now,poll_started_time_seconds=now,
                scale=1.,target_scale=1.))
            model.update([.5,None,None],1.,now)
            snapshot(now,'feature_context_poll')
        def snapshot(now,kind,rid='probe',kv=.5):
            value=dict(method=j.METHOD,native_ready=True,fresh=True,supported_path=True,
                profile=profile,context=context(now,kv),legacy_gain=1.,
                original_queue_excess_seconds=100.,history_revision=sum(e['event']=='feature_context_feedback' for e in events),
                poll_started_time_seconds=now if kind=='feature_context_poll' else 20.,
                fit=model.fit,prediction=model.prediction([kv,None,None],1.,now,True,100.))
            events.append(dict(event=kind,worker=worker,calibration_epoch=0,
                feature_method=j.METHOD,request_id=rid if kind=='feature_context_forecast' else '',
                diagnostic_time_seconds=now,feature_json=json.dumps(value)))
        poll(0.)
        data=[dict(r,context=[r['context'][0],None,None]) for r in samples()]
        for i,row in enumerate(data):
            rid='request-'+str(i);at=1.+i*.3;now=at+.2
            snapshot(at-.001,'feature_context_forecast',rid,row['context'][0])
            events.append(dict(event='calibration_forecast',worker=worker,request_id=rid,
                calibration_epoch=0,diagnostic_time_seconds=at,calibration_input_excess_seconds=100.,
                calibration_baseline_available=True,calibration_baseline_seconds=1.))
            value=dict(reason='accepted',ttft_seconds=1.+100*row['ratio'],queue_excess_seconds=100.,
                baseline_seconds=1.,admission_epoch=0,admission_time_seconds=at,
                ratio=row['ratio'],saved_context=context(at,row['context'][0]),admission_profile=profile,
                current_profile=profile,feature_context_accepted=True,history_revision=i+1)
            events.append(dict(event='first_text',request_id=rid,ttft_seconds=value['ttft_seconds']))
            events.append(dict(event='feature_context_feedback',worker=worker,request_id=rid,
                calibration_epoch=0,diagnostic_time_seconds=now,feature_method=j.METHOD,feature_json=json.dumps(value)))
            # Scalar diagnostic is published after the feature diagnostic, same observed callback time.
            events.append(dict(event='calibration_feedback',worker=worker,request_id=rid,
                diagnostic_time_seconds=now,calibration_time_seconds=now,reason='accepted',ttft_seconds=value['ttft_seconds'],
                sample_count_after=i+1,admission_native_capacity_ready=True))
        model.fit=j.learn(data,32);poll(20.);snapshot(21.,'feature_context_forecast')
        return events

    def test_ordered_feedback_retention_reconstructs_joint_fit(self):
        result=replay.replay_current(self.events())
        self.assertTrue(result['ok']);self.assertEqual(result['counts']['accepted_feedback'],32)
        self.assertEqual(result['counts']['fits'],1)
        self.assertTrue(result['rows'][-1]['prediction']['contextual'])

    def test_mutated_fit_and_mismatched_feedback_clock_fail_closed(self):
        events=self.events();value=json.loads(events[-1]['feature_json'])
        value['fit']['coefficients'][0]+=.01;events[-1]['feature_json']=json.dumps(value)
        with self.assertRaisesRegex(ValueError,'joint fitted model'):replay.replay_current(events)
        events=self.events()
        next(e for e in events if e['event']=='calibration_feedback')['diagnostic_time_seconds']+=1
        with self.assertRaisesRegex(ValueError,'publication time'):replay.replay_current(events)

    def test_mutated_saved_feature_profile_and_numeric_censored_label_fail(self):
        for mutation in ('kv','rates','profile','current_profile','first_text'):
            events=self.events()
            event=next(e for e in events if e['event']=='feature_context_feedback')
            value=json.loads(event['feature_json'])
            if mutation=='kv':value['saved_context']['values'][0]=.1
            elif mutation=='rates':value['saved_context']['counts']['actual_rejected']=1
            elif mutation=='profile':value['admission_profile']=9
            elif mutation=='current_profile':value['current_profile']=9
            else:events=[e for e in events if not(e['event']=='first_text' and e['request_id']==event['request_id'])]
            event['feature_json']=json.dumps(value)
            with self.assertRaises(ValueError,msg=mutation):replay.replay_current(events)

    def test_early_rotation_cannot_validate_its_own_truncated_fit(self):
        events=self.events()
        rotation=dict(event='calibration_rotation',worker='w',calibration_epoch=0,calibration_state='capacity_ready',
            calibration_bank_seconds=300,calibration_bank_index=1,calibration_time_seconds=301.199,
            sample_count_before=32,sample_count_after=32,calibration_history_revision=32,
            calibration_previous_sample_count=32,calibration_current_sample_count=0)
        with self.assertRaisesRegex(ValueError,'clock-derived'):replay.replay_current(events+[rotation])
        rotation['calibration_time_seconds']=301.2
        self.assertTrue(replay.replay_current(events+[rotation])['ok'])
        rotation['calibration_previous_sample_count']=31
        with self.assertRaisesRegex(ValueError,'previous count'):replay.replay_current(events+[rotation])

    def test_unavailable_diagnostic_prediction_uses_explicit_scalar_fallback(self):
        paired=[dict(opc_request_id='r',phase='recovery',first_text_late=False,client_ttft_seconds=100.,
            started_elapsed_seconds=1.,pre_admission_total_ttft_predictions=dict(calibrated=200.),
            feature_context_candidate=dict(total_ttft_seconds=400.,prediction_available=False))]
        candidate=[dict(event='feature_context_forecast',request_id='r',total_ttft_seconds=400.,prediction_available=False,
            prediction={},feature_values=[],original_queue_excess_seconds=200.)]
        rows,cards=replay.compare(paired,candidate)
        self.assertEqual(rows[0]['joint_v3'],200.)
        self.assertEqual(rows[0]['feature_v2'],200.)
        self.assertEqual(cards['all']['joint_v3']['fp'],0)


if __name__=='__main__':unittest.main()
