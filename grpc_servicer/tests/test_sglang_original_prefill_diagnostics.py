"""Diagnostic-only CPU/source tests; no prediction or GPU qualification."""
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from grpc_servicer.tests import test_sglang_engine_work_progress as existing

request,batch,native,installer=existing.request,existing.batch,existing.native,existing.installer


def ordinary_request(output=None):
    req=request(output=output)
    req.origin_input_ids_unpadded=req.origin_input_ids
    req.input_embeds=None
    req.session_id=None
    req.multimodal_inputs=None
    return req


class OriginalPrefillDiagnosticsTests(unittest.TestCase):
    setUp=existing.EngineWorkComponentTests.setUp
    tearDown=existing.EngineWorkComponentTests.tearDown
    engine=existing.EngineWorkComponentTests.engine

    def attach(self,engine,req):
        req.smg_attempt_id='synthetic-observed-attempt-'+str(len(self.schedulers))
        engine._smg_deadline.store.register(req.smg_attempt_id,0,None)
        engine.running_batch.reqs=[req]

    def facts(self,engine):
        return engine._smg_deadline.snapshot()['active_requests'][0]['original_prefill_observation']

    def test_strong_raw_facts_do_not_change_legacy_remaining_inventory_or_ack(self):
        engine=self.engine();req=ordinary_request([7]);req._smg_prefill_completed=100
        self.attach(engine,req);before=dict(vars(req))
        snap=engine._smg_deadline.snapshot();row=snap['active_requests'][0];facts=row['original_prefill_observation']
        self.assertTrue(facts['qualified'])
        self.assertEqual(facts['lifetime_retraction_count'],0)
        self.assertIs(facts['is_retracted'],False)
        self.assertIsNone(row['remaining_prefill_work_tokens'])
        self.assertFalse(row['first_text_emitted'])
        self.assertEqual(snap['native_population'],1)
        self.assertEqual(snap['cancellation_acks'],[])
        self.assertEqual(vars(req),before)
        self.assertNotIn('joint_work_service_capability',snap)
        self.assertLessEqual(facts['observed_at_monotonic_secs'],snap['captured_at_monotonic_secs'])
        self.assertLessEqual(snap['captured_at_monotonic_secs'],snap['engine_work_progress']['snapshot_monotonic_secs'])

    def test_lifetime_retractions_transient_flag_and_missing_values_remain_distinct(self):
        for count,flag in ((1,False),(0,True),(True,False),(0,None)):
            engine=self.engine();req=ordinary_request([7]);req._smg_prefill_completed=100
            req.retraction_count=count;req.is_retracted=flag;self.attach(engine,req)
            facts=self.facts(engine)
            self.assertIsNot(facts['qualified'],True)
            self.assertEqual(facts['lifetime_retraction_count'],count if type(count) is int else None)
            self.assertIs(facts['is_retracted'],flag)
            engine.running_batch.reqs=[]

    def test_input_alias_session_embeds_abort_and_frontier_guards_are_observations(self):
        for kind in ('alias','session','embeds','multimodal','abort','frontier','output','missing'):
            engine=self.engine();req=ordinary_request([7]);req._smg_prefill_completed=100
            if kind=='alias':req.origin_input_ids=[0]
            elif kind=='session':req.session_id='synthetic-session'
            elif kind=='embeds':req.input_embeds=[]
            elif kind=='multimodal':req.multimodal_inputs={}
            elif kind=='abort':req.to_finish=type('FINISH_ABORT',(),{})()
            elif kind=='frontier':req._smg_prefill_completed=99
            elif kind=='output':req.output_ids=[]
            else:del req.origin_input_ids_unpadded
            self.attach(engine,req)
            self.assertIsNot(self.facts(engine)['qualified'],True,kind)
            engine.running_batch.reqs=[]

    def test_service_provenance_is_unknown_before_work_and_sticky_after_bad_contributor(self):
        engine=self.engine();state=engine._smg_deadline
        initial=state.snapshot()['original_prefill_service_provenance']
        self.assertIsNone(initial['available'])
        self.assertIsNone(initial['token_content_identity_verified'])
        req=ordinary_request();first=batch([req],[60],[20]);result=engine.run_batch(first)
        self.assertEqual(state.original_units_batches,0)
        self.assertTrue(all(isinstance(value,(int,bool,type(None))) for row in result._smg_engine_work_record._smg_original_units for value in row))
        engine.process_batch_result(first,result);engine.process_batch_result(first,result)
        valid=state.snapshot()['original_prefill_service_provenance']
        self.assertTrue(valid['available'])
        self.assertEqual(valid['observed_completed_batches'],1)
        self.assertEqual(valid['guarded_completed_batches'],1)
        self.assertEqual(valid['counted_contributor_occurrences'],1)
        req.retraction_count=1
        repeated=batch([req],[70],[60]);engine.process_batch_result(repeated,engine.run_batch(repeated))
        after=state.snapshot()
        self.assertFalse(after['original_prefill_service_provenance']['available'])
        self.assertTrue(after['engine_work_progress']['counters_available'])
        self.assertEqual(after['engine_work_progress']['cumulative']['completed_uncached_prefill_tokens'],50)
        req.retraction_count=0
        last=batch([req],[80],[70]);engine.process_batch_result(last,engine.run_batch(last))
        self.assertFalse(state.snapshot()['original_prefill_service_provenance']['available'])

    def test_origin_rewrite_between_submission_and_completion_invalidates_only_unit_provenance(self):
        engine=self.engine();req=ordinary_request();item=batch([req],[40],[0]);result=engine.run_batch(item)
        req.origin_input_ids=[0]
        engine.process_batch_result(item,result)
        snap=engine._smg_deadline.snapshot()
        self.assertFalse(snap['original_prefill_service_provenance']['available'])
        self.assertTrue(snap['engine_work_progress']['counters_available'])
        self.assertEqual(snap['engine_work_progress']['cumulative']['completed_uncached_prefill_tokens'],40)

    def test_disabled_and_faulted_diagnostics_do_not_modify_legacy_success(self):
        engine=self.engine();req=ordinary_request([7]);req._smg_prefill_completed=100;self.attach(engine,req)
        with patch.object(engine._smg_deadline,'request_original_prefill',side_effect=RuntimeError('diagnostic')):
            snap=engine._smg_deadline.snapshot()
        self.assertFalse(snap['active_requests'][0]['original_prefill_observation']['available'])
        self.assertIsNone(snap['active_requests'][0]['remaining_prefill_work_tokens'])
        self.assertEqual(snap['cancellation_acks'],[])
        with patch.dict(os.environ,{'SMG_NATIVE_ENGINE_WORK_DIAGNOSTICS':'0'}):
            off=self.engine()
        off.running_batch.reqs=[ordinary_request([7])]
        snap=off._smg_deadline.snapshot()
        self.assertNotIn('original_prefill_observation',snap['active_requests'][0])
        self.assertNotIn('original_prefill_service_provenance',snap)

    def test_new_submission_completion_observer_faults_preserve_legacy_credit_and_return(self):
        for method in ('begin_original_units','complete_original_units'):
            engine=self.engine();req=ordinary_request();item=batch([req],[40],[20])
            with patch.object(engine._smg_deadline,method,side_effect=RuntimeError('diagnostic')):
                result=engine.run_batch(item)
                self.assertEqual(engine.process_batch_result(item,result),'native-return-value')
            snapshot=engine._smg_deadline.snapshot()
            self.assertTrue(snapshot['engine_work_progress']['counters_available'])
            self.assertEqual(snapshot['engine_work_progress']['cumulative']['completed_uncached_prefill_tokens'],20)
            self.assertFalse(snapshot['original_prefill_service_provenance']['available'])

    def test_pinned_normal_return_commits_frontier_and_grammar_abort_breaks_alias(self):
        source=os.environ.get('SGLANG_NATIVE_SOURCE')
        if not source:self.skipTest('requires exact pinned c9e source')
        self.source=Path(source);installer.prepare(self.source)
        method=lambda file,cls,name,ns:existing.PinnedEngineWorkTests.method(self,file,cls,name,ns)
        stage=NS(PREFILL_CHUNKED_FORWARD=0,PREFILL_FORWARD=1,DECODE_LOOP=2)
        prefill=method('scheduler_output_processor_mixin.py','SchedulerOutputProcessorMixin','process_batch_result_prefill',{'trace_slice':lambda *a,**k:None,'RequestStage':stage})
        dispatch=method('scheduler.py','Scheduler','process_batch_result',{'trace_slice_batch':lambda *a,**k:None,'RequestStage':stage})
        engine=self.engine(dispatch);engine.process_batch_result_prefill=prefill.__get__(engine)
        req=ordinary_request();req.is_chunked=0;self.attach(engine,req)
        item=batch([req],[100],[60]);result=engine.run_batch(item)
        result.copy_done=NS(synchronize=lambda:engine.events.append('existing-wait'))
        result.logits_output=None;result.next_token_ids=NS(tolist=lambda:[7])
        result.extend_input_len_per_req=result.extend_logprob_start_len_per_req=None
        self.assertFalse(self.facts(engine)['available'])
        engine.process_batch_result(item,result)
        self.assertEqual(engine.events,['existing-wait','stream'])
        self.assertTrue(self.facts(engine)['qualified'])
        self.assertEqual(req._smg_prefill_completed,100)
        abort=method('schedule_batch.py','Req','set_finish_with_abort',{
            'get_tensor_model_parallel_rank':lambda:1,'FINISH_ABORT':lambda *a:object(),
            'HTTPStatus':NS(BAD_REQUEST=400)})
        abort(req,'synthetic-abort')
        req.to_finish=None  # Consumed pending abort is not an original-input witness.
        self.assertFalse(self.facts(engine)['original_alias_verified'])
        self.assertFalse(self.facts(engine)['qualified'])


if __name__=='__main__':unittest.main(verbosity=2)
