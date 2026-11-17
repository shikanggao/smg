"""CPU observations/noninterference; exact c9 methods via SGLANG_NATIVE_SOURCE."""
import ast
from contextlib import contextmanager
from enum import Enum, auto
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS, MethodType
import unittest
from unittest.mock import patch

from grpc_servicer.tests import test_sglang_engine_work_progress as existing

receipt, installer = existing.receipt, existing.installer
SOURCE = Path(os.environ.get("SGLANG_NATIVE_SOURCE", "/nonexistent")) / "python/sglang/srt"


class Request(NS):
    __hash__ = object.__hash__
    __eq__ = object.__eq__


def component(source, name, env):
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
    if not nodes:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "<exact-c9-method>", "exec"), env)
    return env[name]


class ReceiptTests(unittest.TestCase):
    def new(self):
        return receipt.SchedulerDecisionReceipt("engine", "sglang-c9e5799:profile", 0)

    def test_choice_stop_counters_sum_epoch_and_exception(self):
        r=self.new()
        for name in ("EXTEND","DECODE","IDLE","MIXED"):
            r.begin();r.stop("waiting_queue_exhausted");r.finish(NS(forward_mode=NS(name=name)))
        snap=r.snapshot()
        self.assertEqual(snap['decision_seq'],4);self.assertEqual(sum(snap['choice_counts'].values()),4)
        self.assertEqual(sum(snap['prefill_stop_counts'].values()),4)
        self.assertEqual(snap['choice_counts'],dict(prefill=1,decode=1,idle=1,unsupported=1))
        r.begin();r.raised();snap=r.snapshot();self.assertFalse(snap['available']);self.assertEqual(snap['scheduler_exceptions'],1)
        self.assertEqual(sum(snap['choice_counts'].values()),snap['decision_seq'])
        fresh=receipt.SchedulerDecisionReceipt('new','sglang-c9e5799:new',0).snapshot()
        self.assertEqual(fresh['decision_seq'],0);self.assertEqual(sum(fresh['choice_counts'].values()),0)

    def test_identity_huge_numbers_bound_and_overflow(self):
        for bad in ('x'*1000000,'bad\nidentity','\ud800'):
            r=receipt.SchedulerDecisionReceipt(bad,bad,0);s=r.snapshot();self.assertFalse(s['available']);self.assertIsNone(s['engine_epoch'])
            self.assertLessEqual(len(json.dumps(s).encode()),receipt.MAX_BYTES)
        self.assertFalse(receipt._number(10**10000));self.assertFalse(receipt._number(True))
        r=self.new();r.decision_seq=receipt.MAX_EXACT;r.choice_counts['idle']=r.decision_seq;r.stop_counts['not_evaluated']=r.decision_seq
        r.begin();r.finish(None);s=r.snapshot();self.assertFalse(s['counters_available']);self.assertEqual(s['decision_seq'],receipt.MAX_EXACT)
        self.assertEqual(sum(s['choice_counts'].values()),s['decision_seq'])

    def test_observer_clock_failure_atomicity_and_recovery(self):
        r=self.new();owner=NS(_smg_scheduler_receipt=r)
        with patch.object(receipt.time,'monotonic',return_value=1.):r.begin();r.finish(None)
        r.begin()
        with patch.object(receipt.time,'monotonic',side_effect=RuntimeError('diagnostic clock')):
            with self.assertRaises(RuntimeError):r.finish(None)
        receipt.observer_failed(owner,abandon=True);s=r.snapshot(3.)
        self.assertFalse(s['available']);self.assertEqual(s['decision_seq'],sum(s['choice_counts'].values()));self.assertEqual(s['decision_seq'],1)
        self.assertEqual(s['diagnostic_error_count'],1);self.assertFalse(s['decision_in_progress'])
        with patch.object(receipt.time,'monotonic',return_value=4.):r.begin();r.finish(None)
        s=r.snapshot(5.);self.assertTrue(s['available']);self.assertEqual(s['diagnostic_error_count'],1);self.assertEqual(s['decision_seq'],2)

    def test_failure_gap_attempt_fields_disabled_binding_and_link_clocks(self):
        r=self.new();owner=NS(_smg_scheduler_receipt=r);r.begin();r.adder_begin();r.values('allocator_property_return',rem_total_tokens=5.)
        r.adder_begin();self.assertNotIn('rem_total_tokens',r.pending['evaluated_values'])
        receipt.note(owner,bad_unrecognized_field=7);self.assertEqual(r.diagnostic_errors,1)
        with patch.object(receipt.time,'monotonic',return_value=10.):r.finish(NS(forward_mode=NS(name='EXTEND')))
        self.assertFalse(r.snapshot(11.)['available'])
        r=self.new();batch=NS(forward_mode=NS(name='EXTEND'))
        with patch.object(receipt.time,'monotonic',return_value=10.):r.begin();r.finish(batch)
        r.link_submission(batch,NS(sequence=1,scheduled_at=9.));self.assertFalse(r.snapshot(11.)['available'])
        r=self.new()
        with patch.object(receipt.time,'monotonic',return_value=10.):r.begin();r.finish(batch)
        r.link_submission(batch,NS(sequence=2,scheduled_at=10.5));self.assertEqual(r.snapshot(11.)['batch_sequence'],2)
        self.assertFalse(r.snapshot(10.25)['available'])
        adder=NS(rem_input_tokens=3,rem_chunk_tokens=None);receipt.bind_adder(NS(),adder);self.assertNotIn('_smg_scheduler_receipt',vars(adder))


class NativeIntegrationTests(unittest.TestCase):
    setUp=existing.EngineWorkComponentTests.setUp
    tearDown=existing.EngineWorkComponentTests.tearDown
    def test_original_choice_identity_exception_and_observer_failures(self):
        native_error=ValueError('original scheduler error')
        def choose(engine):
            engine.events.append('choose')
            if getattr(engine,'native_error',None):raise engine.native_error
            return engine.selected
        engine=existing.scheduler_class(choose=choose)();self.schedulers.append(engine)
        engine.selected=NS(forward_mode=NS(name='DECODE'))
        self.assertIs(engine.get_next_batch_to_run(),engine.selected);self.assertEqual(engine.events,['choose'])
        r=engine._smg_scheduler_receipt
        for method in ('begin','finish'):
            with patch.object(r,method,side_effect=RuntimeError('observer')):
                self.assertIs(engine.get_next_batch_to_run(),engine.selected)
        engine.native_error=native_error
        with self.assertRaises(ValueError) as caught:engine.get_next_batch_to_run()
        self.assertIs(caught.exception,native_error)
        self.assertEqual(r.exceptions,1);self.assertFalse(r.snapshot()['available'])
        with patch.object(r,'snapshot',side_effect=RuntimeError('observer')):
            snap=engine._smg_deadline.snapshot()
        self.assertFalse(snap['scheduler_decision_receipt']['available'])
        self.assertTrue(snap['engine_work_progress']['counters_available'])


@unittest.skipUnless((SOURCE/'managers/scheduler.py').exists(), 'requires exact SGLANG_NATIVE_SOURCE')
class PinnedMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import hashlib
        cls.scheduler=(SOURCE/'managers/scheduler.py').read_text();cls.policy=(SOURCE/'managers/schedule_policy.py').read_text()
        assert hashlib.sha256(cls.scheduler.encode()).hexdigest()==installer.FILES['managers/scheduler.py']
        assert hashlib.sha256(cls.policy.encode()).hexdigest()==installer.UNCHANGED_FILES['managers/schedule_policy.py']
        cls.patched=installer.scheduler_receipt_sites(cls.scheduler)

    def adder(self, calls, wrapped):
        class Alloc:
            def available_size(self):calls.append('allocator.available');return 10
        class Cache:
            disable=False
            def evictable_size(self):calls.append('cache.evictable');return 0
            def inc_lock_ref(self,node):calls.append('cache.lock')
            def dec_lock_ref(self,node):calls.append('cache.unlock')
        env=dict(Enum=Enum,auto=auto,contextmanager=contextmanager,SWATokenToKVPoolAllocator=type('SWA',(),{}),MambaRadixCache=type('Mamba',(),{}),CLIP_MAX_NEW_TOKENS=4096,IGNORE_EOS_RESERVE_TOKENS=1)
        component(self.policy,'AddReqResult',env);cls=component(self.policy,'PrefillAdder',env)
        if wrapped:receipt.install_adder(cls)
        return cls,Alloc(),Cache()

    def test_getter_short_circuit_has_no_extra_allocator_queries(self):
        results=[]
        for wrapped in (False,True):
            calls=[];cls,alloc,cache=self.adder(calls,wrapped);alloc.available_size=lambda:(calls.append('allocator.available') or 0)
            a=cls(1,cache,alloc,NS(reqs=[]),0.,100,None)
            r=receipt.SchedulerDecisionReceipt('e','p',0);r.begin();receipt.bind_adder(NS(_smg_scheduler_receipt=r),a)
            result=a.budget_state();results.append((result.name,calls))
            if wrapped:
                self.assertIn('rem_total_tokens',r.pending['evaluated_values']);self.assertNotIn('cur_rem_tokens',r.pending['evaluated_values'])
        self.assertEqual(results[0],results[1]);self.assertEqual(results[0],('NO_TOKEN',['allocator.available','cache.evictable']))

    def test_pinned_prefill_stop_sites_slot_and_adder_call_parity(self):
        for case in ('full','empty','slots','adder','other','success'):
            results=[]
            for wrapped in (False,True):
                calls=[];Adder,alloc,cache=self.adder(calls,wrapped)
                if case in ('other','success'):alloc.available_size=lambda:(calls.append('allocator.available') or 1000)
                created=[]
                def init_batch(reqs,*args,**kwargs):
                    value=NS(reqs=reqs,return_logprob=False,forward_mode=NS(name='EXTEND'))
                    value.prepare_for_extend=lambda:calls.append('prepare.extend')
                    created.append(value);return value
                env=dict(PrefillAdder=Adder,AddReqResult=Adder.add_one_req.__wrapped__.__globals__["AddReqResult"] if wrapped else Adder.add_one_req.__globals__["AddReqResult"],TEST_RETRACT=False,TEST_RETRACT_NO_PREFILL_BS=0,DisaggregationMode=NS(PREFILL='prefill'),
                    get_global_server_args=lambda:NS(pp_max_micro_batch_size=0 if case=='slots' else 100),ScheduleBatch=NS(init_new=init_batch),
                    _smg_sched_note=receipt.note,_smg_bind_adder=receipt.bind_adder,_smg_slots_value=receipt.slots_value,_smg_prefetch_skip=receipt.prefetch_skip)
                text=self.patched if wrapped else self.scheduler;prefill=component(text,'get_new_batch_prefill',env);slots=component(text,'get_num_allocatable_reqs',env)
                class ShortCircuitQueue:
                    def __len__(self):raise AssertionError('short circuit changed')
                req=Request(sampling_params=NS(ignore_eos=False,max_new_tokens=5),extend_input_len=20,host_hit_length=0,prefix_indices=[],output_ids=[],fill_ids=list(range(20)),last_node=object(),time_stats=NS(forward_entry_time=1),init_next_round_input=lambda cache:calls.append('req.init'))
                running=NS(reqs=[],batch_is_full=case=='full')
                s=NS(grammar_queue=[],try_preemption=False,running_batch=running,waiting_queue=ShortCircuitQueue() if case=='full' else [] if case=='empty' else [req],chunked_req=None,
                    pp_size=1,req_to_token_pool=NS(),enable_hierarchical_cache=False,enable_lora=False,enable_hicache_storage=False,disaggregation_mode='null',
                    truncation_align_size=None,is_mixed_chunk=False,page_size=1,tree_cache=cache,token_to_kv_pool_allocator=alloc,new_token_ratio=0.,max_prefill_tokens=100,chunked_prefill_size=0 if case=='other' else None,priority_scheduling_preemption_threshold=0,enable_metrics=False,enable_overlap=True,model_config=NS(),spec_algorithm=NS(),current_scheduler_metrics_enabled=lambda:False,
                    policy=NS(calc_priority=lambda queue:calls.append('priority')))
                def counted_slots(engine,bs):calls.append('slots');return slots(engine,bs)
                s.get_num_allocatable_reqs=MethodType(counted_slots,s)
                r=receipt.SchedulerDecisionReceipt('e','p',0);r.begin()
                if wrapped:s._smg_scheduler_receipt=r
                result=prefill(s)
                if case=='success':self.assertIs(result,created[0]);self.assertEqual(result.reqs,[req])
                results.append((None if result is None else [x.extend_input_len for x in result.reqs],calls,running.batch_is_full,[] if case=='full' else [x.extend_input_len for x in s.waiting_queue]))
                if wrapped:
                    expected={'full':'full_or_empty_no_chunk','empty':'full_or_empty_no_chunk','slots':'no_request_slots','adder':'adder_no_token','other':'adder_other','success':'waiting_queue_exhausted'}[case]
                    self.assertEqual(r.pending['prefill_stop_code'],expected)
            self.assertEqual(results[0],results[1],case)

    def test_actual_adder_cache_lock_finally_preserves_exception(self):
        marker=RuntimeError('cache lock native failure');results=[]
        for wrapped in (False,True):
            calls=[];Adder,alloc,cache=self.adder(calls,wrapped)
            alloc.available_size=lambda:(calls.append('allocator.available') or 1000)
            def fail(node):calls.append('cache.lock');raise marker
            cache.inc_lock_ref=fail
            a=Adder(1,cache,alloc,NS(reqs=[]),0.,100,None)
            r=receipt.SchedulerDecisionReceipt('e','p',0);r.begin();receipt.bind_adder(NS(_smg_scheduler_receipt=r),a)
            req=Request(sampling_params=NS(ignore_eos=False,max_new_tokens=5),extend_input_len=20,host_hit_length=0,prefix_indices=[],output_ids=[],last_node=object())
            with self.assertRaises(RuntimeError) as caught:a.add_one_req(req,False,None)
            self.assertIs(caught.exception,marker);results.append((calls,a.rem_input_tokens,a.can_run_list,req.extend_input_len))
        self.assertEqual(results[0],results[1]);self.assertEqual(results[0][0].count('cache.lock'),1);self.assertEqual(results[0][0].count('cache.unlock'),1)

    def test_source_hook_drift_refuses_before_preparation(self):
        with self.assertRaises(ValueError):installer.scheduler_receipt_sites(self.scheduler.replace('if len(can_run_list) == 0:', 'if not can_run_list:'))

if __name__=='__main__':unittest.main(verbosity=2)
