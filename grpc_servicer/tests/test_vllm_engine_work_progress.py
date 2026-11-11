"""CPU-only hooks at exact pinned scheduler advance and EngineCore future paths."""
import ast
from contextlib import nullcontext
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

from smg_grpc_servicer import native_deadline_vllm as native


def source_method(root, module, cls_name, method_name):
    source = root.joinpath(*module.split(".")).with_suffix(".py")
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != native.SOURCE_HASHES[module]:
        raise AssertionError("source drift from pinned official engine")
    tree = ast.parse(data)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace[method_name]


def request(start=0, prompt=100, output=0):
    return NS(request_id="r", num_computed_tokens=start, num_prompt_tokens=prompt,
        num_output_tokens=output, num_cached_tokens=start, num_preemptions=0,
        num_tokens=prompt + output, num_output_placeholders=0, use_structured_output=False,
        has_encoder_inputs=False, mm_features=[], prompt_embeds=None, pooling_params=None,
        is_finished=lambda:False, status=0, sampling_params=None)


@unittest.skipUnless(os.environ.get("SMG_VLLM_019_SOURCE"), "requires exact pinned source")
class VllmCompletedWorkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = native.NativeStore(Path(self.temp.name)/"native.db")
        root = Path(os.environ["SMG_VLLM_019_SOURCE"])
        advance = source_method(root,"vllm.v1.core.sched.scheduler","Scheduler","_update_after_schedule")
        step = source_method(root,"vllm.v1.engine.core","EngineCore","step")
        class Scheduler:
            def __init__(s):
                s.parallel_config=NS(data_parallel_rank=0)
                s.requests={"r":request()};s.running=list(s.requests.values())
                s.waiting=[];s.skipped_waiting=[];s.finished_req_ids=set()
                s.n=20;s.fail_result=False
            def schedule(s):
                out=NS(num_scheduled_tokens={"r":s.n},finished_req_ids=set(),
                    scheduled_spec_decode_tokens={},scheduled_encoder_inputs={},
                    has_structured_output_requests=False,total_num_scheduled_tokens=s.n)
                advance(s,out)
                return out
            def update_from_output(s,scheduler_output,model_runner_output):
                if s.fail_result:
                    raise RuntimeError("native processing failure")
                return {"native_return":"unchanged"}
            def add_request(s,r):s.requests[r.request_id]=r
            def finish_requests(s,*a):pass
            def _free_blocks(s,*a):pass
            def has_requests(s):return True
            def get_grammar_bitmask(s,out):return None
        self.env=patch.dict(os.environ,{"SMG_NATIVE_ENGINE_WORK_DIAGNOSTICS":"1"})
        self.env.start();self.addCleanup(self.env.stop)
        native.install_scheduler_hooks(Scheduler,4,lambda:self.store)
        self.scheduler=Scheduler()
        self.step=native._wrap_engine_step(step)

    def progress(self):
        return self.scheduler._smg_signals.work_progress

    def model_output(self, ids=("r",)):
        return NS(req_id_to_index={r:i for i,r in enumerate(ids)},kv_connector_output=None)

    def engine(self, callback):
        class Future:
            def result(s):return callback()
        return NS(scheduler=self.scheduler,
            model_executor=NS(execute_model=lambda *a,**kw:Future()),
            log_error_detail=lambda *a:nullcontext(),log_iteration_details=lambda *a:nullcontext(),
            _process_aborts_queue=lambda:None)

    def test_actual_scheduler_advance_is_not_completion_async_ranges_frozen(self):
        s=self.scheduler;s.requests["r"].num_computed_tokens=40
        first=s.schedule();second=s.schedule()
        self.assertEqual(s.requests["r"].num_computed_tokens,80)
        self.assertEqual(self.progress().cumulative["completed_uncached_prefill_tokens"],0)
        s.requests["r"].num_computed_tokens=9999
        s.update_from_output(first,self.model_output())
        s.update_from_output(first,self.model_output())
        s.update_from_output(second,self.model_output())
        self.assertEqual(self.progress().cumulative["completed_uncached_prefill_tokens"],40)
        self.assertEqual(self.progress().cumulative["completed_batches"],2)
        self.assertFalse(s._smg_signals.work_batches)

    def test_actual_core_future_completion_can_credit_canceled_inflight_work(self):
        def done():
            self.assertEqual(self.progress().cumulative["completed_batches"],0)
            self.scheduler.requests.clear();self.scheduler.running.clear()
            return self.model_output()
        out,executed=self.step(self.engine(done))
        self.assertEqual(out,{"native_return":"unchanged"})
        self.assertTrue(executed)
        self.assertEqual(self.progress().cumulative["completed_uncached_prefill_tokens"],20)

    def test_canceled_pruned_output_has_no_completion_credit(self):
        self.step(self.engine(lambda:self.model_output(ids=())))
        self.assertFalse(self.progress().available)
        self.assertEqual(self.progress().cumulative["completed_batches"],0)

    def test_model_future_failure_and_native_processing_failure_propagate(self):
        def failed():raise RuntimeError("future failed")
        with self.assertRaisesRegex(RuntimeError,"future failed"):
            self.step(self.engine(failed))
        self.assertFalse(self.progress().available)
        self.assertEqual(self.progress().cumulative["completed_batches"],0)
        self.scheduler.fail_result=True
        with self.assertRaisesRegex(RuntimeError,"native processing failure"):
            self.step(self.engine(self.model_output))
        self.assertEqual(self.progress().cumulative["completed_batches"],0)

    def test_no_text_generation_context_remains_unknown_native_work(self):
        s=self.scheduler;s.requests["r"]=request(100,100,1);s.running=[s.requests["r"]];s.n=1
        self.step(self.engine(self.model_output))
        self.assertEqual(self.progress().cumulative["completed_generation_context_tokens"],1)
        self.assertEqual(self.progress().cumulative["completed_generation_context_attention_tokens"],101)
        native_row=self.store.loads(100)["loads"][0]["deadline_recovery"]["active_requests"][0]
        self.assertIsNone(native_row["remaining_prefill_work_tokens"])

    def test_reexecution_and_cache_input_ranges_count_only_on_distinct_completion(self):
        s=self.scheduler;s.requests["r"].num_computed_tokens=40
        self.step(self.engine(self.model_output))
        s.requests["r"].num_computed_tokens=40;s.requests["r"].num_preemptions=1
        self.step(self.engine(self.model_output))
        self.assertEqual(self.progress().cumulative["completed_uncached_prefill_tokens"],40)
        self.assertEqual(self.progress().cumulative["completed_batches"],2)

    def test_unsupported_metadata_or_bad_diagnostic_never_changes_native_success(self):
        self.scheduler.requests["r"].mm_features=["image"]
        self.assertEqual(self.step(self.engine(self.model_output))[0],{"native_return":"unchanged"})
        self.assertFalse(self.progress().available)
        self.assertEqual(self.progress().cumulative["completed_batches"],0)
        self.scheduler._smg_signals.capture_engine_work=lambda out:1/0
        self.assertEqual(self.step(self.engine(self.model_output))[0],{"native_return":"unchanged"})

    def test_off_by_default_omits_field_and_does_not_capture(self):
        with patch.dict(os.environ,{"SMG_NATIVE_ENGINE_WORK_DIAGNOSTICS":"0"}):
            signals=native.SchedulerSignals(self.scheduler,self.store,0)
        signals.publish()
        self.assertIsNone(signals.work_progress)
        envelope=self.store.loads(100)["loads"][0]["deadline_recovery"]
        self.assertNotIn("engine_work_progress",envelope)
        self.assertNotIn("engine_work_state",envelope["active_requests"][0])

    def test_snapshot_failure_does_not_break_native_result_or_inventory(self):
        self.progress().snapshot=lambda *args:1/0
        self.assertEqual(self.step(self.engine(self.model_output))[0],{"native_return":"unchanged"})
        envelope=self.store.loads(100)["loads"][0]["deadline_recovery"]
        self.assertFalse(envelope["engine_work_progress"]["counters_available"])
        self.assertEqual(envelope["native_population"],1)

    def test_request_state_flags_are_observations_without_request_mutation(self):
        s=self.scheduler;r=s.requests["r"];signals=s._smg_signals
        r.num_output_tokens=2;r.num_preemptions=1;r.status=4
        signals.retired["r"]=r
        before=dict(vars(r))
        signals.publish()
        row=self.store.loads(100)["loads"][0]["deadline_recovery"]["active_requests"][0]
        state=row["engine_work_state"]
        self.assertTrue(state["available"])
        self.assertEqual(state["native_request_id"],row["native_request_id"])
        self.assertEqual(state["attempt_id"],row["attempt_id"])
        self.assertEqual(state["output_token_count"],2)
        self.assertEqual(state["preemption_count"],1)
        self.assertIsNone(state["retraction_count"])
        self.assertTrue(state["abort_pending"])
        self.assertTrue(state["cache_release_observed"])
        self.assertEqual(vars(r),before)
        self.assertIsNone(row["remaining_prefill_work_tokens"])

    def test_request_state_unknowns_and_fault_do_not_break_native_snapshot(self):
        s=self.scheduler;signals=s._smg_signals
        del s.requests["r"].num_preemptions
        signals.publish()
        envelope=self.store.loads(100)["loads"][0]["deadline_recovery"]
        state=envelope["active_requests"][0]["engine_work_state"]
        self.assertFalse(state["available"])
        self.assertEqual(state["output_token_count"],0)
        self.assertIsNone(state["preemption_count"])
        self.assertIsNone(state["abort_pending"])
        self.assertFalse(state["cache_release_observed"])
        signals._request_engine_work_state=lambda *a:1/0
        signals.publish()
        envelope=self.store.loads(100)["loads"][0]["deadline_recovery"]
        self.assertEqual(envelope["native_population"],1)
        self.assertFalse(envelope["active_requests"][0]["engine_work_state"]["available"])


if __name__ == "__main__":
    unittest.main()
