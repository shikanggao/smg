"""CPU metadata and exact pinned-method tests; no engine/GPU qualification.

SGLANG_NATIVE_SOURCE=/path/to/exact/c9e5799 checkout enables source-bound tests.
"""

import ast
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


native = load("sglang_work_native_test", ROOT / "native_patches/sglang/native_deadline.py")
installer = load("sglang_work_installer_test", ROOT / "native_patches/sglang/apply_patch.py")
helper = load(
    "sglang.srt.managers.smg_engine_work_progress",
    ROOT / "smg_grpc_servicer/engine_work_progress.py",
)


receipt = load(
    "sglang.srt.managers.smg_scheduler_receipt",
    ROOT / "native_patches/sglang/scheduler_receipt.py",
)


class Mode:
    def __init__(self, value):
        self.value = value

    def is_extend(self):
        return self.value in ("extend", "mixed")

    def is_decode(self):
        return self.value == "decode"

    def is_idle(self):
        return self.value == "idle"

    def is_prebuilt(self):
        return self.value == "prebuilt"


def request(name="req", prompt=100, output=None):
    req = SimpleNamespace(
        rid=name, origin_input_ids=list(range(prompt)), output_ids=output or [],
        retraction_count=0, is_retracted=False, is_chunked=1, to_finish=None,
        finished_reason=None, smg_attempt_id=None, req_pool_idx=None, mamba_pool_idx=None,
        return_hidden_states=False, grammar=None, extend_input_len=0, fill_ids=[],
    )
    req.finished = lambda: req.finished_reason is not None
    req.check_finished = lambda: None
    return req


def batch(reqs, ends, prefixes=None, mode="extend", decoding=None):
    # These fields are exactly those read by pinned ScheduleBatch.copy.
    value = SimpleNamespace(**{key: None for key in (
        "req_to_token_pool", "req_pool_indices", "model_config", "out_cache_loc",
        "spec_algorithm", "global_num_tokens", "global_num_tokens_for_logprob",
        "can_run_dp_cuda_graph", "is_extend_in_batch", "is_prefill_only",
    )})
    value.reqs = list(reqs)
    value.seq_lens_cpu = list(ends)
    value.prefix_lens = list(prefixes or [])
    value.decoding_reqs = decoding
    value.forward_mode = Mode(mode)
    value.return_logprob = False
    value.enable_overlap = True
    return value


def scheduler_class(process_result=None, choose=None):
    class Scheduler:
        def __init__(self):
            self.server_args = SimpleNamespace(tp_size=1, pp_size=1, dp_size=1)
            self.dp_rank = 0
            self.disaggregation_mode = "null"
            self.enable_pdmux = False
            self.is_generation = True
            self.model_config = SimpleNamespace(is_encoder_decoder=False, is_multimodal=False)
            self.waiting_queue = []
            self.grammar_queue = []
            self.chunked_req = self.cur_batch = self.last_batch = None
            self.running_batch = SimpleNamespace(reqs=[])
            self.result_queue = []
            self.events = []
            self.stream_output = lambda *args: self.events.append("stream")
            self.maybe_send_health_check_signal = lambda: None
            self.tree_cache = SimpleNamespace(cache_unfinished_req=lambda req: None)

        def handle_generate_request(self, req):
            self.waiting_queue.append(req)

        def abort_request(self, req):
            return None

        def process_input_requests(self, reqs):
            return None

        def get_load(self, recv_req=None):
            return SimpleNamespace()

        def run_batch(self, batch):
            return SimpleNamespace()

        def process_batch_result(self, batch, result):
            if getattr(result, "fail", False):
                raise RuntimeError("native result processing failed")
            self.events.append("native-result-return")
            return "native-return-value"

    if process_result is not None:
        Scheduler.process_batch_result = process_result
    if choose is not None:
        Scheduler.get_next_batch_to_run = choose
    native.install_scheduler(Scheduler)
    return Scheduler


class EngineWorkComponentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            "SMG_NATIVE_DEADLINE_SIGNALS": "1",
            "SMG_NATIVE_ENGINE_WORK_DIAGNOSTICS": "1",
            "SMG_NATIVE_DEADLINE_DB": str(Path(self.temp.name) / "deadline.db"),
        })
        self.env.start()
        self.schedulers = []

    def tearDown(self):
        for engine in self.schedulers:
            engine._smg_deadline.store.db.close()
        self.env.stop()
        self.temp.cleanup()

    def engine(self, process_result=None):
        engine = scheduler_class(process_result)()
        self.schedulers.append(engine)
        return engine

    def counters(self, engine):
        return engine._smg_deadline.snapshot()["engine_work_progress"]

    def test_off_by_default_does_not_import_helper_or_change_deadline_envelope(self):
        with patch.dict(os.environ, {"SMG_NATIVE_ENGINE_WORK_DIAGNOSTICS": "0"}), patch.dict(
            sys.modules, {"sglang.srt.managers.smg_engine_work_progress": None}
        ):
            engine = self.engine()
        self.assertIsNone(engine._smg_deadline.engine_work)
        self.assertNotIn("engine_work_progress", engine._smg_deadline.snapshot())
        engine.waiting_queue = [request()]
        self.assertNotIn("engine_work_state", engine._smg_deadline.snapshot()["active_requests"][0])

    def test_request_state_records_counts_abort_and_release_without_changing_req(self):
        engine = self.engine()
        req = request(output=[7, 8])
        req.smg_attempt_id = "observed-attempt"
        engine._smg_deadline.store.register(req.smg_attempt_id, 0, None)
        req.retraction_count = 2
        req.to_finish = type("FINISH_ABORT", (), {})()
        engine.running_batch.reqs = [req]
        before = dict(vars(req))
        row = engine._smg_deadline.snapshot()["active_requests"][0]
        state = row["engine_work_state"]
        self.assertEqual(vars(req), before)
        self.assertEqual(state, dict(
            schema_version=1, diagnostic_only=True, native_request_id=req.rid,
            attempt_id=req.smg_attempt_id, available=True, unavailable_reason=None,
            output_token_count=2, retraction_count=2, preemption_count=None,
            abort_pending=True, abort_pending_source="req_to_finish_FINISH_ABORT",
            cache_release_observed=False,
        ))
        self.assertIsNone(row["remaining_prefill_work_tokens"])
        self.assertFalse(row["first_text_emitted"])
        req._smg_cache_released = True
        self.assertTrue(engine._smg_deadline.snapshot()["active_requests"][0]["engine_work_state"]["cache_release_observed"])
        self.assertEqual(engine._smg_deadline.snapshot()["cancellation_acks"], [])

    def test_invalid_request_state_metadata_is_null_and_unavailable(self):
        engine = self.engine()
        req = request()
        req.output_ids = ()
        req.retraction_count = -1
        req._smg_cache_released = "invalid"
        del req.to_finish
        state = engine._smg_deadline.request_engine_work_state(req)
        self.assertFalse(state["available"])
        for field in ("output_token_count", "retraction_count", "preemption_count", "abort_pending", "cache_release_observed"):
            self.assertIsNone(state[field])

    def test_request_state_scope_and_guard_failure_leave_native_inventory_intact(self):
        engine = self.engine()
        req = request(output=[7])
        engine.running_batch.reqs = [req]
        engine._smg_deadline.engine_work_supported = False
        state = engine._smg_deadline.snapshot()["active_requests"][0]["engine_work_state"]
        self.assertFalse(state["available"])
        self.assertEqual(state["unavailable_reason"], "unsupported_engine_configuration")
        for field in ("output_token_count", "retraction_count", "preemption_count", "abort_pending", "cache_release_observed"):
            self.assertIsNone(state[field])
        engine._smg_deadline.engine_work_supported = True
        with patch.object(engine._smg_deadline, "request_engine_work_state", side_effect=RuntimeError("diagnostic failure")):
            snapshot = engine._smg_deadline.snapshot()
        self.assertTrue(snapshot["active_inventory_complete"])
        self.assertEqual(snapshot["native_population"], 1)
        self.assertIsNone(snapshot["active_requests"][0]["remaining_prefill_work_tokens"])
        state = snapshot["active_requests"][0]["engine_work_state"]
        self.assertEqual(state["native_request_id"], req.rid)
        self.assertFalse(state["available"])
        self.assertEqual(state["unavailable_reason"], "request_work_metadata_observation_failed")

    def test_schedule_chunk_cache_overlap_and_duplicate_completion(self):
        engine = self.engine()
        req = request()
        first = batch([req], [60], [20])
        result = engine.run_batch(first)
        self.assertEqual(self.counters(engine)["cumulative"]["completed_batches"], 0)
        first.prefix_lens[0] = 55
        req.extend_input_len = 999
        self.assertEqual(engine.process_batch_result(first, result), "native-return-value")
        engine.process_batch_result(first, result)
        second = batch([req], [100], [60])
        engine.process_batch_result(second, engine.run_batch(second))
        values = self.counters(engine)
        self.assertEqual(values["cumulative"]["completed_batches"], 2)
        self.assertEqual(values["cumulative"]["completed_uncached_prefill_tokens"], 80)
        self.assertEqual(values["cumulative"]["completed_generation_context_tokens"], 0)
        self.assertEqual(values["completion_sequence"], 2)
        self.assertEqual(values["engine_epoch"], engine._smg_deadline.epoch)
        self.assertEqual(values["profile_id"], engine._smg_deadline.profile_id)

    def test_retraction_reexecution_is_new_work_not_duplicate_credit(self):
        engine = self.engine()
        req = request()
        first = batch([req], [100], [20])
        engine.process_batch_result(first, engine.run_batch(first))
        req.retraction_count = 1
        recompute = batch([req], [103], [50])
        engine.process_batch_result(recompute, engine.run_batch(recompute))
        values = self.counters(engine)["cumulative"]
        self.assertEqual(values["completed_uncached_prefill_tokens"], 130)
        self.assertEqual(values["completed_generation_context_tokens"], 3)
        self.assertEqual(values["completed_generation_context_attention_tokens"], 101+102+103)

    def test_mixed_decode_and_no_text_work_preserve_unknown_remaining_guard(self):
        engine = self.engine()
        prefill, decode = request("prefill"), request("decode", output=[1, 2])
        decode.smg_attempt_id = "hidden-text"
        engine._smg_deadline.store.register("hidden-text", 0, None)
        engine.waiting_queue = [decode]
        mixed = batch([prefill, decode], [80, 103], [40, 102], "mixed", [decode])
        engine.process_batch_result(mixed, engine.run_batch(mixed))
        values = self.counters(engine)["cumulative"]
        self.assertEqual(values["completed_uncached_prefill_tokens"], 40)
        self.assertEqual(values["completed_generation_context_tokens"], 1)
        self.assertEqual(values["completed_generation_context_attention_tokens"], 103)
        self.assertIsNone(engine._smg_deadline.snapshot()["active_requests"][0]["remaining_prefill_work_tokens"])

    def test_cancelled_inflight_work_requires_successful_result_witness(self):
        engine = self.engine()
        req = request()
        launched = batch([req], [40], [10])
        result = engine.run_batch(launched)
        req.finished_reason = object()
        req.to_finish = object()
        self.assertEqual(self.counters(engine)["cumulative"]["completed_uncached_prefill_tokens"], 0)
        result.fail = True
        with self.assertRaisesRegex(RuntimeError, "native result"):
            engine.process_batch_result(launched, result)
        self.assertEqual(self.counters(engine)["cumulative"]["completed_uncached_prefill_tokens"], 0)
        self.assertFalse(self.counters(engine)["counters_available"])
        result.fail = False
        engine.process_batch_result(launched, result)
        self.assertEqual(self.counters(engine)["cumulative"]["completed_uncached_prefill_tokens"], 30)

    def test_bad_metadata_fails_unavailable_without_changing_native_result(self):
        engine = self.engine()
        malformed = batch([request()], [20.5], [0])
        result = engine.run_batch(malformed)
        self.assertEqual(engine.process_batch_result(malformed, result), "native-return-value")
        values = self.counters(engine)
        self.assertFalse(values["counters_available"])
        self.assertEqual(values["cumulative"]["completed_batches"], 0)

    def test_ambiguous_model_or_execution_scope_reports_unavailable(self):
        for setting in ("encoder_decoder", "multimodal", "embedding", "rank", "speculative"):
            with self.subTest(setting=setting):
                engine = self.engine()
                if setting == "encoder_decoder":
                    engine.model_config.is_encoder_decoder = True
                elif setting == "multimodal":
                    engine.model_config.is_multimodal = True
                elif setting == "embedding":
                    engine.is_generation = False
                elif setting == "rank":
                    engine.dp_rank = 1
                else:
                    engine.server_args.speculative_algorithm = "EAGLE"
                engine._smg_deadline = native.SchedulerDeadlineState(
                    engine, store=engine._smg_deadline.store
                )
                self.assertIsNone(engine._smg_deadline.begin_engine_work(batch([request()], [60], [0])))
                values = self.counters(engine)
                self.assertFalse(values["counters_available"])
                self.assertEqual(values["cumulative"]["completed_batches"], 0)

    def test_diagnostic_failure_does_not_change_native_result_or_load(self):
        engine = self.engine()
        launched = batch([request()], [40], [10])
        with patch.object(engine._smg_deadline.engine_work, "begin", side_effect=RuntimeError("diagnostic failure")):
            result = engine.run_batch(launched)
        self.assertEqual(engine.process_batch_result(launched, result), "native-return-value")
        self.assertFalse(self.counters(engine)["counters_available"])
        with patch.object(engine._smg_deadline.engine_work, "snapshot", side_effect=RuntimeError("diagnostic failure")):
            envelope = engine.get_load().deadline_recovery
        self.assertFalse(envelope["engine_work_progress"]["counters_available"])
        self.assertTrue(envelope["active_inventory_complete"])

    def test_device_metadata_is_rejected_without_reading_device_values(self):
        class DeviceTensor:
            device = SimpleNamespace(type="cuda")

            def __iter__(self):
                raise AssertionError("Observer must not read device values")

        engine = self.engine()
        launched = batch([request()], [40], [10])
        launched.seq_lens_cpu = DeviceTensor()
        self.assertIsNone(engine._smg_deadline.begin_engine_work(launched))
        self.assertFalse(self.counters(engine)["counters_available"])


class PinnedEngineWorkTests(EngineWorkComponentTests):
    """Execute real pinned CPU method bodies with native device objects doubled."""

    def setUp(self):
        source = os.environ.get("SGLANG_NATIVE_SOURCE")
        if not source:
            self.skipTest("Set SGLANG_NATIVE_SOURCE for exact pinned method tests")
        super().setUp()
        self.source = Path(source)
        installer.prepare(self.source)  # All twelve upstream hashes must match.

    def method(self, filename, cls_name, method_name, globals):
        tree = ast.parse((self.source / "python/sglang/srt/managers" / filename).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
        # Strip evaluation of annotations, leaving the pinned executable body intact.
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
        method.decorator_list = []
        values = dict(globals)
        exec(compile(ast.Module(body=[method], type_ignores=[]), filename, "exec"), values)
        return values[method_name]

    def test_exact_overlap_copy_uses_result_record_and_native_completion_wait(self):
        copy_batch = self.method("schedule_batch.py", "ScheduleBatch", "copy", {"ScheduleBatch":SimpleNamespace})
        stage = SimpleNamespace(PREFILL_CHUNKED_FORWARD=0, PREFILL_FORWARD=1, DECODE_LOOP=2)
        prefill = self.method("scheduler_output_processor_mixin.py", "SchedulerOutputProcessorMixin", "process_batch_result_prefill", {"trace_slice":lambda *a,**k:None,"RequestStage":stage})
        dispatch = self.method("scheduler.py", "Scheduler", "process_batch_result", {"trace_slice_batch":lambda *a,**k:None,"RequestStage":stage})
        engine = self.engine(dispatch)
        engine.process_batch_result_prefill = prefill.__get__(engine)
        req = request()
        launched = batch([req], [60], [20])
        result = engine.run_batch(launched)
        result.copy_done = SimpleNamespace(synchronize=lambda: engine.events.append("existing-completion-wait"))
        result.logits_output = None
        result.next_token_ids = SimpleNamespace(tolist=lambda:[7])
        result.extend_input_len_per_req = result.extend_logprob_start_len_per_req = None
        queued = copy_batch(launched)
        self.assertFalse(hasattr(queued, "prefix_lens"))
        launched.seq_lens_cpu = [99]
        launched.prefix_lens[0] = 55
        req.fill_ids = list(range(999))
        self.assertEqual(self.counters(engine)["cumulative"]["completed_batches"], 0)
        engine.process_batch_result(queued, result)
        self.assertEqual(engine.events, ["existing-completion-wait", "stream"])
        self.assertEqual(self.counters(engine)["cumulative"]["completed_uncached_prefill_tokens"], 40)
        self.assertEqual(req._smg_prefill_completed, 60)

    def test_exact_native_check_finished_consumes_pending_abort_only(self):
        check_finished = self.method("schedule_batch.py", "Req", "check_finished", {})
        engine = self.engine()
        req = request(output=[7, 8])
        req.retraction_count = 1
        abort = type("FINISH_ABORT", (), {})()
        req.to_finish = abort
        engine.running_batch.reqs = [req]
        before = engine._smg_deadline.snapshot()["active_requests"][0]
        self.assertEqual(before["state"], "decode")
        self.assertTrue(before["engine_work_state"]["abort_pending"])
        check_finished(req)
        after = engine._smg_deadline.snapshot()["active_requests"][0]
        self.assertEqual(after["state"], "unknown")
        self.assertFalse(after["engine_work_state"]["abort_pending"])
        self.assertEqual(after["engine_work_state"]["output_token_count"], 2)
        self.assertEqual(after["engine_work_state"]["retraction_count"], 1)
        self.assertIsNone(after["remaining_prefill_work_tokens"])
        self.assertIs(req.finished_reason, abort)


if __name__ == "__main__":
    unittest.main()
