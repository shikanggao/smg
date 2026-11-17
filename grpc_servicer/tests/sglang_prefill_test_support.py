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


class NativePrefillFixture(unittest.TestCase):
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


class PinnedNativePrefillFixture(NativePrefillFixture):
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
