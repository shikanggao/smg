"""CPU component tests and optional exact-upstream method tests.

SGLANG_NATIVE_SOURCE=/path/to/pinned/checkout python -m unittest discover \
  -s grpc_servicer/tests -p test_sglang_native_deadline.py -v

The method tests compile the real pinned upstream AST without importing CUDA.
Request/cache/scheduler context doubles are component fixtures, not a GPU engine.
"""

import ast
import asyncio
import importlib.util
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "native_patches/sglang"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


native = load("sglang_native_deadline_test", ROOT / "native_deadline.py")
installer = load("sglang_native_installer_test", ROOT / "apply_patch.py")


def args(**kwargs):
    return SimpleNamespace(
        tp_size=1,
        pp_size=1,
        dp_size=1,
        enable_pdmux=False,
        disaggregation_mode="null",
        disaggregation_decode_enable_offload_kvcache=False,
        **kwargs,
    )


def req(rid, output=None, pool=None):
    value = SimpleNamespace(
        rid=rid,
        origin_input_ids=list(range(100)),
        output_ids=output or [],
        req_pool_idx=pool,
        mamba_pool_idx=None,
        grammar=None,
        finished_reason=None,
        to_finish=None,
    )
    value.finished = lambda: value.finished_reason is not None
    return value


def scheduler(*requests):
    return SimpleNamespace(
        server_args=args(),
        dp_rank=0,
        disaggregation_mode="null",
        enable_pdmux=False,
        waiting_queue=list(requests),
        grammar_queue=[],
        chunked_req=None,
        running_batch=SimpleNamespace(reqs=[]),
        cur_batch=None,
        last_batch=None,
        result_queue=[],
    )


class DeadlineComponentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "deadline.db")
        self.env = patch.dict(
            os.environ,
            {
                "SMG_NATIVE_DEADLINE_SIGNALS": "1",
                "SMG_NATIVE_HARD_TTFT_ENFORCEMENT": "1",
                "SMG_NATIVE_DEADLINE_DB": self.path,
            },
        )
        self.env.start()
        self.store = native.TextDeadlineStore()

    def tearDown(self):
        self.store.db.close()
        self.env.stop()
        self.tmp.cleanup()

    def state(self, sched, limit=4096):
        return native.SchedulerDeadlineState(sched, journal_limit=limit, store=self.store)

    def test_first_text_wins_atomic_race_and_never_expires(self):
        self.store.register("opaque-A", 100, 300)
        peer = native.TextDeadlineStore(self.path)
        self.assertTrue(self.store.observe_text("opaque-A", 299))
        self.assertFalse(peer.claim_expired("opaque-A", 301))
        self.assertTrue(peer.first_text("opaque-A"))
        peer.db.close()

    def test_deadline_wins_atomic_race_and_late_text_is_refused(self):
        self.store.register("opaque-A", 100, 300)
        peer = native.TextDeadlineStore(self.path)
        self.assertTrue(peer.claim_expired("opaque-A", 300))
        self.assertFalse(self.store.observe_text("opaque-A", 301))
        self.assertFalse(peer.first_text("opaque-A"))
        peer.db.close()

    def test_text_observation_after_deadline_cannot_win_before_scheduler_tick(self):
        self.store.register("A", 100, 300)
        self.assertFalse(self.store.observe_text("A", 300))
        self.assertTrue(self.store.claim_expired("A", 301))

    def test_lock_wait_cannot_backdate_first_text_before_deadline(self):
        import time
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event

        ready, go = Event(), Event()

        def observe_after_signal():
            peer = native.TextDeadlineStore(self.path)
            ready.set()
            go.wait(timeout=2)
            try:
                return peer.observe_text("locked")
            finally:
                peer.db.close()

        with ThreadPoolExecutor(1) as pool:
            result = pool.submit(observe_after_signal)
            self.assertTrue(ready.wait(timeout=2))
            now = time.monotonic()
            self.store.register("locked", now, now + 0.05)
            self.store.db.execute("BEGIN IMMEDIATE")
            go.set()
            time.sleep(0.1)
            self.store.db.execute("COMMIT")
            self.assertFalse(result.result(timeout=2))

    def test_attempt_retention_is_bounded_and_invalid_reservations_pruned(self):
        with patch.object(native, "MAX_ATTEMPTS", 1):
            self.store.register("invalid", 100, None)
            with self.assertRaises(ValueError):
                self.store.register("second", 101, None)
            self.store.discard_reservation("invalid")
            self.store.register("accepted", 102, None)
            self.store.activate("accepted")
            self.store.discard_reservation("accepted")
            with self.assertRaises(ValueError):
                self.store.register("third", 103, None)

    def test_wrong_id_never_inherits_other_attempt_deadline(self):
        self.store.register("A", 100, 300)
        self.assertFalse(self.store.claim_expired("a", 400))
        with self.assertRaises(ValueError):
            self.store.observe_text("a", 200)
        self.assertTrue(self.store.claim_expired("A", 400))

    def test_duplicate_attempt_rejected_even_after_retirement(self):
        self.store.register("A", 100, 300)
        self.store.discard("A")
        with self.assertRaises(ValueError):
            self.store.register("A", 400, 500)

    def test_mixed_inventory_deduplicates_all_scheduler_owners(self):
        waiting, grammar, prefill, decode, inflight = [req(str(i)) for i in range(5)]
        decode.output_ids = [7]
        sched = scheduler(waiting)
        sched.grammar_queue = [grammar]
        sched.chunked_req = prefill
        sched.running_batch.reqs = [decode]
        sched.cur_batch = SimpleNamespace(reqs=[prefill, decode])
        sched.last_batch = SimpleNamespace(reqs=[decode])
        sched.result_queue = [(SimpleNamespace(reqs=[inflight, prefill]), None)]
        prefill._smg_matched_prefix = 10
        prefill._smg_prefill_completed = 40
        snap = self.state(sched).snapshot()
        self.assertEqual(snap["native_population"], 5)
        self.assertTrue(snap["active_inventory_complete"])
        self.assertEqual(len(snap["active_requests"]), 5)
        rows = {row["native_request_id"]: row for row in snap["active_requests"]}
        self.assertIsNone(rows["0"]["remaining_prefill_work_tokens"])
        self.assertEqual(rows["2"]["remaining_prefill_work_tokens"], 60)
        self.assertIsNone(rows["3"]["remaining_prefill_work_tokens"])
        self.assertEqual(snap["remaining_work_source"], "unknown")
        self.assertIsNone(snap["effective_capacity_tokens_per_sec"])
        self.assertIsNone(snap["abort_max_lag_seconds"])
        self.assertEqual(snap["scheduler_semantics"], "continuous_batching")

    def test_ack_requires_native_release_and_all_inflight_references_gone(self):
        active = req("native-id", pool=2)
        active.smg_attempt_id = "opaque-A"
        self.store.register("opaque-A", 0, None)
        sched = scheduler()
        sched.running_batch.reqs = [active]
        sched.result_queue = [(SimpleNamespace(reqs=[active]), None)]
        state = self.state(sched)
        state.reconcile()
        state.mark_abort(SimpleNamespace(rid="native-id", abort_all=False, smg_exact=True))
        self.assertEqual(state.snapshot()["cancellation_acks"], [])
        active.finished_reason = "abort"
        sched.running_batch.reqs = []
        self.assertEqual(state.snapshot()["cancellation_acks"], [])
        active._smg_cache_released = True
        retained = state.snapshot()
        self.assertEqual(retained["native_population"], 1)
        self.assertEqual(retained["active_requests"][0]["state"], "unknown")
        self.assertEqual(retained["cancellation_acks"], [])
        sched.result_queue = []
        snap = state.snapshot()
        self.assertEqual(len(snap["cancellation_acks"]), 1)
        ack = snap["cancellation_acks"][0]
        self.assertEqual(ack["attempt_id"], "opaque-A")
        self.assertEqual(ack["native_request_id"], "native-id")
        self.assertLessEqual(ack["event_seq"], snap["snapshot_seq"])
        self.assertEqual(state.snapshot()["cancellation_acks"], [ack])

    def test_retracted_decoder_remaining_work_is_unknown(self):
        active = req("retracted", output=[1, 2])
        active.retraction_count = 1
        active._smg_prefill_completed = 100
        sched = scheduler(active)
        snapshot = self.state(sched).snapshot()
        self.assertIsNone(snapshot["active_requests"][0]["remaining_prefill_work_tokens"])
        self.assertEqual(snapshot["remaining_work_source"], "unknown")

    def test_native_removal_without_abort_never_fabricates_ack(self):
        active = req("A")
        active.smg_attempt_id = "attempt"
        sched = scheduler(active)
        state = self.state(sched)
        state.reconcile()
        sched.waiting_queue = []
        self.assertEqual(state.snapshot()["cancellation_acks"], [])

    def test_exact_attempt_metadata_survives_native_request_id(self):
        active = req("engine-id")
        sched = scheduler(active)
        state = self.state(sched)
        state.remember(
            SimpleNamespace(rid="engine-id", smg_attempt_id="Case:/opaque%ID", smg_deadline=123.0)
        )
        row = state.snapshot()["active_requests"][0]
        self.assertEqual(row["attempt_id"], "Case:/opaque%ID")
        self.assertEqual(row["native_request_id"], "engine-id")
        self.assertEqual(row["hard_ttft_deadline_monotonic_secs"], 123)

    def test_bounded_ack_journal_floor_and_restart_epoch(self):
        sched = scheduler()
        state = self.state(sched, limit=1)
        for name in ("A", "B"):
            active = req(name)
            active.smg_attempt_id = name
            sched.waiting_queue = [active]
            state.reconcile()
            state.mark_abort(SimpleNamespace(rid=name, abort_all=False, smg_exact=True))
            sched.waiting_queue = []
            state.reconcile()
        snap = state.snapshot()
        self.assertEqual(len(snap["cancellation_acks"]), 1)
        self.assertGreater(snap["ack_floor_seq"], 0)
        restarted = self.state(scheduler()).snapshot()
        self.assertNotEqual(snap["engine_epoch"], restarted["engine_epoch"])
        self.assertEqual(restarted["cancellation_acks"], [])
        self.assertEqual(restarted["ack_floor_seq"], 0)

    def test_unsupported_topology_is_explicit_and_default_enforcement_off(self):
        sched = scheduler(req("A"))
        sched.server_args.tp_size = 2
        snap = self.state(sched).snapshot()
        self.assertFalse(snap["active_inventory_complete"])
        self.assertFalse(snap["cancellation_ack_supported"])
        self.assertFalse(snap["hard_ttft_deadline_supported"])
        self.assertFalse(snap["abort_enforcement_enabled"])
        with patch.dict(os.environ, {"SMG_NATIVE_HARD_TTFT_ENFORCEMENT": "0"}):
            self.assertFalse(self.state(scheduler()).snapshot()["abort_enforcement_enabled"])

    def test_profile_changes_on_scheduler_config(self):
        sched = scheduler()
        before = self.state(sched).profile_id
        sched.server_args.chunked_prefill_size = 1024
        self.assertNotEqual(before, self.state(sched).profile_id)

    def test_sse_parser_ignores_roles_empty_tokens_and_handles_split_text(self):
        observer = native.SSETextObserver()
        self.assertFalse(
            observer.feed(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\r\n\r\n')
        )
        self.assertFalse(observer.feed(b'data: {"choices":[{"delta":{"content":""}}]}\n\n'))
        self.assertFalse(observer.feed(b'data: {"choices":[{"delta":{"reasoning_content":"hel'))
        self.assertTrue(observer.feed(b'lo"}}]}\n\n'))
        self.assertFalse(observer.feed(b"data: [DONE]\n\n"))

    def test_release_observer_marks_only_after_success_and_disabled_identity(self):
        active = req("A")

        def fail(value, cache):
            raise RuntimeError("cache cleanup failed")

        with self.assertRaises(RuntimeError):
            native.release_observer(fail)(active, None)
        self.assertFalse(getattr(active, "_smg_cache_released", False))
        native.release_observer(lambda value, cache: None)(active, None)
        self.assertTrue(active._smg_cache_released)
        with patch.dict(os.environ, {"SMG_NATIVE_DEADLINE_SIGNALS": "0"}):
            self.assertIs(native.release_observer(fail), fail)

    def test_disabled_hooks_do_not_change_methods_or_register_routes(self):
        class App:
            def add_middleware(self, *args, **kwargs):
                self.fail("Disabled signals must not add middleware")

            def add_api_route(self, *args, **kwargs):
                self.fail("Disabled signals must not register routes")

        class Scheduler:
            pass

        class Tokenizer:
            pass

        with patch.dict(os.environ, {"SMG_NATIVE_DEADLINE_SIGNALS": "0"}):
            before_scheduler = dict(vars(Scheduler))
            before_tokenizer = dict(vars(Tokenizer))
            native.install_scheduler(Scheduler)
            native.install_tokenizer(Tokenizer)
            native.install_http(App(), lambda: None)
            self.assertEqual(before_scheduler, dict(vars(Scheduler)))
            self.assertEqual(before_tokenizer, dict(vars(Tokenizer)))

    def test_http_late_text_suppressed_and_receive_origin_recorded(self):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send(
                {
                    "type": "http.response.body",
                    "body": b'data: {"choices":[{"delta":{"content":"late"}}]}\n\n',
                    "more_body": False,
                }
            )

        middleware = native.DeadlineHTTPMiddleware(app, lambda: SimpleNamespace(server_args=args()))
        messages = []

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "path": "/v1/chat/completions",
            "headers": [(b"x-smg-attempt-id", b"opaque"), (b"x-smg-hard-ttft-timeout-ms", b"1000")],
        }
        with patch.object(native, "time", SimpleNamespace(monotonic=iter([100, 102]).__next__)):
            asyncio.run(middleware(scope, None, send))
        self.assertEqual(scope["smg_deadline"], ("opaque", 101))
        self.assertNotIn(b'"content":"late"', messages[-1]["body"])
        self.assertIn(b"native_ttft_deadline", messages[-1]["body"])
        self.assertIn(b"[DONE]", messages[-1]["body"])
        middleware.store.db.close()

    def test_http_ttl_disabled_and_unsupported_topology_rejected(self):
        async def app(scope, receive, send):
            self.fail("Must reject before native inference")

        messages = []

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "path": "/generate",
            "headers": [(b"x-smg-attempt-id", b"A"), (b"x-smg-hard-ttft-timeout-ms", b"1000")],
        }
        with patch.dict(os.environ, {"SMG_NATIVE_HARD_TTFT_ENFORCEMENT": "0"}):
            middleware = native.DeadlineHTTPMiddleware(
                app, lambda: SimpleNamespace(server_args=args())
            )
            asyncio.run(middleware(scope, None, send))
        self.assertEqual(messages[0]["status"], 400)
        middleware.store.db.close()
        config = args()
        config.tp_size = 2
        middleware = native.DeadlineHTTPMiddleware(app, lambda: SimpleNamespace(server_args=config))
        messages.clear()
        asyncio.run(middleware(scope, None, send))
        self.assertEqual(messages[0]["status"], 400)
        middleware.store.db.close()

    def test_http_committed_expiry_without_text_emits_explicit_deadline_error(self):
        payloads = [
            b"",
            b"data: [DONE]\n\n",
            b'data: {"choices":[{"text":"","finish_reason":"abort"}]}\n\ndata: [DONE]\n\n',
        ]
        for index, payload in enumerate(payloads):
            with self.subTest(payload=payload):
                attempt = f"expired-terminal-{index}"

                async def app(scope, receive, send):
                    self.store.activate(attempt)
                    await send({"type": "http.response.start", "status": 200, "headers": []})
                    self.assertTrue(self.store.claim_expired(attempt, now=102))
                    await send({"type": "http.response.body", "body": payload, "more_body": False})

                middleware = native.DeadlineHTTPMiddleware(
                    app, lambda: SimpleNamespace(server_args=args())
                )
                messages = []

                async def send(message):
                    messages.append(message)

                scope = {
                    "type": "http",
                    "path": "/v1/completions",
                    "headers": [
                        (b"x-smg-attempt-id", attempt.encode()),
                        (b"x-smg-hard-ttft-timeout-ms", b"1000"),
                    ],
                }
                with patch.object(native, "time", SimpleNamespace(monotonic=lambda: 100)):
                    asyncio.run(middleware(scope, None, send))
                wire = b"".join(m.get("body", b"") for m in messages)
                self.assertIn(b"native_ttft_deadline", wire)
                self.assertEqual(wire.count(b"[DONE]"), 1)
                self.assertFalse(messages[-1]["more_body"])
                middleware.store.db.close()

    def test_http_first_text_winner_and_no_deadline_terminal_remain_unchanged(self):
        for winner in (False, True):
            with self.subTest(first_text_winner=winner):
                attempt = f"nonexpired-terminal-{winner}"
                text = b'data: {"choices":[{"text":"first"}]}\n\n'
                terminal = b"data: [DONE]\n\n"

                async def app(scope, receive, send):
                    self.store.activate(attempt)
                    await send({"type": "http.response.start", "status": 200, "headers": []})
                    if winner:
                        await send({"type": "http.response.body", "body": text, "more_body": True})
                    self.assertFalse(self.store.claim_expired(attempt, now=102))
                    await send({"type": "http.response.body", "body": terminal, "more_body": False})

                middleware = native.DeadlineHTTPMiddleware(
                    app, lambda: SimpleNamespace(server_args=args())
                )
                messages = []

                async def send(message):
                    messages.append(message)

                headers = [(b"x-smg-attempt-id", attempt.encode())]
                if winner:
                    headers.append((b"x-smg-hard-ttft-timeout-ms", b"1000"))
                scope = {"type": "http", "path": "/v1/completions", "headers": headers}
                with patch.object(native, "time", SimpleNamespace(monotonic=lambda: 100)):
                    asyncio.run(middleware(scope, None, send))
                wire = b"".join(m.get("body", b"") for m in messages)
                self.assertEqual(wire, (text if winner else b"") + terminal)
                middleware.store.db.close()

    def test_http_validation_response_after_deadline_is_not_native_expiry(self):
        payload = b'{"detail":"native validation error"}'
        state = self.state(scheduler())

        async def app(scope, receive, send):
            # Validation rejects before scheduler admission. Passing the wall
            # deadline does not imply a committed native cancellation.
            self.assertEqual(native.time.monotonic(), 102)
            await send(
                {
                    "type": "http.response.start",
                    "status": 422,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": payload, "more_body": False})

        middleware = native.DeadlineHTTPMiddleware(app, lambda: SimpleNamespace(server_args=args()))
        messages = []

        async def send(message):
            messages.append(message)

        scope = {
            "type": "http",
            "path": "/v1/completions",
            "headers": [
                (b"x-smg-attempt-id", b"invalid-before-admission"),
                (b"x-smg-hard-ttft-timeout-ms", b"1000"),
            ],
        }
        with patch.object(native, "time", SimpleNamespace(monotonic=iter([100, 102]).__next__)):
            asyncio.run(middleware(scope, None, send))
        self.assertEqual(messages[0]["status"], 422)
        self.assertEqual(b"".join(m.get("body", b"") for m in messages), payload)
        self.assertFalse(self.store.expired_without_text("invalid-before-admission"))
        self.assertEqual(
            self.store.db.execute(
                "SELECT COUNT(*) FROM attempts WHERE attempt='invalid-before-admission'"
            ).fetchone()[0],
            0,
        )
        self.assertEqual(state.snapshot()["cancellation_acks"], [])
        middleware.store.db.close()


class PinnedNativeMethodTests(unittest.TestCase):
    """Real source method bodies, CPU context doubles; not full engine execution."""

    def setUp(self):
        source = os.environ.get("SGLANG_NATIVE_SOURCE")
        if not source:
            self.skipTest("Set SGLANG_NATIVE_SOURCE for pinned upstream method tests")
        DeadlineComponentTests.setUp(self)
        self.source = Path(source)
        _, outputs = installer.prepare(source)
        scheduler_tree = ast.parse(outputs["managers/scheduler.py"])
        cls = next(
            n for n in scheduler_tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
        )
        abort = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "abort_request"
        )
        io_tree = ast.parse((self.source / "python/sglang/srt/managers/io_struct.py").read_text())
        batch_tree = ast.parse(
            (self.source / "python/sglang/srt/managers/schedule_batch.py").read_text()
        )
        import dataclasses
        import typing
        from abc import ABC

        globs = {
            **vars(typing),
            **vars(dataclasses),
            "ABC": ABC,
            "logger": logging.getLogger(__name__),
            "DisaggregationMode": SimpleNamespace(NULL="null", PREFILL="prefill", DECODE="decode"),
        }
        classes = [
            n
            for n in io_tree.body
            if isinstance(n, ast.ClassDef) and n.name in ("BaseReq", "AbortReq")
        ]
        classes += [
            n
            for n in batch_tree.body
            if isinstance(n, ast.ClassDef) and n.name in ("BaseFinishReason", "FINISH_ABORT")
        ]
        exec(
            compile(ast.Module(body=classes, type_ignores=[]), "pinned-native-classes", "exec"),
            globs,
        )
        self.AbortReq = globs["AbortReq"]
        self.FinishAbort = globs["FINISH_ABORT"]
        exec(
            compile(ast.Module(body=[abort], type_ignores=[]), "pinned-native-abort", "exec"), globs
        )
        self.abort = globs["abort_request"]
        process_input = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "process_input_requests"
        )
        exec(
            compile(
                ast.Module(body=[process_input], type_ignores=[]), "pinned-native-input", "exec"
            ),
            globs,
        )
        self.process_input = globs["process_input_requests"]
        self.globs = globs

    tearDown = DeadlineComponentTests.tearDown
    state = DeadlineComponentTests.state

    def test_pinned_abort_waiting_exact_id_does_not_cancel_prefix_sibling(self):
        target, sibling = req("native-A"), req("native-A-sibling")
        target.smg_attempt_id = "opaque-attempt"
        sched = scheduler(target, sibling)
        sched.enable_hicache_storage = False
        sched.send_to_tokenizer = SimpleNamespace(send_output=lambda *args: None)
        state = self.state(sched)
        state.reconcile()
        abort = self.AbortReq(rid="native-A")
        abort.smg_exact = True
        state.mark_abort(abort)
        self.abort(sched, abort)
        snap = state.snapshot()
        self.assertEqual(sched.waiting_queue, [sibling])
        self.assertEqual(snap["cancellation_acks"][0]["attempt_id"], "opaque-attempt")
        self.assertIsNone(sibling.to_finish)

    def test_pinned_running_abort_is_not_ack_and_real_release_follows_finish(self):
        active = req("native-A", pool=3)
        active.smg_attempt_id = "opaque"
        sched = scheduler()
        sched.running_batch.reqs = [active]
        sched.enable_hicache_storage = False
        state = self.state(sched)
        state.reconcile()
        abort = self.AbortReq(rid="native-A")
        abort.smg_exact = True
        state.mark_abort(abort)
        self.abort(sched, abort)
        self.assertIsInstance(active.to_finish, self.FinishAbort)
        self.assertEqual(state.snapshot()["cancellation_acks"], [])
        # Execute actual pinned Req.check_finished before cache cleanup.
        batch_tree = ast.parse(
            (self.source / "python/sglang/srt/managers/schedule_batch.py").read_text()
        )
        cls = next(n for n in batch_tree.body if isinstance(n, ast.ClassDef) and n.name == "Req")
        check = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "check_finished"
        )
        exec(
            compile(ast.Module(body=[check], type_ignores=[]), "pinned-native-finish", "exec"),
            self.globs,
        )
        self.globs["check_finished"](active)
        self.assertTrue(active.finished())
        # Execute actual pinned release_kv_cache with a cache/allocator context double.
        common = ast.parse((self.source / "python/sglang/srt/mem_cache/common.py").read_text())
        release = next(
            n
            for n in common.body
            if isinstance(n, ast.FunctionDef) and n.name == "release_kv_cache"
        )
        release.args.args[0].annotation = release.args.args[1].annotation = None
        resources = {"freed": False}
        pop_overallocated = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "pop_overallocated_kv_cache"
        )
        exec(
            compile(
                ast.Module(body=[pop_overallocated], type_ignores=[]),
                "pinned-native-pop-overallocated",
                "exec",
            ),
            self.globs,
        )
        from types import MethodType

        active.kv_committed_len = 2
        active.kv_allocated_len = 4
        active.kv_overallocated_freed = False
        active.pop_overallocated_kv_cache = MethodType(
            self.globs["pop_overallocated_kv_cache"], active
        )
        cache = SimpleNamespace(
            cache_finished_req=lambda *a, **kw: resources.update(freed=True),
            req_to_token_pool=SimpleNamespace(req_to_token={3: [0, 1, 2, 3]}),
            token_to_kv_pool_allocator=SimpleNamespace(
                free=lambda tokens: resources.update(freed_tokens=tokens)
            ),
        )
        self.globs["get_global_server_args"] = lambda: SimpleNamespace(
            page_size=1, speculative_algorithm="EAGLE"
        )
        exec(
            compile(ast.Module(body=[release], type_ignores=[]), "pinned-native-release", "exec"),
            self.globs,
        )
        native.release_observer(self.globs["release_kv_cache"])(active, cache)
        self.assertTrue(resources["freed"])
        self.assertTrue(active.kv_overallocated_freed)
        self.assertEqual(resources["freed_tokens"], [2, 3])
        self.assertEqual(state.snapshot()["cancellation_acks"], [])
        sched.running_batch.reqs = []
        self.assertEqual(len(state.snapshot()["cancellation_acks"]), 1)

    def test_pinned_scheduler_tick_deadline_firsttext_and_hidden_inflight(self):
        actual_abort, actual_input = self.abort, self.process_input

        class NativeScheduler:
            def __init__(self):
                self.__dict__.update(vars(scheduler()))
                self.enable_hicache_storage = False
                self.send_to_tokenizer = SimpleNamespace(send_output=lambda *args: None)

            abort_request = actual_abort
            process_input_requests = actual_input

            def handle_generate_request(self, request):
                self.waiting_queue.append(req(request.rid))

            def get_load(self, recv_req=None):
                return SimpleNamespace()

            def run_batch(self, batch):
                return None

            def process_batch_result(self, batch, result):
                return None

        io_module = ModuleType("sglang.srt.managers.io_struct")
        io_module.AbortReq = self.AbortReq
        batch_module = ModuleType("sglang.srt.managers.schedule_batch")
        batch_module.FINISH_ABORT = self.FinishAbort
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.managers.io_struct": io_module,
                "sglang.srt.managers.schedule_batch": batch_module,
            },
        ):
            native.install_scheduler(NativeScheduler)
            engine = NativeScheduler()
            state = engine._smg_deadline
            try:
                self.store.register("expired-attempt", 100, 101)
                engine.handle_generate_request(
                    SimpleNamespace(
                        rid="native-expired", smg_attempt_id="expired-attempt", smg_deadline=101
                    )
                )
                # An actual scheduler tick calls the actual pinned abort body.
                with patch.object(native, "time", SimpleNamespace(monotonic=lambda: 102)):
                    engine.process_input_requests([])
                self.assertEqual(engine.waiting_queue, [])
                ack = state.snapshot()["cancellation_acks"][0]
                self.assertEqual(ack["reason"], "hard_ttft_deadline")
                self.assertEqual(ack["attempt_id"], "expired-attempt")
                self.store.register("healthy-attempt", 100, 101)
                self.assertTrue(self.store.observe_text("healthy-attempt", 100.9))
                engine.handle_generate_request(
                    SimpleNamespace(
                        rid="native-healthy", smg_attempt_id="healthy-attempt", smg_deadline=101
                    )
                )
                with patch.object(native, "time", SimpleNamespace(monotonic=lambda: 102)):
                    engine.process_input_requests([])
                self.assertEqual(len(engine.waiting_queue), 1)
                self.assertIsNone(engine.waiting_queue[0].to_finish)
                # Native abort misses a request held only in overlap/chunk owners;
                # the installed hook marks native to_finish without premature ACK.
                self.store.register("chunk-attempt", 100, 101)
                chunk = req("native-chunk", pool=2)
                chunk.smg_attempt_id = "chunk-attempt"
                chunk.smg_deadline = 101
                engine.chunked_req = chunk
                engine.result_queue = [(SimpleNamespace(reqs=[chunk]), None)]
                with patch.object(native, "time", SimpleNamespace(monotonic=lambda: 102)):
                    engine.process_input_requests([])
                self.assertIsInstance(chunk.to_finish, self.FinishAbort)
                self.assertEqual(len(state.snapshot()["cancellation_acks"]), 1)
            finally:
                state.store.db.close()

    def test_actual_pinned_native_api_key_middleware_protects_new_route(self):
        common = ast.parse((self.source / "python/sglang/srt/utils/common.py").read_text())
        auth = next(
            n
            for n in common.body
            if isinstance(n, ast.FunctionDef) and n.name == "add_api_key_middleware"
        )
        auth_globals = {"ORJSONResponse": lambda **kwargs: SimpleNamespace(**kwargs)}
        exec(
            compile(ast.Module(body=[auth], type_ignores=[]), "pinned-native-auth", "exec"),
            auth_globals,
        )

        class App:
            def middleware(self, kind):
                def register(callback):
                    self.callback = callback
                    return callback

                return register

        app = App()
        auth_globals["add_api_key_middleware"](app, "component-test-key")

        async def call_next(request):
            return SimpleNamespace(status_code=200)

        for header, expected in (
            (None, 401),
            ("Bearer wrong", 401),
            ("Bearer component-test-key", 200),
        ):
            request = SimpleNamespace(
                method="GET",
                url=SimpleNamespace(path="/v1/native_deadline_snapshot"),
                headers={} if header is None else {"Authorization": header},
            )
            response = asyncio.run(app.callback(request, call_next))
            self.assertEqual(response.status_code, expected)

    def test_installer_drift_refusal_is_atomic(self):
        with tempfile.TemporaryDirectory() as path:
            copyroot = Path(path)
            import shutil

            shutil.copytree(self.source / "python", copyroot / "python")
            target = copyroot / "python/sglang/srt/managers/scheduler.py"
            before = target.read_bytes()
            other = copyroot / "python/sglang/srt/entrypoints/http_server.py"
            other.write_text(other.read_text() + "\n# source drift\n")
            with self.assertRaisesRegex(ValueError, "source drift"):
                installer.apply(copyroot)
            self.assertEqual(target.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
