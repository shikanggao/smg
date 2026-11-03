"""GPU-free native hook tests, including pinned upstream scheduler methods.

Set SMG_VLLM_019_SOURCE to the official v0.19.0 source directory to run the
upstream lifecycle tests. Those tests execute upstream code, not a replacement
implementation of finish_requests / cache release. GPU/model execution is not
tested here.
"""

import ast
import asyncio
import enum
import hashlib
import importlib.util
import json
import os
import secrets
import sqlite3
import tempfile
import unittest
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    from starlette.datastructures import URL, Headers
    from starlette.responses import JSONResponse
    from starlette.routing import Route, Router
except ImportError:
    Router = None

MODULE = Path(__file__).parents[1] / "smg_grpc_servicer" / "native_deadline_vllm.py"
SPEC = importlib.util.spec_from_file_location("native_deadline_vllm", MODULE)
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


class Status(enum.IntEnum):
    WAITING = 0
    RUNNING = 1
    WAITING_FOR_STREAMING_REQ = 2
    WAITING_FOR_REMOTE_KVS = 3
    FINISHED_ABORTED = 4
    FINISHED_STOPPED = 5

    @staticmethod
    def is_finished(value):
        return value >= Status.FINISHED_ABORTED


class Request:
    def __init__(self, rid="internal-exact-UUID", attempt="dispatch-exact-UUID"):
        self.request_id, self.status = rid, Status.WAITING
        self.sampling_params = SimpleNamespace(
            extra_args={native.ATTEMPT_XARG: attempt, native.BINDING_XARG: "source-test"}
        )
        self.client_index = 0
        self.num_prompt_tokens, self.num_computed_tokens = 100, 0
        self.num_output_tokens, self.num_cached_tokens = 0, -1
        self.resumable = False
        self.delay_free = False

    def is_finished(self):
        return Status.is_finished(self.status)


class StoreTestBase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = native.NativeStore(Path(self.temp.name) / "native.db")

    def native_ready(self):
        scheduler = SimpleNamespace(
            requests={},
            running=[],
            waiting=[],
            skipped_waiting=[],
            parallel_config=SimpleNamespace(data_parallel_rank=0),
        )
        native.SchedulerSignals(scheduler, self.store, 0).publish()


class StoreTests(StoreTestBase):
    def test_first_nonempty_text_wins_expiry_race(self):
        self.store.register("a", 1000, 10.0)
        self.assertTrue(self.store.first_text("a", 10.999))
        self.assertEqual(self.store.expire(["a"], 100.0), set())
        self.assertTrue(self.store.first_text("a", 100.0))

    def test_expiry_wins_and_late_text_cannot_escape(self):
        self.store.register("a", 1000, 10.0)
        self.assertEqual(self.store.expire(["a"], 11.0), {"a"})
        self.assertFalse(self.store.first_text("a", 11.0))
        self.assertEqual(self.store.request_state("a")[3], "hard_ttft_deadline")

    def test_text_at_exact_deadline_loses_without_scheduler_tick(self):
        self.store.register("a", 1000, 10.0)
        self.assertFalse(self.store.first_text("a", 11.0))
        self.assertEqual(self.store.expire(["a"], 11.0), {"a"})

    def test_threshold_is_not_default_deadline_and_duplicate_attempt_is_rejected(self):
        self.assertIsNone(self.store.register("a", None, 10.0))
        self.assertEqual(self.store.expire(["a"], 10000000), set())
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.register("a", 300000, 11.0)
        self.assertFalse(self.store.first_text("unknown", 10))

    def test_disconnect_does_not_create_ack(self):
        self.store.register("a", None, 10)
        self.store.disconnect("a")
        self.assertFalse(self.store.loads(100)["loads"])

    def test_two_connections_serialize_firsttext_against_expiry(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        other = native.NativeStore(self.store.path)
        for n in range(10):
            attempt = str(n)
            self.store.register(attempt, 1000, 10)
            barrier = Barrier(2)

            def text():
                barrier.wait()
                return self.store.first_text(attempt, 10.5)

            def expiry():
                barrier.wait()
                return bool(other.expire([attempt], 11))

            with ThreadPoolExecutor(2) as pool:
                first, second = pool.submit(text), pool.submit(expiry)
                self.assertNotEqual(first.result(), second.result())


def upstream_scheduler(source):
    """Load unmodified upstream lifecycle bodies without GPU import side effects."""
    path = source / "vllm/v1/core/sched/scheduler.py"
    code = path.read_bytes()
    expected = native.SOURCE_HASHES["vllm.v1.core.sched.scheduler"]
    if hashlib.sha256(code).hexdigest() != expected:
        raise AssertionError("upstream Scheduler source is not the pinned official commit")
    tree = ast.parse(code)
    upstream = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    names = {"add_request", "finish_requests", "_free_request", "_free_blocks"}
    methods = [n for n in upstream.body if isinstance(n, ast.FunctionDef) and n.name in names]
    # Deferred annotations avoid importing unrelated GPU/runtime dependencies.
    code_tree = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            ast.ClassDef(
                name="UpstreamScheduler", bases=[], keywords=[], body=methods, decorator_list=[]
            ),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(code_tree)
    namespace = {
        "RequestStatus": Status,
        "deque": deque,
        "remove_all": lambda rows, removed: [r for r in rows if r not in removed],
    }
    exec(compile(code_tree, str(path), "exec"), namespace)
    cls = namespace["UpstreamScheduler"]

    def init(self):
        self.parallel_config = SimpleNamespace(data_parallel_rank=0)
        self.requests, self.running = {}, []
        self.waiting, self.skipped_waiting = Queue(), Queue()
        self.finished_req_ids, self.finished_req_ids_dict = set(), defaultdict(set)
        self.finished_recving_kv_req_ids, self.failed_recving_kv_req_ids = set(), set()
        self.num_waiting_for_streaming_input, self.log_stats = 0, False
        self.releases = []
        self.encoder_cache_manager = SimpleNamespace(
            free=lambda r: self.releases.append((r.request_id, "encoder"))
        )
        self.kv_cache_manager = SimpleNamespace(
            free=lambda r: self.releases.append((r.request_id, "kv"))
        )
        self._connector_finished = lambda request: (request.delay_free, None)
        self._enqueue_waiting_request = self.waiting.add_request

    cls.__init__ = init

    # The schedule and model-output call boundaries below are instrumentation
    # fixtures. Only the four lifecycle methods above are upstream execution.
    def schedule(self):
        output = SimpleNamespace(num_scheduled_tokens={}, finished_req_ids=self.finished_req_ids)
        self.finished_req_ids = set()
        return output

    cls.schedule = schedule
    cls.update_from_output = lambda self, scheduler_output, output=None: None
    return cls


class Queue(deque):
    def add_request(self, request):
        self.append(request)

    def remove_requests(self, requests):
        for request in requests:
            if request in self:
                self.remove(request)


@unittest.skipUnless(os.environ.get("SMG_VLLM_019_SOURCE"), "requires pinned upstream source")
class UpstreamLifecycleTests(StoreTestBase):
    def setUp(self):
        super().setUp()
        cls = upstream_scheduler(Path(os.environ["SMG_VLLM_019_SOURCE"]))
        native.install_scheduler_hooks(cls, Status.FINISHED_ABORTED, lambda: self.store)
        self.scheduler = cls()
        self.store.register("dispatch-exact-UUID", None, 10)

    def snapshot(self):
        return self.store.loads(100)["loads"][0]["deadline_recovery"]

    def test_actual_upstream_abort_releases_encoder_and_kv_before_ack(self):
        request = Request()
        self.scheduler.add_request(request)
        self.scheduler.finish_requests(request.request_id, Status.FINISHED_ABORTED)
        self.assertEqual(
            self.scheduler.releases, [(request.request_id, "encoder"), (request.request_id, "kv")]
        )
        ack = self.snapshot()["cancellation_acks"]
        self.assertEqual(len(ack), 1)
        self.assertEqual(ack[0]["native_request_id"], request.request_id)
        self.assertEqual(ack[0]["attempt_id"], "dispatch-exact-UUID")
        self.assertFalse(self.scheduler.requests)
        self.assertFalse(self.scheduler.waiting)

    def test_wrong_native_id_and_duplicate_abort_never_add_acks(self):
        request = Request()
        self.scheduler.add_request(request)
        self.scheduler.finish_requests(request.request_id + "-wrong", Status.FINISHED_ABORTED)
        self.assertFalse(self.snapshot()["cancellation_acks"])
        self.scheduler.finish_requests(request.request_id, Status.FINISHED_ABORTED)
        self.scheduler.finish_requests(request.request_id, Status.FINISHED_ABORTED)
        self.assertEqual(len(self.snapshot()["cancellation_acks"]), 1)

    def test_user_xargs_cannot_alias_or_abort_another_native_attempt(self):
        original = Request()
        self.scheduler.add_request(original)
        counterfeit = Request("counterfeit-native-id")
        counterfeit.sampling_params.extra_args[native.BINDING_XARG] = "user-controlled-wrong-token"
        self.scheduler.add_request(counterfeit)
        duplicate = Request("different-native-id")
        self.scheduler.add_request(duplicate)
        rows = self.snapshot()["active_requests"]
        self.assertEqual([r["attempt_id"] for r in rows], ["dispatch-exact-UUID", None, None])
        self.scheduler.finish_requests(counterfeit.request_id, Status.FINISHED_ABORTED)
        self.scheduler.finish_requests(duplicate.request_id, Status.FINISHED_ABORTED)
        self.assertFalse(self.snapshot()["cancellation_acks"])
        self.assertIn(original.request_id, self.scheduler.requests)

    def test_delayed_native_connector_release_defers_ack(self):
        request = Request()
        request.delay_free = True
        self.scheduler.add_request(request)
        self.scheduler.finish_requests(request.request_id, Status.FINISHED_ABORTED)
        self.assertFalse(self.snapshot()["cancellation_acks"])
        self.assertIn(request.request_id, self.scheduler.requests)
        self.scheduler._free_blocks(request)
        self.assertEqual(len(self.snapshot()["cancellation_acks"]), 1)

    def test_inflight_gpu_completion_is_also_required_for_ack(self):
        request = Request()
        self.scheduler.add_request(request)
        output = SimpleNamespace(num_scheduled_tokens={request.request_id: 20})
        self.scheduler._smg_signals.inflight[id(output)] = output.num_scheduled_tokens
        self.scheduler._smg_signals.ever_scheduled.add(request.request_id)
        self.scheduler.finish_requests(request.request_id, Status.FINISHED_ABORTED)
        self.assertFalse(self.snapshot()["cancellation_acks"])
        self.scheduler.update_from_output(output)
        self.assertFalse(self.snapshot()["cancellation_acks"])
        self.assertEqual(self.snapshot()["native_population"], 1)
        cleanup = self.scheduler.schedule()
        self.scheduler.update_from_output(cleanup)
        self.assertEqual(len(self.snapshot()["cancellation_acks"]), 1)

    def test_native_completed_work_subtracts_inflight_and_unknown_prefix(self):
        request = Request()
        self.scheduler.add_request(request)
        self.assertIsNone(self.snapshot()["active_requests"][0]["remaining_prefill_work_tokens"])
        request.num_cached_tokens, request.num_computed_tokens = 40, 70
        self.scheduler._smg_signals.inflight[1] = {request.request_id: 20}
        self.scheduler._smg_signals.publish()
        self.assertEqual(self.snapshot()["active_requests"][0]["remaining_prefill_work_tokens"], 50)
        request.num_output_tokens = 1
        self.scheduler._smg_signals.publish()
        self.assertIsNone(self.snapshot()["active_requests"][0]["remaining_prefill_work_tokens"])

    def test_preempted_decoder_work_and_unsupported_topology_are_unknown(self):
        request = Request()
        self.scheduler.add_request(request)
        self.store.first_text("dispatch-exact-UUID", 10)
        request.num_preemptions = 1
        self.scheduler._smg_signals.publish()
        self.assertIsNone(self.snapshot()["active_requests"][0]["remaining_prefill_work_tokens"])
        self.scheduler.parallel_config.tensor_parallel_size = 2
        signals = native.SchedulerSignals(self.scheduler, self.store, 0)
        signals.publish()
        self.assertFalse(self.snapshot()["active_inventory_complete"])
        self.assertFalse(self.snapshot()["cancellation_ack_supported"])

    def test_explicit_deadline_aborts_via_real_scheduler_and_notifies_frontend(self):
        self.store.register("expired", 1, 0)
        request = Request(attempt="expired")
        notifications = []
        self.scheduler._smg_abort_outputs = notifications.append
        self.scheduler.add_request(request)
        self.scheduler.schedule()
        self.assertEqual(notifications, [[(request.request_id, 0)]])
        self.assertEqual(self.snapshot()["cancellation_acks"][0]["reason"], "hard_ttft_deadline")

    def test_first_text_prevents_actual_upstream_abort(self):
        self.store.register("text", 1000, 0)
        self.assertTrue(self.store.first_text("text", 0.5))
        request = Request(attempt="text")
        self.scheduler._smg_abort_outputs = lambda _: self.fail("text must not be aborted")
        self.scheduler.add_request(request)
        self.scheduler.schedule()
        self.assertIn(request.request_id, self.scheduler.requests)

    def test_full_inventory_includes_non_gateway_work_and_restart_epoch_changes(self):
        self.scheduler.add_request(Request(attempt="dispatch-exact-UUID"))
        self.scheduler.add_request(Request("native-untracked-exact-id", None))
        snap = self.snapshot()
        self.assertEqual(snap["native_population"], 2)
        self.assertIsNone(snap["active_requests"][1]["attempt_id"])
        self.assertIsNone(snap["effective_capacity_tokens_per_sec"])
        self.assertIsNone(snap["abort_max_lag_seconds"])
        other = type(self.scheduler)()
        self.assertNotEqual(self.snapshot()["engine_epoch"], snap["engine_epoch"])
        self.assertEqual(other._smg_signals.profile, self.scheduler._smg_signals.profile)

    def test_ack_retransmit_and_journal_eviction_make_lost_ack_detectable(self):
        with patch.object(native, "ACK_LIMIT", 1):
            for n in range(2):
                self.store.register(f"attempt-{n}", None, 10)
                request = Request(f"native-{n}", f"attempt-{n}")
                self.scheduler.add_request(request)
                self.scheduler.finish_requests(request.request_id, Status.FINISHED_ABORTED)
                if n == 0:
                    first = self.snapshot()["cancellation_acks"][0]
                    self.scheduler._smg_signals.publish()
                    self.assertEqual(self.snapshot()["cancellation_acks"][0], first)
            snap = self.snapshot()
            self.assertEqual(snap["ack_floor_seq"], first["event_seq"] + 1)
            self.assertLessEqual(snap["cancellation_acks"][0]["event_seq"], snap["snapshot_seq"])


class MiddlewareTests(StoreTestBase):
    def call(self, app, body, headers=None):
        output = []
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": headers or [(b"x-smg-attempt-id", b"exact-id")],
        }
        incoming = iter(
            [{"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}]
        )

        async def receive():
            return next(incoming)

        async def send(event):
            output.append(event)

        middleware = native.NativeDeadlineMiddleware.__new__(native.NativeDeadlineMiddleware)
        middleware.request_app, middleware.store = app, self.store
        asyncio.run(middleware._handle_request(scope, receive, send))
        return output

    def test_http_identity_survives_exact_xargs_binding_and_partial_sse(self):
        async def app(scope, receive, send):
            data = json.loads((await receive())["body"])
            self.assertEqual(data["vllm_xargs"][native.ATTEMPT_XARG], "exact-id")
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send(
                {
                    "type": "http.response.body",
                    "body": b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
                    "more_body": True,
                }
            )
            self.assertFalse(self.store.request_state("exact-id")[1])
            await send(
                {
                    "type": "http.response.body",
                    "body": b'data: {"choices":[{"delta":{"content":"h',
                    "more_body": True,
                }
            )
            self.assertFalse(self.store.request_state("exact-id")[1])
            await send(
                {"type": "http.response.body", "body": b'i"}}]}\n\ndata: [DO', "more_body": True}
            )
            self.assertTrue(self.store.request_state("exact-id")[1])
            await send({"type": "http.response.body", "body": b"NE]\n\n", "more_body": False})

        output = self.call(app, {"stream": True})
        payload = b"".join(event.get("body", b"") for event in output)
        self.assertIn(b'"content":"hi"', payload)
        self.assertIn(b"data: [DONE]", payload)

    def test_expiry_winner_emits_error_and_never_false_healthy_text(self):
        self.native_ready()

        async def app(scope, receive, send):
            await receive()
            self.store.expire(["exact-id"], 1e30)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send(
                {
                    "type": "http.response.body",
                    "body": b'data: {"choices":[{"delta":{"content":"too late"}}]}\n\n',
                    "more_body": True,
                }
            )
            await send(
                {"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False}
            )

        output = self.call(
            app,
            {"stream": True},
            [(b"x-smg-attempt-id", b"exact-id"), (b"x-smg-hard-ttft-timeout-ms", b"1")],
        )
        payload = b"".join(event.get("body", b"") for event in output)
        self.assertIn(b"native_ttft_deadline", payload)
        self.assertNotIn(b"too late", payload)

    @unittest.skipUnless(os.environ.get("SMG_VLLM_019_SOURCE"), "requires pinned upstream source")
    def test_actual_scheduler_expiry_without_text_emits_error_and_one_done(self):
        cls = upstream_scheduler(Path(os.environ["SMG_VLLM_019_SOURCE"]))
        native.install_scheduler_hooks(cls, Status.FINISHED_ABORTED, lambda: self.store)
        scheduler = cls()
        notifications = []
        scheduler._smg_abort_outputs = notifications.append
        scheduler._smg_signals.publish()
        finish = b'data: {"choices":[{"delta":{},"finish_reason":"abort"}]}\n\n'
        done = b"data: [DONE]\n\n"
        for n, chunks in enumerate(((b"",), (done,), (finish, done), (finish + done,))):
            with self.subTest(chunks=chunks):
                attempt = f"expired-without-text-{n}"
                request = Request(f"native-without-text-{n}", attempt)

                async def app(scope, receive, send):
                    data = json.loads((await receive())["body"])
                    request.sampling_params.extra_args = data["vllm_xargs"]
                    scheduler.add_request(request)
                    deadline = self.store.request_state(attempt)[0]
                    with patch.object(native.time, "monotonic", return_value=deadline + 1):
                        scheduler.schedule()
                    self.assertEqual(request.status, Status.FINISHED_ABORTED)
                    self.assertEqual(notifications[-1], [(request.request_id, 0)])
                    self.assertEqual(self.store.request_state(attempt)[1:3], (0, 1))
                    await send({"type": "http.response.start", "status": 200, "headers": []})
                    for index, chunk in enumerate(chunks):
                        await send(
                            {
                                "type": "http.response.body",
                                "body": chunk,
                                "more_body": index < len(chunks) - 1,
                            }
                        )

                output = self.call(
                    app,
                    {"stream": True},
                    [
                        (b"x-smg-attempt-id", attempt.encode()),
                        (b"x-smg-hard-ttft-timeout-ms", b"1000"),
                    ],
                )
                payload = b"".join(event.get("body", b"") for event in output)
                self.assertIn(b"native_ttft_deadline", payload)
                self.assertEqual(payload.count(b"data: [DONE]"), 1)
                self.assertNotIn(b'"finish_reason":"abort"', payload)
                self.assertFalse(output[-1]["more_body"])
                acks = self.store.loads(0)["loads"][0]["deadline_recovery"]["cancellation_acks"]
                self.assertEqual(len(acks), n + 1)
                self.assertEqual(acks[-1]["attempt_id"], attempt)
                self.assertEqual(acks[-1]["reason"], "hard_ttft_deadline")
                self.assertEqual(
                    scheduler.releases[-2:],
                    [(request.request_id, "encoder"), (request.request_id, "kv")],
                )

    def test_first_text_winner_keeps_later_terminal_stream_unchanged(self):
        self.native_ready()
        text = b'data: {"choices":[{"delta":{"content":"on time"}}]}\n\n'
        finish = b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        done = b"data: [DONE]\n\n"

        async def app(scope, receive, send):
            await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": text, "more_body": True})
            self.assertEqual(self.store.expire(["exact-id"], 1e30), set())
            await send({"type": "http.response.body", "body": finish, "more_body": True})
            await send({"type": "http.response.body", "body": done, "more_body": False})

        output = self.call(
            app,
            {"stream": True},
            [(b"x-smg-attempt-id", b"exact-id"), (b"x-smg-hard-ttft-timeout-ms", b"1000")],
        )
        self.assertEqual(b"".join(event.get("body", b"") for event in output), text + finish + done)
        self.assertFalse(self.store.loads(0)["loads"][0]["deadline_recovery"]["cancellation_acks"])

    def test_no_deadline_no_text_stream_remains_unchanged(self):
        payload = b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'

        async def app(scope, receive, send):
            await receive()
            self.assertEqual(self.store.expire(["exact-id"], 1e30), set())
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": payload, "more_body": False})

        output = self.call(app, {"stream": True})
        self.assertEqual(b"".join(event.get("body", b"") for event in output), payload)
        self.assertFalse(self.store.loads(0)["loads"])

    def test_unsupported_ttl_shapes_and_nonfinite_ttl_are_rejected(self):
        async def app(*args):
            self.fail("unsupported request must not reach the engine")

        for n, body in enumerate(({"stream": False}, {"stream": True, "n": 2})):
            out = self.call(
                app,
                body,
                [(b"x-smg-attempt-id", str(n).encode()), (b"x-smg-hard-ttft-timeout-ms", b"100")],
            )
            self.assertEqual(out[0]["status"], 400)
        for ttl in (b"NaN", b"inf", b"-1", b"0", b"1.1"):
            out = self.call(
                app,
                {"stream": True},
                [(b"x-smg-attempt-id", b"valid"), (b"x-smg-hard-ttft-timeout-ms", ttl)],
            )
            self.assertEqual(out[0]["status"], 400)

    def test_disabled_plugin_does_not_import_vllm(self):
        with patch.dict(os.environ, {"SMG_NATIVE_DEADLINE_ENABLED": "0"}):
            native.register()  # this test host has no installed vLLM


@unittest.skipUnless(
    Router is not None and os.environ.get("SMG_VLLM_019_SOURCE"),
    "requires Starlette and pinned source",
)
class NativeAuthenticationTests(StoreTestBase):
    def auth_class(self):
        path = Path(os.environ["SMG_VLLM_019_SOURCE"]) / "vllm/entrypoints/openai/server_utils.py"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            native.SOURCE_HASHES["vllm.entrypoints.openai.server_utils"],
        )
        tree = ast.parse(path.read_bytes())
        auth = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AuthenticationMiddleware"
        )
        code = ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                auth,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(code)
        namespace = {
            "hashlib": hashlib,
            "secrets": secrets,
            "URL": URL,
            "Headers": Headers,
            "JSONResponse": JSONResponse,
        }
        exec(compile(code, str(path), "exec"), namespace)
        return namespace["AuthenticationMiddleware"]

    def request(self, app, path, authorized, method="GET", body=b"", extra_headers=()):
        output = []
        headers = ([(b"authorization", b"Bearer test-key")] if authorized else []) + list(
            extra_headers
        )
        scope = {
            "type": "http",
            "method": method,
            "path": path,
            "root_path": "",
            "scheme": "http",
            "query_string": b"",
            "headers": headers,
        }

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(event):
            output.append(event)

        asyncio.run(app(scope, receive, send))
        return output

    def test_opt_in_route_preserves_actual_native_api_key_authentication(self):
        async def raw_loads(request):
            return JSONResponse({"original_native_loads": True})

        router = Router(routes=[Route("/v1/loads", raw_loads)])
        app = self.auth_class()(router, tokens=["test-key"])
        with patch.dict(
            os.environ,
            {"SMG_NATIVE_DEADLINE_ENABLED": "1", "SMG_NATIVE_DEADLINE_DB": self.store.path},
        ):
            app = native.NativeDeadlineMiddleware(app)
        self.assertEqual(self.request(app, "/v1/native_deadline_snapshot", False)[0]["status"], 401)
        allowed = self.request(app, "/v1/native_deadline_snapshot", True)
        self.assertEqual(allowed[0]["status"], 200)
        self.assertEqual(json.loads(allowed[1]["body"])["loads"], [])
        original = self.request(app, "/v1/loads", True)
        self.assertEqual(json.loads(original[1]["body"]), {"original_native_loads": True})

    def test_disabled_route_is_absent(self):
        router = Router()
        app = self.auth_class()(router, tokens=["test-key"])
        with patch.dict(os.environ, {"SMG_NATIVE_DEADLINE_ENABLED": "0"}):
            app = native.NativeDeadlineMiddleware(app)
        self.assertEqual(self.request(app, "/v1/native_deadline_snapshot", True)[0]["status"], 404)

    def test_unauthorized_and_native_invalid_post_create_no_accepted_work(self):
        self.native_ready()

        async def original_validation(request):
            await request.body()
            return JSONResponse({"detail": "native validation response"}, status_code=422)

        router = Router(
            routes=[Route("/v1/chat/completions", original_validation, methods=["POST"])]
        )
        app = self.auth_class()(router, tokens=["test-key"])
        with patch.dict(
            os.environ,
            {"SMG_NATIVE_DEADLINE_ENABLED": "1", "SMG_NATIVE_DEADLINE_DB": self.store.path},
        ):
            app = native.NativeDeadlineMiddleware(app)
        headers = [
            (b"x-smg-attempt-id", b"invalid-attempt"),
            (b"x-smg-hard-ttft-timeout-ms", b"1000"),
        ]
        out = self.request(app, "/v1/chat/completions", False, "POST", b'{"stream":true}', headers)
        self.assertEqual(out[0]["status"], 401)
        self.assertIsNone(self.store.request_state("invalid-attempt"))
        out = self.request(app, "/v1/chat/completions", True, "POST", b'{"stream":true}', headers)
        self.assertEqual(out[0]["status"], 422)
        self.assertEqual(json.loads(out[1]["body"]), {"detail": "native validation response"})
        self.assertIsNone(self.store.request_state("invalid-attempt"))
        self.assertEqual(
            self.store.loads(100)["loads"][0]["deadline_recovery"]["native_population"], 0
        )


if __name__ == "__main__":
    unittest.main()
