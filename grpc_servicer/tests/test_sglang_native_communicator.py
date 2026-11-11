"""Exact pinned communicator race and opt-in native HTTP integration tests.

SGLANG_NATIVE_COMMUNICATOR_SOURCE points at tokenizer_communicator_mixin.py
from SGLang c9e5799. Only its real _Communicator AST is executed; no CUDA.
"""

import ast
import asyncio
import copy
import hashlib
import importlib.util
import os
import sys
import typing
import unittest
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

PIN = "2c5f7a551676b210c4703afb1144c5ef595dc12c376117746c9479a639dbf77d"
ROOT = Path(__file__).resolve().parents[1] / "native_patches/sglang"
spec = importlib.util.spec_from_file_location(
    "sglang_native_load_test", ROOT / "native_deadline.py"
)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class NativeLoadCommunicator(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        path = os.environ.get("SGLANG_NATIVE_COMMUNICATOR_SOURCE")
        if not path:
            self.skipTest("Set SGLANG_NATIVE_COMMUNICATOR_SOURCE for exact pinned race tests")
        raw = Path(path).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), PIN)
        tree = ast.parse(raw)
        body = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_Communicator"
        )
        globs = {
            **vars(typing),
            "T": typing.TypeVar("T"),
            "asyncio": asyncio,
            "copy": copy,
            "deque": deque,
            "zmq": SimpleNamespace(Socket=object),
        }
        exec(
            compile(ast.Module(body=[body], type_ignores=[]), "pinned-communicator", "exec"), globs
        )
        self.Communicator = globs["_Communicator"]
        self.env = patch.dict(os.environ, {"SMG_NATIVE_DEADLINE_SIGNALS": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        io_struct = ModuleType("sglang.srt.managers.io_struct")
        io_struct.GetLoadReqInput = type("GetLoadReqInput", (), {})
        self.modules = patch.dict(sys.modules, {"sglang.srt.managers.io_struct": io_struct})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def communicator(self):
        sender = SimpleNamespace(sent=[])
        sender.send_pyobj = sender.sent.append
        return self.Communicator(sender, 1, mode="watching"), sender

    async def sent(self, sender, count=1):
        for _ in range(10):
            if len(sender.sent) == count:
                return
            await asyncio.sleep(0)
        self.fail("Native scheduler load request not sent")

    def endpoint(self, communicator):
        class Tokenizer:
            def __init__(self):
                self.server_args = SimpleNamespace(tokenizer_worker_num=1)
                self.get_load_communicator = communicator

            def auto_create_handle_loop(self):
                pass

            async def generate_request(self, obj, request=None):
                yield obj

            def _send_one_request(self, *args):
                pass

        native.install_tokenizer(Tokenizer)
        manager = Tokenizer()

        class App:
            def add_middleware(self, *args, **kwargs):
                pass

            def add_api_route(self, path, function, methods):
                self.endpoint = function

        app = App()
        native.install_http(app, lambda: manager)
        return getattr(app, "endpoint", None), manager

    def response(self):
        return SimpleNamespace(
            dp_rank=0,
            deadline_recovery=dict(
                captured_at_monotonic_secs=100,
                snapshot_seq=42,
                native_population=1,
                active_requests=[dict(state="prefill", attempt_id="exact-attempt")],
            ),
        )

    async def test_exact_upstream_watching_race_returns_none_to_second_waiter(self):
        communicator, sender = self.communicator()
        first = asyncio.create_task(communicator(object()))
        second = asyncio.create_task(communicator(object()))
        await self.sent(sender)
        await asyncio.sleep(0)
        communicator.handle_recv(self.response())
        values = await asyncio.wait_for(asyncio.gather(first, second), 1)
        self.assertEqual(len(sender.sent), 1)
        self.assertEqual(sum(value is None for value in values), 1)

    async def test_concurrent_native_endpoints_get_complete_independent_same_capture(self):
        communicator, sender = self.communicator()
        endpoint, _ = self.endpoint(communicator)
        first = asyncio.create_task(endpoint())
        second = asyncio.create_task(endpoint())
        await self.sent(sender)
        await asyncio.sleep(0)
        communicator.handle_recv(self.response())
        with patch.object(native, "time", SimpleNamespace(monotonic=iter([101, 102]).__next__)):
            values = await asyncio.wait_for(asyncio.gather(first, second), 1)
        self.assertEqual(len(sender.sent), 1)
        self.assertEqual([v["dp_rank_count"] for v in values], [1, 1])
        lanes = [v["loads"][0]["deadline_recovery"] for v in values]
        self.assertEqual([v["snapshot_seq"] for v in lanes], [42, 42])
        self.assertEqual([v["captured_at_monotonic_secs"] for v in lanes], [100, 100])
        self.assertEqual([v["snapshot_age_seconds"] for v in lanes], [1, 2])
        self.assertEqual(
            [v["active_requests"][0]["attempt_id"] for v in lanes], ["exact-attempt"] * 2
        )
        lanes[1]["active_requests"].clear()
        self.assertEqual(len(lanes[0]["active_requests"]), 1)

    async def test_cancelled_http_waiter_cannot_cancel_shared_native_operation(self):
        communicator, sender = self.communicator()
        endpoint, manager = self.endpoint(communicator)
        cancelled = asyncio.create_task(endpoint())
        survivor = asyncio.create_task(manager.get_load_communicator(object()))
        await self.sent(sender)
        await asyncio.sleep(0)
        cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled
        communicator.handle_recv(self.response())
        values = await asyncio.wait_for(survivor, 1)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0].deadline_recovery["snapshot_seq"], 42)
        self.assertEqual(len(sender.sent), 1)

    async def test_only_cancelled_waiter_still_drains_then_next_poll_is_fresh(self):
        communicator, sender = self.communicator()
        endpoint, manager = self.endpoint(communicator)
        cancelled = asyncio.create_task(endpoint())
        await self.sent(sender)
        cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled
        communicator.handle_recv(self.response())
        for _ in range(10):
            if communicator._result_event is None:
                break
            await asyncio.sleep(0)
        self.assertIsNone(communicator._result_event)
        fresh = asyncio.create_task(manager.get_load_communicator(object()))
        await self.sent(sender, count=2)
        response = self.response()
        response.deadline_recovery["snapshot_seq"] = 43
        communicator.handle_recv(response)
        result = await asyncio.wait_for(fresh, 1)
        self.assertEqual(result[0].deadline_recovery["snapshot_seq"], 43)

    async def test_exception_propagates_to_every_waiter_without_empty_inventory(self):
        communicator, sender = self.communicator()
        release = asyncio.Event()
        calls = []

        async def failure(obj):
            calls.append(obj)
            await release.wait()
            raise RuntimeError("native load failed")

        communicator.watching_call = failure
        endpoint, _ = self.endpoint(communicator)
        first, second = asyncio.create_task(endpoint()), asyncio.create_task(endpoint())
        await asyncio.sleep(0)
        release.set()
        values = await asyncio.gather(first, second, return_exceptions=True)
        self.assertTrue(all(isinstance(v, RuntimeError) for v in values))
        self.assertTrue(all(str(v) == "native load failed" for v in values))
        self.assertEqual(sender.sent, [])
        self.assertEqual(len(calls), 1)

    async def test_wrapping_is_instance_scoped_and_default_disabled_unchanged(self):
        first, _ = self.communicator()
        original = first.watching_call
        with patch.dict(os.environ, {"SMG_NATIVE_DEADLINE_SIGNALS": "0"}):
            endpoint, manager = self.endpoint(first)
        self.assertIsNone(endpoint)
        self.assertEqual(manager.get_load_communicator.watching_call, original)
        active, _ = self.communicator()
        untouched, _ = self.communicator()
        before = untouched.watching_call
        self.endpoint(active)
        self.assertEqual(untouched.watching_call, before)
        self.assertIsNot(
            active.watching_call.__func__
            if hasattr(active.watching_call, "__func__")
            else active.watching_call,
            before.__func__,
        )


if __name__ == "__main__":
    unittest.main()
