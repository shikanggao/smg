"""Opt-in native V1 scheduler cancellation evidence for vLLM 0.19.0.

This module does not import vLLM until its general-plugin entry point is enabled.
The HTTP middleware and scheduler coordinate the first visible text boundary in
a local SQLite transaction. An HTTP disconnect alone never creates an ACK.
"""

import functools
import hashlib
import json
import os
import sqlite3
import time
import uuid
from collections import Counter, deque
from contextlib import contextmanager
from pathlib import Path

VERSION = "0.19.0"
UPSTREAM_COMMIT = "2a69949bdadf0e8942b7a1619b229cb475beef20"
ATTEMPT_XARG = "smg_native_attempt_id"
BINDING_XARG = "smg_native_binding_token"
ACK_LIMIT = 4096
MAX_ATTEMPTS = 65536
SOURCE_HASHES = {
    "vllm.v1.core.sched.scheduler": "9f3da2dfce94963e1e1cefc156e4239120f79c6eaf824b2829cac0ee7736b58a",
    "vllm.v1.engine.core": "896730e749cbcabb487ce50c703974594d197fc31a1d3b26fe096197d142d2d5",
    "vllm.v1.core.sched.async_scheduler": "4df37e8842fb06d5cce3bc414e9423a09808e9336a96a25530dd2270f078d093",
    "vllm.v1.request": "86e5aae8fb5ad78163e085a69bfd7478e51fb639fcc965f7978c5f2dc07a6be6",
    "vllm.v1.engine.input_processor": "6e0db9a48cde52857c257f70c5562fa907828a86ecf9b6c3137093e18f19dfd0",
    "vllm.entrypoints.openai.server_utils": "1feff0399e2b723c51a5bbaff19fed017a01c0d9bcb6484fc7fd37a54e51c636",
    "vllm.entrypoints.openai.chat_completion.protocol": (
        "1a2492bdb99e944ca177853496408a2a00d578bb2fa51c6e49241567d6601a82"
    ),
    "vllm.entrypoints.openai.completion.protocol": "d529d72221c54beb3b2ddd54aa431b6bb6f9c7f0553db57768d8a1bf5f8d9b20",
    "vllm.entrypoints.openai.api_server": "cab836e021b1d8a41e1708be11070294b3b87685ffa15f88a140c1bbdd1039c3",
}


def _enabled():
    return os.environ.get("SMG_NATIVE_DEADLINE_ENABLED") == "1"


class NativeStore:
    """One host-local store shared by the API processes and scheduler ranks."""

    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(
                "CREATE TABLE IF NOT EXISTS ingress ("
                "attempt TEXT PRIMARY KEY, binding TEXT NOT NULL, native_id TEXT, "
                "deadline REAL, first_text INTEGER DEFAULT 0, "
                "expired INTEGER DEFAULT 0, activated INTEGER DEFAULT 0, reason TEXT DEFAULT 'explicit_abort');"
                "CREATE TABLE IF NOT EXISTS snapshots (rank INTEGER PRIMARY KEY, body TEXT);"
                "CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT);"
            )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=1.0)
        try:
            with db:
                yield db
        finally:
            db.close()

    def register(self, attempt, timeout_ms, now, binding="source-test"):
        deadline = None if timeout_ms is None else now + timeout_ms / 1000
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT COUNT(*) FROM ingress").fetchone()[0] >= MAX_ATTEMPTS:
                raise ValueError(
                    "native attempt retention capacity reached; restart the opt-in session"
                )
            db.execute(
                "INSERT INTO ingress(attempt,binding,deadline) VALUES (?,?,?)",
                (attempt, binding, deadline),
            )
        return deadline

    def activate(self, attempt, binding, native_id):
        if isinstance(attempt, str) and isinstance(binding, str):
            with self.connect() as db:
                changed = db.execute(
                    "UPDATE ingress SET activated=1,native_id=? WHERE attempt=? AND binding=? "
                    "AND (native_id IS NULL OR native_id=?)",
                    (native_id, attempt, binding, native_id),
                ).rowcount
                return attempt if changed else None
        return None

    def discard_reservation(self, attempt):
        # Invalid/auth-rejected HTTP work is never native accepted work. Keep
        # accepted attempt IDs for duplicate rejection throughout this session.
        with self.connect() as db:
            db.execute("DELETE FROM ingress WHERE attempt=? AND activated=0", (attempt,))

    def first_text(self, attempt, now=None):
        # Serialize the first text / deadline race across processes. The winner
        # is committed before either text is emitted or scheduler work aborted.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            now = time.monotonic() if now is None else now
            row = db.execute(
                "SELECT deadline,first_text,expired FROM ingress WHERE attempt=?", (attempt,)
            ).fetchone()
            if row is None:
                return False
            deadline, emitted, expired = row
            if emitted:
                return True
            if expired or (deadline is not None and now >= deadline):
                db.execute(
                    "UPDATE ingress SET expired=1,reason='hard_ttft_deadline' WHERE attempt=?",
                    (attempt,),
                )
                return False
            db.execute("UPDATE ingress SET first_text=1 WHERE attempt=?", (attempt,))
            return True

    def request_state(self, attempt):
        if attempt is None:
            return None
        with self.connect() as db:
            return db.execute(
                "SELECT deadline,first_text,expired,reason FROM ingress WHERE attempt=?",
                (attempt,),
            ).fetchone()

    def request_states(self, attempts):
        result = {}
        attempts = list(attempts)
        with self.connect() as db:
            for start in range(0, len(attempts), 500):
                batch = attempts[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = db.execute(
                    f"SELECT attempt,deadline,first_text,expired,reason FROM ingress WHERE attempt IN ({placeholders})",
                    batch,
                ).fetchall()
                result.update((row[0], row[1:]) for row in rows)
        return result

    def expire(self, attempts, now=None):
        expired = set()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            now = time.monotonic() if now is None else now
            for attempt in attempts:
                row = db.execute(
                    "SELECT deadline,first_text,expired FROM ingress WHERE attempt=?", (attempt,)
                ).fetchone()
                if (
                    row is not None
                    and not row[1]
                    and (row[2] or (row[0] is not None and now >= row[0]))
                ):
                    db.execute(
                        "UPDATE ingress SET expired=1,reason='hard_ttft_deadline' WHERE attempt=?",
                        (attempt,),
                    )
                    expired.add(attempt)
        return expired

    def disconnect(self, attempt):
        with self.connect() as db:
            db.execute(
                "UPDATE ingress SET reason='client_abort' WHERE attempt=? AND expired=0", (attempt,)
            )

    def publish(self, rank, body):
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO snapshots(rank,body) VALUES (?,?)", (rank, json.dumps(body))
            )

    def loads(self, now):
        with self.connect() as db:
            rows = db.execute("SELECT rank,body FROM snapshots ORDER BY rank").fetchall()
        result = []
        for rank, encoded in rows:
            body = json.loads(encoded)
            body["snapshot_age_seconds"] = max(0.0, now - body["captured_at_monotonic_secs"])
            active = body["active_requests"]
            result.append(
                {
                    "dp_rank": rank,
                    "num_running_reqs": sum(r["state"] != "waiting" for r in active),
                    "num_waiting_reqs": sum(r["state"] == "waiting" for r in active),
                    "deadline_recovery": body,
                }
            )
        return {"version": VERSION, "dp_rank_count": len(result), "loads": result}


def _store_from_env():
    path = os.environ.get("SMG_NATIVE_DEADLINE_DB")
    if not path:
        raise RuntimeError("SMG_NATIVE_DEADLINE_DB must identify a host-local SQLite file")
    return NativeStore(path)


def _attempt(request):
    return getattr(request, "_smg_native_attempt_id", None)


class SchedulerSignals:
    def __init__(self, scheduler, store, rank):
        self.scheduler, self.store, self.rank = scheduler, store, rank
        self.epoch = str(uuid.uuid4())
        self.sequence = 0
        self.inflight = {}
        self.cleanup_batches = {}
        self.ever_scheduled = set()
        self.cleanup_complete = set()
        self.retired = {}
        self.pending_acks = {}
        self.acks = deque()
        self.ack_floor = 0
        self.call_depth = 0
        config = getattr(scheduler, "vllm_config", None)
        parallel = scheduler.parallel_config
        self.supported = all(
            getattr(parallel, name, 1) == 1
            for name in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
            )
        ) and not any(
            getattr(config, name, None) is not None
            for name in ("kv_transfer_config", "ec_transfer_config", "speculative_config")
        )
        config_hash = config.compute_hash() if config is not None else "source-test"
        profile = {
            "upstream_commit": UPSTREAM_COMMIT,
            "engine_config": config_hash,
            "scheduler_config": str(getattr(scheduler, "scheduler_config", None)),
            "cache_config": str(getattr(scheduler, "cache_config", None)),
            "semantics": "continuous_batching:first_nonempty_text:v1",
        }
        self.profile = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()

    def next_seq(self):
        self.sequence += 1
        return self.sequence

    def expire(self):
        if not self.supported:
            return
        requests = self.scheduler.requests
        attempts = {a for r in requests.values() if (a := _attempt(r)) is not None}
        expired = self.store.expire(attempts)
        ids = [
            rid
            for rid, request in requests.items()
            if _attempt(request) in expired and not request.is_finished()
        ]
        # EngineCoreProc sends real finish outputs to unblock the native
        # frontend. Scheduler-only / offline use cannot advertise TTFT support.
        notify = getattr(self.scheduler, "_smg_abort_outputs", None)
        if ids and notify is not None:
            status = self.scheduler._smg_aborted_status
            notify(self.scheduler.finish_requests(ids, status))

    def freed(self, request):
        self.retired[request.request_id] = request
        if request.status != self.scheduler._smg_aborted_status:
            return
        attempt = _attempt(request)
        if attempt is None:
            return
        self.pending_acks[request.request_id] = attempt

    def publish(self):
        scheduler = self.scheduler
        inflight = Counter()
        for counts in self.inflight.values():
            inflight.update(counts)
        queued = {
            r.request_id
            for queue in (scheduler.running, scheduler.waiting, scheduler.skipped_waiting)
            for r in queue
        }
        inventory = {**self.retired, **scheduler.requests}
        ingress_states = self.store.request_states(
            {a for request in inventory.values() if (a := _attempt(request)) is not None}
        )
        released = {
            rid
            for rid in self.retired
            if rid not in scheduler.requests
            and rid not in queued
            and not inflight[rid]
            and (rid not in self.ever_scheduled or rid in self.cleanup_complete)
        }
        for rid, attempt in list(self.pending_acks.items()):
            # Cache release plus scheduler removal is insufficient while a
            # previously scheduled GPU batch still has this request in flight.
            if rid not in released:
                continue
            ingress = ingress_states.get(attempt)
            reason = ingress[3] if ingress else "explicit_abort"
            self.acks.append(
                {
                    "attempt_id": attempt,
                    "native_request_id": rid,
                    "event_seq": self.next_seq(),
                    "ack_monotonic_secs": time.monotonic(),
                    "reason": reason,
                }
            )
            del self.pending_acks[rid]
        for rid in released:
            del self.retired[rid]
            self.ever_scheduled.discard(rid)
            self.cleanup_complete.discard(rid)
        while len(self.acks) > ACK_LIMIT:
            self.ack_floor = self.acks.popleft()["event_seq"] + 1
        seq, now = self.next_seq(), time.monotonic()
        rows = []
        inventory = {**self.retired, **scheduler.requests}
        for rid, request in inventory.items():
            attempt = _attempt(request)
            ingress = ingress_states.get(attempt)
            first_text = bool(ingress[1]) if ingress is not None else None
            generated = request.num_output_tokens > 0
            # Cached-prefix matching is not known for a newly queued request.
            # A generated token with empty visible text also has unknown work
            # until first text: another decode may be required.
            work = None
            if first_text:
                work = 0.0
            elif not generated and request.num_cached_tokens >= 0:
                completed = max(0, request.num_computed_tokens - inflight[rid])
                if completed < request.num_prompt_tokens:
                    work = float(request.num_prompt_tokens - completed)
            if getattr(request, "num_preemptions", 0) > 0:
                work = None  # decoder recomputation can interfere with pending TTFT work
            state = (
                "decode"
                if generated
                else ("prefill" if request in scheduler.running else "waiting")
            )
            if request.is_finished():
                state = "unknown"  # e.g. delayed remote KV-release completion
            rows.append(
                {
                    "attempt_id": attempt,
                    "native_request_id": rid,
                    "state_seq": seq,
                    "remaining_prefill_work_tokens": work,
                    "hard_ttft_deadline_monotonic_secs": ingress[0] if ingress else None,
                    "queue_position": None,
                    "state": state,
                    "first_text_emitted": first_text,
                }
            )
        self.store.publish(
            self.rank,
            {
                "schema_version": 1,
                "engine_epoch": self.epoch,
                "profile_id": self.profile,
                "dp_rank": self.rank,
                "snapshot_seq": seq,
                "captured_at_monotonic_secs": now,
                "snapshot_age_seconds": 0.0,
                "active_inventory_complete": self.supported,
                "native_population": len(rows),
                "scheduler_semantics": "continuous_batching",
                "deadline_boundary": "first_nonempty_text",
                "deadline_origin": "engine_http_receive_monotonic",
                "remaining_work_source": "uncached_uncomputed_prompt_tokens",
                "effective_capacity_tokens_per_sec": None,
                "capacity_source": None,
                "hard_ttft_deadline_supported": self.supported
                and hasattr(scheduler, "_smg_abort_outputs"),
                "abort_enforcement_enabled": self.supported
                and hasattr(scheduler, "_smg_abort_outputs"),
                "abort_max_lag_seconds": None,
                "cancellation_ack_supported": self.supported,
                "ack_floor_seq": self.ack_floor,
                "active_requests": rows,
                "cancellation_acks": list(self.acks),
            },
        )


def install_scheduler_hooks(scheduler_cls, aborted_status, store_factory=_store_from_env):
    """Wrap real upstream methods; separate entry point enables source-level tests."""
    if getattr(scheduler_cls, "_smg_native_hooks_installed", False):
        return
    scheduler_cls._smg_native_hooks_installed = True
    original_init = scheduler_cls.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._smg_aborted_status = aborted_status
        rank = self.parallel_config.data_parallel_rank
        self._smg_signals = SchedulerSignals(self, store_factory(), rank)
        self._smg_signals.publish()

    scheduler_cls.__init__ = init
    for name in (
        "add_request",
        "schedule",
        "update_from_output",
        "finish_requests",
        "_free_blocks",
    ):
        original = getattr(scheduler_cls, name)

        def wrap(method, method_name):
            @functools.wraps(method)
            def wrapped(self, *args, **kwargs):
                signals = self._smg_signals
                signals.call_depth += 1
                try:
                    if method_name == "schedule":
                        signals.expire()
                    result = method(self, *args, **kwargs)
                    if method_name == "schedule":
                        signals.inflight[id(result)] = dict(result.num_scheduled_tokens)
                        signals.ever_scheduled.update(result.num_scheduled_tokens)
                        signals.cleanup_batches[id(result)] = set(result.finished_req_ids)
                    elif method_name == "update_from_output":
                        output = args[0] if args else kwargs["scheduler_output"]
                        signals.inflight.pop(id(output), None)
                        signals.cleanup_complete.update(signals.cleanup_batches.pop(id(output), ()))
                    elif method_name == "_free_blocks":
                        request = args[0] if args else kwargs["request"]
                        signals.freed(request)
                    elif method_name == "add_request":
                        request = args[0] if args else kwargs["request"]
                        if self.requests.get(request.request_id) is request:
                            params = getattr(request, "sampling_params", None)
                            extras = getattr(params, "extra_args", None) or {}
                            request._smg_native_attempt_id = signals.store.activate(
                                extras.get(ATTEMPT_XARG),
                                extras.get(BINDING_XARG),
                                request.request_id,
                            )
                finally:
                    signals.call_depth -= 1
                if signals.call_depth == 0:
                    signals.publish()
                return result

            return wrapped

        setattr(scheduler_cls, name, wrap(original, name))


def register():
    """vllm.general_plugins entry point; a no-op unless explicitly enabled."""
    if not _enabled():
        return
    from vllm.version import __version__

    if __version__ != VERSION:
        raise RuntimeError(f"SMG native deadline hooks require vLLM {VERSION}, got {__version__}")
    # A vendor image carrying the same version string is not enough evidence
    # of compatible private scheduler internals. Fail closed on source drift.
    import vllm
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.request import RequestStatus

    for name, expected in SOURCE_HASHES.items():
        source = Path(vllm.__file__).parent.joinpath(*name.split(".")[1:]).with_suffix(".py")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError(
                f"Unsupported vLLM scheduler source: {name}; expected {UPSTREAM_COMMIT}"
            )

    if getattr(EngineCore, "_smg_native_step_hooks_installed", False):
        return
    EngineCore._smg_native_step_hooks_installed = True

    install_scheduler_hooks(Scheduler, RequestStatus.FINISHED_ABORTED)
    for name in ("step", "step_with_batch_queue"):
        original = getattr(EngineCore, name)

        def wrap(method):
            @functools.wraps(method)
            def step(self, *args, **kwargs):
                if hasattr(self, "_send_abort_outputs"):
                    self.scheduler._smg_abort_outputs = self._send_abort_outputs
                return method(self, *args, **kwargs)

            return step

        setattr(EngineCore, name, wrap(original))


def _contains_text(frame):
    for line in frame.splitlines():
        if not line.startswith(b"data:"):
            continue
        try:
            obj = json.loads(line[5:].strip())
        except (ValueError, UnicodeDecodeError):
            continue
        for choice in obj.get("choices", []):
            delta = choice.get("delta") or {}
            if any(
                isinstance(value, str) and bool(value)
                for value in (
                    choice.get("text"),
                    delta.get("content"),
                    delta.get("reasoning_content"),
                    delta.get("reasoning"),
                )
            ):
                return True
    return False


class NativeDeadlineMiddleware:
    """Pass via vLLM ``--middleware`` together with the enabled general plugin."""

    def __init__(self, app):
        self.app = app
        self.store = _store_from_env() if _enabled() else None
        if self.store is not None:
            # Custom vLLM middleware is outside its API-key middleware. Put
            # the endpoint on the existing inner router so GET requests still
            # traverse native authentication (which protects /v1 paths only).
            from starlette.responses import JSONResponse
            from starlette.routing import Route

            router = app
            parent = None
            while not hasattr(router, "routes") and hasattr(router, "app"):
                parent = router
                router = router.app
            if not hasattr(router, "routes"):
                raise RuntimeError("Cannot find native vLLM router for authenticated telemetry")
            path = "/v1/native_deadline_snapshot"
            if any(getattr(route, "path", None) == path for route in router.routes):
                raise RuntimeError("Native deadline telemetry route already exists")

            async def snapshot(request):
                return JSONResponse(self.store.loads(time.monotonic()))

            router.routes.append(Route(path, snapshot, methods=["GET"]))
            self.request_app = router
            if parent is None:
                self.app = self._handle_request
            else:
                # Request preprocessing also belongs inside the authentication
                # middleware, so unauthorized requests create no reservations.
                parent.app = self._handle_request

    async def __call__(self, scope, receive, send):
        return await self.app(scope, receive, send)

    async def _handle_request(self, scope, receive, send):
        if self.store is None or scope["type"] != "http":
            return await self.request_app(scope, receive, send)
        path = scope.get("path")
        if path not in ("/v1/chat/completions", "/v1/completions") or scope["method"] != "POST":
            return await self.request_app(scope, receive, send)
        raw_headers = scope.get("headers", [])
        headers = dict(raw_headers)
        attempt_raw = headers.get(b"x-smg-attempt-id")
        ttl_raw = headers.get(b"x-smg-hard-ttft-timeout-ms")
        if attempt_raw is None:
            if ttl_raw is not None:
                return await self.respond(send, 400, {"error": "TTL requires x-smg-attempt-id"})
            return await self.request_app(scope, receive, send)
        try:
            if any(
                sum(k == header for k, _ in raw_headers) > 1
                for header in (b"x-smg-attempt-id", b"x-smg-hard-ttft-timeout-ms")
            ):
                raise ValueError("duplicate native deadline headers")
            attempt = attempt_raw.decode("ascii")
            if (
                not attempt
                or len(attempt) > 256
                or any(ord(c) < 33 or ord(c) > 126 for c in attempt)
            ):
                raise ValueError("invalid attempt ID")
            ttl = None if ttl_raw is None else int(ttl_raw)
            if ttl is not None and (ttl <= 0 or ttl > 2**64 - 1 or not ttl_raw.isdigit()):
                raise ValueError("invalid TTL")
            # TTL starts at native HTTP ingress, before body parsing/tokenizing.
            started = time.monotonic()
            chunks = []
            while True:
                event = await receive()
                if event["type"] == "http.disconnect":
                    return
                chunks.append(event.get("body", b""))
                if not event.get("more_body", False):
                    break
            raw_body = b"".join(chunks)
            try:
                body = json.loads(raw_body)
            except ValueError:
                if ttl is None:
                    return await self.replay(self.request_app, scope, receive, send, raw_body)
                raise
            if not isinstance(body, dict):
                if ttl is None:
                    return await self.replay(self.request_app, scope, receive, send, raw_body)
                raise ValueError("hard TTFT deadline requires a JSON object")
            if (
                body.get("n", 1) != 1
                or body.get("use_beam_search")
                or (path == "/v1/completions" and not isinstance(body.get("prompt"), str))
            ):
                if ttl is None:
                    return await self.replay(self.request_app, scope, receive, send, raw_body)
                raise ValueError("native deadline correlation requires a single generated request")
            if ttl is not None and body.get("stream") is not True:
                raise ValueError("hard TTFT deadline requires stream=true")
            if ttl is not None:
                snapshots = self.store.loads(time.monotonic())["loads"]
                if (
                    len(snapshots) != 1
                    or not snapshots[0]["deadline_recovery"]["cancellation_ack_supported"]
                ):
                    return await self.respond(
                        send, 503, {"error": "Supported native scheduler hooks are not initialized"}
                    )
            xargs = dict(body.get("vllm_xargs") or {})
            xargs[ATTEMPT_XARG] = attempt
            binding = str(uuid.uuid4())
            xargs[BINDING_XARG] = binding
            body["vllm_xargs"] = xargs
            self.store.register(attempt, ttl, started, binding)
            encoded = json.dumps(body).encode()
        except (ValueError, TypeError, UnicodeDecodeError, sqlite3.IntegrityError) as error:
            return await self.respond(send, 400, {"error": str(error)})
        scope = dict(scope)
        scope["headers"] = [(k, v) for k, v in scope["headers"] if k != b"content-length"] + [
            (b"content-length", str(len(encoded)).encode())
        ]
        delivered, seen_text, terminal = False, False, False
        buffered = b""

        async def wrapped_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": encoded, "more_body": False}
            event = await receive()
            if event["type"] == "http.disconnect":
                self.store.disconnect(attempt)
            return event

        async def wrapped_send(event):
            nonlocal buffered, seen_text, terminal
            if terminal:
                return
            if event["type"] != "http.response.body" or seen_text or not body.get("stream"):
                return await send(event)
            buffered += event.get("body", b"")
            if event.get("more_body", False) and b"\n\n" not in buffered:
                return
            if event.get("more_body", False):
                split = buffered.rfind(b"\n\n") + 2
                ready, buffered = buffered[:split], buffered[split:]
            else:
                ready, buffered = buffered, b""
            has_text = _contains_text(ready)
            expired = False
            if has_text:
                expired = not self.store.first_text(attempt)
            elif ttl is not None:
                # A real scheduler abort can end with only an empty/finish/DONE
                # frame. Report its committed expiry even when no text arrives.
                state = self.store.request_state(attempt)
                expired = state is not None and bool(state[2]) and not state[1]
            if expired:
                terminal = True
                error = (
                    b'data: {"error":{"message":"Native hard TTFT deadline exceeded",'
                    b'"type":"native_ttft_deadline"}}\n\ndata: [DONE]\n\n'
                )
                return await send({"type": "http.response.body", "body": error, "more_body": False})
            if has_text:
                seen_text = True
                ready, buffered = ready + buffered, b""
            await send({**event, "body": ready})

        try:
            await self.request_app(scope, wrapped_receive, wrapped_send)
        finally:
            self.store.discard_reservation(attempt)

    @staticmethod
    async def replay(app, scope, receive, send, body):
        delivered = False

        async def replay_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        return await app(scope, replay_receive, send)

    @staticmethod
    async def respond(send, status, body):
        encoded = json.dumps(body).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": encoded, "more_body": False})
