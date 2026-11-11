"""Opt-in scheduler signals for SGLang c9e5799; installed by apply_patch.py.

No GPU dependencies. This module runs inside the native HTTP and scheduler
processes. SQLite arbitrates first text versus deadline expiry on the same host;
a frontend disconnect or successful abort submission never creates an ACK.
"""

import asyncio
import copy
import functools
import hashlib
import json
import math
import operator
import os
import re
import sqlite3
import time
import uuid
from collections import deque

ATTEMPT_HEADER = b"x-smg-attempt-id"
TIMEOUT_HEADER = b"x-smg-hard-ttft-timeout-ms"
MAX_ATTEMPTS = 65536
SUPPORTED_PATHS = {"/generate", "/v1/chat/completions", "/v1/completions"}


def enabled():
    return os.environ.get("SMG_NATIVE_DEADLINE_SIGNALS") == "1"


def enforcement_enabled():
    return enabled() and os.environ.get("SMG_NATIVE_HARD_TTFT_ENFORCEMENT") == "1"


def engine_work_enabled():
    return enabled() and os.environ.get("SMG_NATIVE_ENGINE_WORK_DIAGNOSTICS") == "1"


def unavailable_engine_work_state(reason, native_request_id=None, attempt_id=None):
    return dict(
        schema_version=1,
        diagnostic_only=True,
        native_request_id=native_request_id,
        attempt_id=attempt_id,
        available=False,
        unavailable_reason=reason,
        output_token_count=None,
        retraction_count=None,
        preemption_count=None,
        abort_pending=None,
        abort_pending_source="req_to_finish_FINISH_ABORT",
        cache_release_observed=None,
    )


def supported_args(args):
    return (
        getattr(args, "tp_size", 1) == 1
        and getattr(args, "pp_size", 1) == 1
        and getattr(args, "dp_size", 1) == 1
        and getattr(args, "tokenizer_worker_num", 1) == 1
        and not getattr(args, "enable_pdmux", False)
        and getattr(args, "speculative_algorithm", None) is None
        and not getattr(args, "enable_hierarchical_cache", False)
        and not getattr(args, "hicache_storage_backend", None)
        and getattr(args, "disaggregation_mode", "null") in (None, "null")
        and not getattr(args, "disaggregation_decode_enable_offload_kvcache", False)
    )


class TextDeadlineStore:
    """Local host only; never use this database on a network filesystem.

    Each attempt is single-use. BEGIN IMMEDIATE gives text emission and scheduler
    expiry a single order. The scheduler commits expiry before asking the native
    abort path to stop the request; HTTP suppresses later text for that attempt.
    """

    def __init__(self, path=None):
        self.path = path or os.environ["SMG_NATIVE_DEADLINE_DB"]
        self.db = sqlite3.connect(self.path, timeout=2, isolation_level=None)
        self.db.execute("PRAGMA busy_timeout=2000")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS attempts (attempt TEXT PRIMARY KEY, "
            "received REAL NOT NULL, deadline REAL, first_text REAL, "
            "expired INTEGER NOT NULL DEFAULT 0, retired INTEGER NOT NULL DEFAULT 0, "
            "activated INTEGER NOT NULL DEFAULT 0)"
        )

    def register(self, attempt, received, deadline):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] >= MAX_ATTEMPTS:
                raise ValueError(
                    "Native attempt retention capacity reached; restart the opt-in server session"
                )
            self.db.execute(
                "INSERT INTO attempts(attempt,received,deadline) VALUES(?,?,?)",
                (attempt, received, deadline),
            )
            self.db.execute("COMMIT")
        except BaseException as exc:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            if isinstance(exc, sqlite3.IntegrityError):
                raise ValueError("x-smg-attempt-id must be unique for every dispatch") from exc
            raise

    def activate(self, attempt):
        self.db.execute("UPDATE attempts SET activated=1 WHERE attempt=?", (attempt,))

    def discard_reservation(self, attempt):
        self.db.execute("DELETE FROM attempts WHERE attempt=? AND activated=0", (attempt,))

    def observe_text(self, attempt, now=None):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            now = time.monotonic() if now is None else now
            row = self.db.execute(
                "SELECT deadline,first_text,expired FROM attempts WHERE attempt=?", (attempt,)
            ).fetchone()
            if row is None:
                raise ValueError("Unregistered SMG attempt")
            deadline, first_text, expired = row
            if expired or (first_text is None and deadline is not None and now >= deadline):
                self.db.execute("UPDATE attempts SET expired=1 WHERE attempt=?", (attempt,))
                self.db.execute("COMMIT")
                return False
            self.db.execute(
                "UPDATE attempts SET first_text=COALESCE(first_text,?) WHERE attempt=?",
                (now, attempt),
            )
            self.db.execute("COMMIT")
            return True
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def claim_expired(self, attempt, now=None):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            now = time.monotonic() if now is None else now
            row = self.db.execute(
                "SELECT deadline,first_text,expired FROM attempts WHERE attempt=?", (attempt,)
            ).fetchone()
            expired = bool(row and row[1] is None and row[0] is not None and now >= row[0])
            if expired:
                self.db.execute("UPDATE attempts SET expired=1 WHERE attempt=?", (attempt,))
            self.db.execute("COMMIT")
            return expired
        except BaseException:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def first_text(self, attempt):
        row = self.db.execute(
            "SELECT first_text FROM attempts WHERE attempt=?", (attempt,)
        ).fetchone()
        return bool(row and row[0] is not None)

    def expired_without_text(self, attempt):
        # Read a committed expiry; HTTP must not fabricate native release or
        # turn an unrelated validation failure into a deadline cancellation.
        row = self.db.execute(
            "SELECT expired,first_text FROM attempts WHERE attempt=?", (attempt,)
        ).fetchone()
        return bool(row and row[0] and row[1] is None)

    def discard(self, attempt):
        self.db.execute("UPDATE attempts SET retired=1 WHERE attempt=?", (attempt,))


def contains_text(payload):
    if not isinstance(payload, dict):
        return False
    if isinstance(payload.get("text"), str) and payload["text"]:
        return True
    for choice in payload.get("choices", []):
        if not isinstance(choice, dict):
            continue
        if isinstance(choice.get("text"), str) and choice["text"]:
            return True
        for key in ("delta", "message"):
            value = choice.get(key) or {}
            if isinstance(value, dict) and any(
                isinstance(value.get(k), str) and value[k]
                for k in ("content", "reasoning_content", "reasoning")
            ):
                return True
    return False


class SSETextObserver:
    """Incremental SSE parser; observes bytes without changing accepted chunks."""

    def __init__(self):
        self.buffer = b""

    def feed(self, chunk):
        self.buffer += chunk
        if len(self.buffer) > 1024 * 1024:
            raise ValueError("SMG deadline observer SSE frame exceeds 1 MiB")
        # Both legal SSE line endings are accepted.
        self.buffer = self.buffer.replace(b"\r\n", b"\n")
        seen = False
        while b"\n\n" in self.buffer:
            frame, self.buffer = self.buffer.split(b"\n\n", 1)
            data = b"\n".join(
                line[5:].lstrip(b" ") for line in frame.split(b"\n") if line.startswith(b"data:")
            )
            if not data or data == b"[DONE]":
                continue
            try:
                seen = contains_text(json.loads(data)) or seen
            except (ValueError, UnicodeDecodeError):
                continue
        return seen


class DeadlineHTTPMiddleware:
    """Native ASGI receive timestamp and first nonempty HTTP text boundary."""

    def __init__(self, app, manager_getter):
        self.app = app
        self.manager_getter = manager_getter
        self.store = TextDeadlineStore()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in SUPPORTED_PATHS:
            return await self.app(scope, receive, send)
        received = time.monotonic()
        raw_headers = scope.get("headers", [])
        headers = dict(raw_headers)
        attempt_raw = headers.get(ATTEMPT_HEADER)
        timeout_raw = headers.get(TIMEOUT_HEADER)
        if attempt_raw is None and timeout_raw is None:
            return await self.app(scope, receive, send)
        try:
            if any(
                sum(key == header for key, _ in raw_headers) > 1
                for header in (ATTEMPT_HEADER, TIMEOUT_HEADER)
            ):
                raise ValueError("Duplicate native deadline headers are not supported")
            if attempt_raw is None:
                raise ValueError("Hard TTFT timeout requires x-smg-attempt-id")
            attempt = attempt_raw.decode("ascii")
            if (
                not attempt
                or len(attempt) > 256
                or any(ord(c) < 33 or ord(c) > 126 for c in attempt)
            ):
                raise ValueError("x-smg-attempt-id must be 1..256 opaque printable ASCII bytes")
            timeout = None
            if timeout_raw is not None:
                if not enforcement_enabled():
                    raise ValueError("Native hard TTFT enforcement is disabled")
                if not supported_args(self.manager_getter().server_args):
                    raise ValueError(
                        "Native hard TTFT deadline does not support this scheduler topology"
                    )
                timeout = float(timeout_raw)
                if not math.isfinite(timeout) or timeout <= 0:
                    raise ValueError("Hard TTFT timeout must be finite and positive")
            deadline = None if timeout is None else received + timeout / 1000
            self.store.register(attempt, received, deadline)
            scope["smg_deadline"] = (attempt, deadline)
        except (ValueError, UnicodeError) as exc:
            await send(
                {
                    "type": "http.response.start",
                    "status": 400,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {"type": "http.response.body", "body": json.dumps({"error": str(exc)}).encode()}
            )
            return
        observer = SSETextObserver()
        first_text = False
        terminal = False
        buffered = b""

        async def observe_send(message):
            nonlocal first_text, terminal, buffered
            if terminal:
                return
            if message["type"] != "http.response.body" or first_text:
                return await send(message)
            buffered += message.get("body", b"")
            boundaries = list(re.finditer(rb"\r?\n\r?\n", buffered))
            if not boundaries and message.get("more_body", False):
                if len(buffered) > 1024 * 1024:
                    raise ValueError("SMG deadline observer SSE frame exceeds 1 MiB")
                return
            split = boundaries[-1].end() if message.get("more_body", False) else len(buffered)
            ready, buffered = buffered[:split], buffered[split:]
            has_text = observer.feed(ready)
            expired = self.store.expired_without_text(attempt)
            if has_text and not expired:
                first_text = self.store.observe_text(attempt)
                expired = not first_text
            # A native abort may finish with no text at all. Surface its
            # committed deadline outcome even for empty/finish-only bodies.
            if expired:
                terminal = True
                error = (
                    b'data: {"error":{"message":"Native hard TTFT deadline exceeded",'
                    b'"type":"native_ttft_deadline"}}\n\ndata: [DONE]\n\n'
                )
                return await send({"type": "http.response.body", "body": error, "more_body": False})
            if has_text:
                ready, buffered = ready + buffered, b""
            await send({**message, "body": ready})

        try:
            await self.app(scope, receive, observe_send)
        finally:
            self.store.discard_reservation(attempt)


def scheduler_inventory(scheduler):
    """All pinned standalone scheduler owners, deduplicated by object identity.

    Finished requests remain in the inventory while a native batch still owns
    a reference, including previously launched overlap work awaiting completion.
    """
    refs = {}

    def add(req, state):
        if req is not None:
            refs.setdefault(id(req), (req, state))

    for req in scheduler.waiting_queue:
        add(req, "waiting")
    for req in scheduler.grammar_queue:
        add(req, "waiting")
    add(scheduler.chunked_req, "prefill")
    for name in ("running_batch", "cur_batch", "last_batch"):
        batch = getattr(scheduler, name, None)
        for req in getattr(batch, "reqs", ()):
            add(req, "decode" if req.output_ids else "prefill")
    for batch, _ in getattr(scheduler, "result_queue", ()):
        for req in batch.reqs:
            add(req, "decode" if req.output_ids else "prefill")
    return refs


class SchedulerDeadlineState:
    def __init__(self, scheduler, journal_limit=4096, store=None):
        self.scheduler = scheduler
        self.epoch = str(uuid.uuid4())
        self.seq = 0
        self.floor = 0
        self.journal = deque()
        self.journal_limit = journal_limit
        self.tracked = {}
        self.incoming = {}
        self.store = store or TextDeadlineStore()
        args = scheduler.server_args
        mode = getattr(scheduler.disaggregation_mode, "value", scheduler.disaggregation_mode)
        # These modes have additional queue owners/cache release paths. Fail
        # closed rather than claim a complete inventory or usable cancellation.
        self.supported = (
            supported_args(args) and mode in (None, "null") and not scheduler.enable_pdmux
        )
        profile = {
            key: getattr(args, key, None)
            for key in (
                "model_path",
                "tp_size",
                "pp_size",
                "dp_size",
                "chunked_prefill_size",
                "schedule_policy",
                "speculative_algorithm",
                "enable_hierarchical_cache",
                "hicache_storage_backend",
                "tokenizer_worker_num",
                "page_size",
                "disable_radix_cache",
                "disable_overlap_schedule",
                "disaggregation_mode",
                "enable_pdmux",
            )
        }
        profile["source_commit"] = "c9e5799bd12fd5fefb0fb375ac537a6b7d53892e"
        profile["signals_version"] = 1
        profile["hard_ttft_enforcement"] = enforcement_enabled()
        if engine_work_enabled():
            profile["engine_work_diagnostics_version"] = 1
        self.profile_id = (
            "sglang-c9e5799:"
            + hashlib.sha256(json.dumps(profile, sort_keys=True, default=str).encode()).hexdigest()
        )
        self.enforce = enforcement_enabled() and self.supported
        self.engine_work = None
        self.engine_work_supported = False
        if engine_work_enabled():
            from sglang.srt.managers.smg_engine_work_progress import CompletedWorkProgress

            model = getattr(scheduler, "model_config", None)
            work_supported = (
                self.supported
                and scheduler.dp_rank in (None, 0)
                and getattr(scheduler, "is_generation", False)
                and not getattr(model, "is_encoder_decoder", True)
                and not getattr(model, "is_multimodal", True)
            )
            self.engine_work = CompletedWorkProgress(
                self.epoch,
                self.profile_id,
                scheduler.dp_rank or 0,
                "c9e5799bd12fd5fefb0fb375ac537a6b7d53892e",
                "normal_process_batch_result_return_after_native_completion_wait",
                supported=work_supported,
            )
            self.engine_work_supported = work_supported

    def begin_engine_work(self, batch):
        if self.engine_work is None or not self.engine_work_supported:
            return None
        try:
            mode = batch.forward_mode
            if mode.is_idle():
                return None
            if not (mode.is_decode() or mode.is_extend()):
                raise ValueError("unsupported forward mode")
            reqs = tuple(batch.reqs)
            # Reading only CPU scheduling metadata adds no device synchronization.
            cpu = batch.seq_lens_cpu
            if getattr(getattr(cpu, "device", None), "type", "cpu") != "cpu":
                raise ValueError("sequence lengths are not CPU metadata")
            ends = tuple(operator.index(end) for end in cpu)
            if len(ends) != len(reqs):
                raise ValueError("incomplete sequence lengths")
            decoding = {id(req) for req in (batch.decoding_reqs or [])}
            if not decoding.issubset({id(req) for req in reqs}):
                raise ValueError("mixed decode membership mismatch")
            prefixes = tuple(batch.prefix_lens) if mode.is_extend() else ()
            if mode.is_extend() and len(prefixes) != len(reqs):
                raise ValueError("incomplete prefix lengths")
            rows = tuple(
                (
                    end - 1 if mode.is_decode() or id(req) in decoding else operator.index(prefixes[i]),
                    end,
                    len(req.origin_input_ids),
                )
                for i, (req, end) in enumerate(zip(reqs, ends))
            )
            return self.observe_engine_work("begin", rows)
        except Exception:
            self.engine_work.invalidate("unsupported_or_incomplete_batch_work_metadata")
            return None

    def observe_engine_work(self, method, *args):
        if self.engine_work is None:
            return None
        try:
            return getattr(self.engine_work, method)(*args)
        except Exception:
            self.engine_work.invalidate("engine_work_diagnostic_observation_failed")
            return None

    def request_engine_work_state(self, req):
        native_id = getattr(req, "rid", None)
        attempt = getattr(req, "smg_attempt_id", None)
        if not self.engine_work_supported:
            return unavailable_engine_work_state(
                "unsupported_engine_configuration", native_id, attempt
            )
        output = unavailable_engine_work_state(
            "request_work_metadata_missing_or_invalid", native_id, attempt
        )
        ids = getattr(req, "output_ids", None)
        if isinstance(ids, list):
            output["output_token_count"] = len(ids)
        retractions = getattr(req, "retraction_count", None)
        if type(retractions) is int and retractions >= 0:
            output["retraction_count"] = retractions
        missing = object()
        finish = getattr(req, "to_finish", missing)
        if finish is not missing:
            output["abort_pending"] = type(finish).__name__ == "FINISH_ABORT"
        # The existing release observer creates this flag only after returning.
        # Absence means no release callback observed, not no private allocation.
        released = getattr(req, "_smg_cache_released", False)
        if type(released) is bool:
            output["cache_release_observed"] = released
        output["available"] = (
            isinstance(native_id, str)
            and bool(native_id)
            and (attempt is None or isinstance(attempt, str) and bool(attempt))
            and all(
                output[field] is not None
                for field in (
                    "output_token_count", "retraction_count", "abort_pending", "cache_release_observed"
                )
            )
        )
        if output["available"]:
            output["unavailable_reason"] = None
        return output

    def next_seq(self):
        self.seq += 1
        return self.seq

    def remember(self, recv_req):
        self.incoming[recv_req.rid] = (
            getattr(recv_req, "smg_attempt_id", None),
            getattr(recv_req, "smg_deadline", None),
        )

    def reconcile(self):
        refs = scheduler_inventory(self.scheduler)
        active = {}
        for key, (req, state) in refs.items():
            if req.rid in self.incoming:
                req.smg_attempt_id, req.smg_deadline = self.incoming.pop(req.rid)
                if req.smg_attempt_id:
                    self.store.activate(req.smg_attempt_id)
            if key not in self.tracked:
                self.tracked[key] = [req, self.next_seq(), None]
            active[key] = (req, "unknown" if req.finished() else state)
        for key, (req, _, reason) in list(self.tracked.items()):
            if key in refs:
                continue
            if reason and getattr(req, "_smg_cache_released", False):
                attempt = getattr(req, "smg_attempt_id", None)
                if attempt:
                    event_seq = self.next_seq()
                    self.journal.append(
                        dict(
                            attempt_id=attempt,
                            native_request_id=req.rid,
                            event_seq=event_seq,
                            ack_monotonic_secs=time.monotonic(),
                            reason=reason,
                        )
                    )
                    if len(self.journal) > self.journal_limit:
                        self.floor = self.journal.popleft()["event_seq"] + 1
            attempt = getattr(req, "smg_attempt_id", None)
            if attempt:
                self.store.discard(attempt)
            del self.tracked[key]
        return active

    def mark_abort(self, recv_req):
        refs = scheduler_inventory(self.scheduler)
        for key, (req, _) in refs.items():
            if req.finished():
                continue
            exact = getattr(recv_req, "smg_exact", False)
            if recv_req.abort_all or (
                req.rid == recv_req.rid if exact else req.rid.startswith(recv_req.rid)
            ):
                tracked = self.tracked.setdefault(key, [req, self.next_seq(), None])
                tracked[2] = tracked[2] or getattr(recv_req, "smg_reason", "explicit_abort")
                # Waiting standalone requests own no KV allocation; actual native
                # abort must still remove them before an ACK can be appended.
                if (
                    (req in self.scheduler.waiting_queue or req in self.scheduler.grammar_queue)
                    and req.req_pool_idx is None
                    and req.mamba_pool_idx is None
                ):
                    req._smg_cache_released = True

    def tick(self):
        from sglang.srt.managers.io_struct import AbortReq

        active = self.reconcile()
        if not self.enforce:
            return
        for req, _ in active.values():
            attempt = getattr(req, "smg_attempt_id", None)
            if attempt and not req.finished() and self.store.claim_expired(attempt):
                pending = self.tracked[id(req)]
                if pending[2] is None:
                    abort = AbortReq(rid=req.rid)
                    abort.smg_exact = True
                    abort.smg_reason = "hard_ttft_deadline"
                    self.scheduler.abort_request(abort)
        self.reconcile()

    def snapshot(self):
        active = self.reconcile()
        rows = []
        for key, (req, state) in active.items():
            attempt = getattr(req, "smg_attempt_id", None)
            prompt_len = len(req.origin_input_ids)
            completed = getattr(req, "_smg_prefill_completed", None)
            cached = getattr(req, "_smg_matched_prefix", None)
            remaining = (
                None
                if completed is None and cached is None
                else max(0, prompt_len - max(completed or 0, cached or 0))
            )
            first_text = self.store.first_text(attempt) if attempt else None
            if req.output_ids:
                remaining = 0 if first_text else None
            # A retracted decoder must recompute prompt/output context. The
            # original-prompt-only progress markers do not describe that work.
            if getattr(req, "retraction_count", 0) > 0:
                remaining = None
            row = dict(
                attempt_id=attempt,
                native_request_id=req.rid,
                state_seq=self.tracked[key][1],
                remaining_prefill_work_tokens=remaining,
                hard_ttft_deadline_monotonic_secs=getattr(req, "smg_deadline", None)
                if self.enforce
                else None,
                queue_position=None,
                state=state,
                first_text_emitted=first_text,
            )
            if self.engine_work is not None:
                try:
                    diagnostic = self.request_engine_work_state(req)
                    if not isinstance(diagnostic, dict):
                        raise ValueError("invalid request diagnostic record")
                    row["engine_work_state"] = diagnostic
                except Exception:
                    row["engine_work_state"] = unavailable_engine_work_state(
                        "request_work_metadata_observation_failed",
                        row["native_request_id"],
                        row["attempt_id"],
                    )
            rows.append(row)
        seq = self.next_seq()
        output = dict(
            schema_version=1,
            engine_epoch=self.epoch,
            profile_id=self.profile_id,
            dp_rank=self.scheduler.dp_rank or 0,
            snapshot_seq=seq,
            captured_at_monotonic_secs=time.monotonic(),
            snapshot_age_seconds=0.0,
            active_inventory_complete=self.supported,
            native_population=len(rows),
            scheduler_semantics="continuous_batching",
            remaining_work_source="uncached_uncomputed_prompt_tokens"
            if all(r["remaining_prefill_work_tokens"] is not None for r in rows)
            else "unknown",
            effective_capacity_tokens_per_sec=None,
            capacity_source=None,
            cancellation_ack_supported=self.supported,
            hard_ttft_deadline_supported=self.supported,
            abort_enforcement_enabled=self.enforce,
            abort_max_lag_seconds=None,
            deadline_boundary="first_nonempty_text" if self.supported else "unsupported",
            deadline_origin="engine_http_receive_monotonic",
            ack_floor_seq=self.floor,
            active_requests=rows,
            cancellation_acks=list(self.journal),
        )
        if self.engine_work is not None:
            output["engine_work_progress"] = self.observe_engine_work("snapshot") or dict(
                schema_version=1,
                engine_epoch=self.epoch,
                profile_id=self.profile_id,
                dp_rank=self.scheduler.dp_rank or 0,
                counters_available=False,
                unavailable_reason="engine_work_diagnostic_observation_failed",
            )
        return output


def install_scheduler(cls):
    if not enabled():
        return
    original_init = cls.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._smg_deadline = SchedulerDeadlineState(self)

    cls.__init__ = init

    original_generate = cls.handle_generate_request

    def generate(self, recv_req):
        self._smg_deadline.remember(recv_req)
        result = original_generate(self, recv_req)
        self._smg_deadline.reconcile()
        self._smg_deadline.incoming.pop(recv_req.rid, None)
        return result

    cls.handle_generate_request = generate

    original_abort = cls.abort_request

    def abort(self, recv_req):
        self._smg_deadline.mark_abort(recv_req)
        result = original_abort(self, recv_req)
        # The pin's abort_request only walks running_batch/cur_batch. A chunked
        # request or overlap result can still be owned elsewhere; mark the same
        # native finish reason and let its normal result/cache cleanup run.
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        for req, _ in scheduler_inventory(self).values():
            exact = getattr(recv_req, "smg_exact", False)
            matches = recv_req.abort_all or (
                req.rid == recv_req.rid if exact else req.rid.startswith(recv_req.rid)
            )
            if matches and not req.finished():
                req.to_finish = FINISH_ABORT()
        self._smg_deadline.reconcile()
        return result

    cls.abort_request = abort

    original_input = cls.process_input_requests

    def process_input(self, recv_reqs):
        result = original_input(self, recv_reqs)
        self._smg_deadline.tick()
        return result

    cls.process_input_requests = process_input

    original_load = cls.get_load

    def get_load(self, recv_req=None):
        result = original_load(self, recv_req)
        result.deadline_recovery = self._smg_deadline.snapshot()
        return result

    cls.get_load = get_load

    original_run = cls.run_batch

    def run(self, batch):
        work_record = self._smg_deadline.begin_engine_work(batch)
        if batch.forward_mode.is_extend():
            for req, prefix in zip(batch.reqs, batch.prefix_lens):
                req._smg_matched_prefix = max(getattr(req, "_smg_matched_prefix", 0), prefix)
        try:
            result = original_run(self, batch)
        except Exception:
            if work_record is not None:
                self._smg_deadline.engine_work.invalidate("native_batch_execution_failed")
            raise
        if work_record is not None:
            # The result object survives overlap's ScheduleBatch.copy; the
            # mutable batch and Req fields do not define this completed work.
            try:
                result._smg_engine_work_record = work_record
            except Exception:
                self._smg_deadline.engine_work.invalidate("result_work_metadata_not_retained")
        return result

    cls.run_batch = run

    original_result = cls.process_batch_result

    def process_result(self, batch, result):
        # seq_lens_cpu survives ScheduleBatch.copy at this pin; it is the batch's
        # immutable committed prefix+chunk end, unlike mutable Req.fill_ids.
        ends = list(zip(batch.reqs, batch.seq_lens_cpu)) if batch.forward_mode.is_extend() else []
        try:
            work_record = getattr(result, "_smg_engine_work_record", None)
        except Exception:
            work_record = None
            if self._smg_deadline.engine_work is not None:
                self._smg_deadline.engine_work.invalidate("result_work_metadata_not_retained")
        try:
            output = original_result(self, batch, result)
        except Exception:
            if self._smg_deadline.engine_work is not None and work_record is not None:
                self._smg_deadline.engine_work.invalidate("native_result_processing_failed")
            raise
        if self._smg_deadline.engine_work is not None and work_record is not None:
            # The pinned normal output processor has already waited for its
            # result/copy completion. Do not add another GPU synchronization.
            self._smg_deadline.observe_engine_work("complete", work_record)
        elif self._smg_deadline.engine_work_supported and (
            batch.forward_mode.is_extend() or batch.forward_mode.is_decode()
        ):
            self._smg_deadline.engine_work.invalidate("completion_without_submission_record")
        for req, end in ends:
            req._smg_prefill_completed = max(
                getattr(req, "_smg_prefill_completed", 0), min(len(req.origin_input_ids), int(end))
            )
        self._smg_deadline.reconcile()
        return output

    cls.process_batch_result = process_result


def release_observer(original):
    if not enabled():
        return original

    @functools.wraps(original)
    def release(req, *args, **kwargs):
        result = original(req, *args, **kwargs)
        req._smg_cache_released = True
        return result

    return release


def install_tokenizer(cls):
    if not enabled():
        return
    original_init = cls.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if getattr(self.server_args, "tokenizer_worker_num", 1) != 1:
            raise ValueError(
                "SMG native deadline signals require the native single-tokenizer authentication path"
            )
        # The pin's watching_call clears shared results when its first waiter
        # resumes, so a concurrent native poll can receive None. Run it once
        # per in-flight scheduler fetch, with independent response objects.
        communicator = self.get_load_communicator
        original_watching = communicator.watching_call
        in_flight = None

        async def watching(obj):
            nonlocal in_flight
            if in_flight is None or in_flight.done():
                in_flight = asyncio.create_task(original_watching(obj))
            # An HTTP waiter disconnect must not cancel the shared native fetch.
            return copy.deepcopy(await asyncio.shield(in_flight))

        communicator.watching_call = watching

    cls.__init__ = init
    original_generate = cls.generate_request

    async def generate(self, obj, request=None):
        metadata = request.scope.get("smg_deadline") if request is not None else None
        if metadata:
            # One gateway attempt must map to exactly one native request.
            sampling = obj.sampling_params or {}
            n = sampling.get("n", 1) if isinstance(sampling, dict) else 0
            single = isinstance(getattr(obj, "text", None), str) or (
                isinstance(getattr(obj, "input_ids", None), list)
                and bool(obj.input_ids)
                and isinstance(obj.input_ids[0], int)
            )
            if not single or not obj.stream or n != 1 or getattr(obj, "session_params", None):
                raise ValueError(
                    "SMG native attempt contract requires stream=true, n=1, one prompt, no session"
                )
            obj.smg_attempt_id, obj.smg_deadline = metadata
        async for output in original_generate(self, obj, request):
            yield output

    cls.generate_request = generate

    original_send = cls._send_one_request

    def send(self, obj, tokenized_obj, created_time):
        tokenized_obj.smg_attempt_id = getattr(obj, "smg_attempt_id", None)
        tokenized_obj.smg_deadline = getattr(obj, "smg_deadline", None)
        return original_send(self, obj, tokenized_obj, created_time)

    cls._send_one_request = send


def install_http(app, manager_getter):
    if not enabled():
        return
    app.add_middleware(DeadlineHTTPMiddleware, manager_getter=manager_getter)

    async def loads():
        from sglang.srt.managers.io_struct import GetLoadReqInput

        manager = manager_getter()
        manager.auto_create_handle_loop()
        values = await manager.get_load_communicator(GetLoadReqInput())
        now = time.monotonic()
        output = []
        for value in values:
            envelope = getattr(value, "deadline_recovery", None)
            row = dict(dp_rank=value.dp_rank or 0)
            if envelope is not None:
                envelope["snapshot_age_seconds"] = max(
                    0, now - envelope["captured_at_monotonic_secs"]
                )
                row["deadline_recovery"] = envelope
                waiting = sum(req["state"] == "waiting" for req in envelope["active_requests"])
                row.update(
                    num_waiting_reqs=waiting,
                    num_running_reqs=envelope["native_population"] - waiting,
                    num_total_reqs=envelope["native_population"],
                )
            output.append(row)
        return dict(
            timestamp="",
            version="sglang-c9e5799-native-deadline-v1",
            dp_rank_count=len(output),
            loads=output,
        )

    app.add_api_route("/v1/native_deadline_snapshot", loads, methods=["GET"])
