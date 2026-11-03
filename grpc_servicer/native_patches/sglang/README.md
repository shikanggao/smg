Opt-in native SGLang deadline signals
====================================

This source patch targets upstream commit
[`c9e5799bd12fd5fefb0fb375ac537a6b7d53892e`](https://github.com/sgl-project/sglang/commit/c9e5799bd12fd5fefb0fb375ac537a6b7d53892e).
The saved test runtime image used the tag `sglang:c9e5799`; the short tag resolves
to this official upstream commit. This is a source compatibility pin, not proof
of the contents of the historical container image. No image, GPU engine,
deployment, or production qualification is implied by the CPU tests.

The installer checks SHA-256 hashes of all twelve native source files it depends
on before writing anything. It parses the resulting Python and adds hooks to the
real Scheduler, TokenizerManager, output processor, and HTTP app. Source drift
fails closed. Run against a separate checkout of that exact upstream commit:

```sh
python grpc_servicer/native_patches/sglang/apply_patch.py --check /path/to/sglang
python grpc_servicer/native_patches/sglang/apply_patch.py /path/to/sglang
```

Both signal production and enforcement are **off by default**. With both off,
no methods or response streams are wrapped, the new endpoint is absent, and the
existing `/loads`, `/v1/loads`, and `/metrics` behavior is unchanged. Enabling
signals registers only `GET /v1/native_deadline_snapshot`, as an ordinary native
FastAPI route protected by the existing configured native API-key middleware.
The gateway must poll it separately from its raw load-monitor path.

Before starting the patched native server, set:

```sh
export SMG_NATIVE_DEADLINE_SIGNALS=1
export SMG_NATIVE_DEADLINE_DB=/absolute/local/path/dedicated-server-deadlines.sqlite3
# Optional, separate opt-in; has no relation to the admission threshold:
export SMG_NATIVE_HARD_TTFT_ENFORCEMENT=1
```

The SQLite file must be on a writable local filesystem visible to the HTTP and
scheduler processes on the **same host**. It uses WAL and atomic transactions;
no additional service is needed. Use a dedicated file per native server. Keep
it for the server's lifetime and remove it only after that server has stopped.
Completed attempts retain small tombstones to reject reuse of an attempt ID;
the database retains at most 65,536 accepted attempts per opt-in server session.
Further instrumented requests are rejected with an explicit retention-capacity
error until a new server session uses a fresh database. Unaccepted frontend
reservations are pruned when the HTTP handler ends.
The scheduler ACK journal is independently bounded at 4,096 events.

This first implementation supports a single tokenizer, TP=1, PP=1, DP=1,
standalone generation, ordinary normal/overlap scheduling, no PDMux, and no
asynchronous decode KV offload, speculative decoding, or hierarchical/storage
cache modes. Signal opt-in rejects multi-tokenizer startup,
which also avoids the upstream pin's unsupported multi-tokenizer API-key path.
Other unsupported scheduler topologies report incomplete inventory and false
cancellation/deadline capabilities; explicit TTLs are rejected. Instrumented
requests require one prompt, `stream=true`, `n=1`, and no session; native request
validation rejects other shapes before scheduler submission. Noninstrumented
native requests remain in the inventory with a null attempt ID.

The gateway supplies `x-smg-attempt-id` unchanged for every dispatch. Native
request IDs remain separate, exact strings. Optional
`x-smg-hard-ttft-timeout-ms` is a finite positive duration from **engine HTTP
receive**, measured on the host's monotonic clock; it is accepted only with
native enforcement enabled. The gateway may supply its remaining end-to-end
budget, but this producer does not claim zero network transit time or a
client-observed TTFT boundary.

The first-text boundary is server observation of the first complete SSE event
containing nonempty completion text, content, or reasoning text, before ASGI
send. Empty role events, empty text, and generated tokens without text do not
end the deadline. A local SQLite transaction orders this boundary against the
scheduler's expiry transaction. If text wins, later scheduler ticks cannot
expire the attempt. If expiry wins, HTTP suppresses subsequent text and the
scheduler invokes the exact-ID native abort path; HTTP returns an explicit
`native_ttft_deadline` SSE error and `[DONE]`. Committed expiry is also reported
for empty or finish-only terminal output, without creating an HTTP-side
cancellation ACK. Transaction timestamps are read after acquiring the database
write lock, so lock waits cannot backdate text. Sending can still fail after
the server boundary; this is not proof the client received text.

Running/chunked/overlap aborts use native `to_finish` and normal output/cache
cleanup. Already launched work is allowed to drain. The producer declares no
maximum cancellation lag. The cancellation ACK is appended only after actual
native KV release has returned (or a never-started waiting/grammar request had
no private allocation) **and every waiting, grammar, chunked, running, current,
last, and overlap-result reference is gone**. An HTTP 200 abort receipt,
frontend disconnect, request disappearance alone, or `to_finish` flag creates
no ACK. Aborting an exact native ID cannot cancel a prefix sibling.

Concurrent native snapshot and `/get_load` callers share one in-flight native
load fetch. Each caller receives an independent copy of the actual result,
with its original native capture time and measured response age. Cancelling
one HTTP waiter does not cancel that fetch or poison later polls. Exceptions
propagate; a missing native result is never converted to a healthy empty
inventory. This opt-in wrapper affects only the manager's load communicator.

Each native snapshot reports all scheduler owners, an engine boot UUID, rank,
source/config profile hash, monotonic sequences, capture age, nullable exact
remaining uncached/uncomputed prompt work, and the complete retained ACK
journal with an eviction floor. For waiting requests without an authoritative
cache match the remaining work is null, never a full-prompt proxy. Completed
prefill work advances only after native result processing, using the batch's
immutable `seq_lens_cpu`, including overlap copies. Decoders have zero remaining
prefill work only after observed first text; sampled-token-without-text work
remains unknown. Retracted requests report unknown remaining work because they must
recompute context and original-prompt progress no longer describes that work. Continuous batching is reported honestly; effective service
capacity and an abort-lag bound stay null, so this producer does not qualify a
single-FIFO or D3 forecast oracle.

CPU verification:

```sh
SGLANG_NATIVE_SOURCE=/path/to/exact/sglang \
SGLANG_NATIVE_COMMUNICATOR_SOURCE=/path/to/exact/tokenizer_communicator_mixin.py \
  python -m unittest discover -s grpc_servicer/tests \
  -p 'test_sglang_native_*.py' -v
```

The component tests use explicit scheduler/cache/request context doubles. The
pinned tests compile and execute actual upstream `Scheduler.abort_request`,
`process_input_requests`, `Req.check_finished`, `release_kv_cache`, and native
API-key middleware bodies via AST extraction, without importing CUDA. The
communicator tests additionally reproduce the exact pinned watching-mode race
and exercise the registered native endpoint with concurrent/cancelled callers. They
also apply the hooks to those real method bodies, exercise expiry/first-text
arbitration, and check post-removal ACKs. The source-free test run skips only
those pinned-method tests. A real engine GPU test remains required to qualify
streaming, allocator behavior, overlap lifetime, and runtime overhead together.
