# Native vLLM cancellation and TTFT evidence

The opt-in adapter supports native OpenAI HTTP serving on **vLLM 0.19.0**,
upstream commit `2a69949bdadf0e8942b7a1619b229cb475beef20`. It is independent
of the SMG gRPC generation service. The prior Qwen test pod provenance names
`fra.ocir.io/idqj093njucb/vllm-openai:v0.19.0` with image digest
`sha256:7a0f0fdd2771464b6976625c2b2d5dd46f566aa00fbc53eceab86ef50883da90`.
That tag does not prove byte-identical source: startup checks actual scheduler,
engine, request/conversion, HTTP protocol, and authentication source hashes and
rejects drift.

Install `grpc_servicer` in the engine image, retain any existing vLLM plugin
selection, and explicitly enable the general plugin and middleware:

```sh
export SMG_NATIVE_DEADLINE_ENABLED=1
export SMG_NATIVE_DEADLINE_DB=/run/smg-native-deadline/session.sqlite3
# If VLLM_PLUGINS is restricted, append smg_native_deadline to that selection.
vllm serve MODEL --middleware smg_grpc_servicer.native_deadline_vllm.NativeDeadlineMiddleware
```

Use a fresh host-local database per engine launch, shared by its API processes
and scheduler ranks. This adapter does not support API and engine processes on
different hosts, remote filesystems, or independent workers sharing one file.
Supported scope is TP=1, PP=1, DP=1 with no context parallelism, speculative
decoder, KV-transfer connector, or encoder-transfer connector. Other topologies
advertise incomplete/unsupported telemetry and reject opt-in hard deadlines.
With the environment flag absent, the plugin imports no vLLM classes, installs
no hooks, and adds no route. The middleware is also a pass-through when disabled.

The existing API-key middleware protects the new
`GET /v1/native_deadline_snapshot` route. The adapter registers the route on the
existing inner router; it does not respond outside authentication. The route
contains a per-rank `deadline_recovery` envelope for the gateway's separate
shadow poll. Existing `/v1/loads`, `/load`, `/metrics`, raw admission and the
300-second evaluation threshold retain their existing behavior.

The gateway sends a unique opaque `x-smg-attempt-id` for correlation. To opt in
to cancellation, it must separately send an explicit positive integer
`x-smg-hard-ttft-timeout-ms`. The timeout starts at native HTTP ingress. There is
no default timeout and no conversion from the admission/evaluation threshold.
Hard TTFT deadlines support streaming, `n=1`, non-beam chat completions and
single-string text completions. Other deadline shapes are rejected explicitly.
Correlation-only unsupported fanout shapes continue through the original
endpoint without native per-attempt attribution.

The middleware propagates the attempt and a private random binding token through
native `vllm_xargs`, which vLLM carries into `SamplingParams.extra_args` and the
real scheduler `Request`. Binding succeeds only for the exact native request ID
that first claims that reservation. A user-supplied xarg cannot claim another
attempt. Repeated HTTP attempt IDs are rejected. Unauthorized HTTP work creates
no reservation; native validation failures discard unaccepted reservations and
create no inventory row or ACK. Accepted IDs remain retained for duplicate
rejection; this experimental session accepts at most 65,536 retained attempts
and fails closed at that limit rather than silently recycling identities.

First nonempty OpenAI SSE `content`, `reasoning_content`, `reasoning`, or text
completion `text` commits the first-text transition before ASGI emission. Role
chunks, empty content and token IDs alone do not count. First text and expiry
use short SQLite transactions on the same native monotonic clock. If expiry
wins, the HTTP path emits an explicit `native_ttft_deadline` SSE error followed
by one `[DONE]`, including when native abort produces only empty or finish-only
output. It never emits the losing healthy text. This HTTP error does not create
a cancellation ACK. If first text wins, later scheduler ticks cannot cancel that
request for TTFT. No transaction spans model execution, an RPC, or an await.

The real scheduler checks expiry before its next scheduling step, calls native
`finish_requests(FINISHED_ABORTED)`, and sends native abort outputs to unblock
the API client. A cancellation ACK requires all of the following:

- The actual `_free_blocks` path has released request KV blocks; the upstream
  `_free_request` path has already released encoder cache references.
- The request is absent from scheduler maps and all native queues.
- Every previously scheduled batch containing it has completed.
- For a request ever sent to a worker, a completed native scheduler output has
  delivered its `finished_req_ids` cleanup notification to the worker.

Delayed connector releases and pending worker cleanup remain in the complete
native inventory until those conditions hold. HTTP disconnects merely record
the reason; only native release emits an ACK. ACKs use monotone native sequences
and a retained journal, with an eviction floor so a lost journal segment is
detectable. A new scheduler instance creates a new epoch. The profile signature
includes the pinned source, engine hash, scheduler configuration and cache
configuration.

Native remaining prompt work excludes known cached/completed prefix work and
adds back native work scheduled but still in flight. Cache matching for a newly
queued request is unknown; complete prompts awaiting visible text and generated
tokens with no visible text also remain unknown. The adapter never substitutes
the full client prompt for an unknown native value. Full inventory includes
requests without gateway correlation and retired requests awaiting release.
Preempted/recomputing requests report unknown work even after prior first text.

The scheduler is continuous batching. Capacity and maximum abort lag are always
null; no FIFO claim or finite real-time abort bound is made. A stalled model
step can delay abort and release indefinitely. The strict D3 forecast therefore
stays unavailable. Native ACKs can qualify D2 observed release, and the explicit
engine deadline can supply supported D1 evidence once actual rows arrive.

Local validation executes the pinned upstream bodies of `add_request`,
`finish_requests`, `_free_request`, `_free_blocks` and `AuthenticationMiddleware`
without importing GPU dependencies. Queue/model-execution fixtures provide the
remaining boundaries. This is source-level native lifecycle instrumentation,
not a full GPU engine or network end-to-end run. It does not establish GPU
cleanup latency, deployed image source compatibility, or a 300-second FP/FN
improvement.

```sh
SMG_VLLM_019_SOURCE=/tmp/vllm-native-source \
  python3 -m unittest discover -s grpc_servicer/tests -p test_native_deadline_vllm.py -v
```

Install `starlette==0.46.2` in the test environment to include the actual native
authentication/route tests. The engine environment already supplies Starlette.
Without the source path, the source-dependent lifecycle tests are explicitly
skipped. Supply a local checkout of the pinned upstream commit; the lifecycle
tests verify its source hashes. Manual source-download and fixture-generation
helpers are kept outside the repository.
