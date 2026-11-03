"""Emit native-producer JSON using the pinned upstream lifecycle test harness.

This is a CPU instrumentation fixture, not an engine/GPU end-to-end result.
"""

import argparse
import importlib.util
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    path = Path(__file__).with_name("test_native_deadline_vllm.py")
    spec = importlib.util.spec_from_file_location("native_fixture_harness", path)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    native = harness.native
    args.destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        store = native.NativeStore(Path(temp) / "native.sqlite3")
        cls = harness.upstream_scheduler(args.source)
        native.install_scheduler_hooks(cls, harness.Status.FINISHED_ABORTED, lambda: store)
        scheduler = cls()
        scheduler._smg_abort_outputs = lambda _: None
        attempt = "00000000-0000-4000-8000-000000000061"
        store.register(attempt, 10000, time.monotonic())
        request = harness.Request("chatcmpl-upstream-exact-0-12345678", attempt)
        scheduler.add_request(request)
        request.num_cached_tokens, request.num_computed_tokens = 40, 70
        batch = SimpleNamespace(num_scheduled_tokens={request.request_id: 20})
        scheduler._smg_signals.inflight[id(batch)] = batch.num_scheduled_tokens
        scheduler._smg_signals.ever_scheduled.add(request.request_id)
        scheduler._smg_signals.publish()

        def write(name):
            target = args.destination / name
            target.write_text(json.dumps(store.loads(time.monotonic()), indent=2) + "\n")
            print(target)

        write("vllm-native-progress.json")
        scheduler.finish_requests(request.request_id, harness.Status.FINISHED_ABORTED)
        write("vllm-native-abort-pending-cleanup.json")
        scheduler.update_from_output(batch)
        cleanup = scheduler.schedule()
        scheduler.update_from_output(cleanup)
        write("vllm-native-ack.json")
        manifest = {
            "upstream_commit": native.UPSTREAM_COMMIT,
            "producer_module": str(Path(native.__file__).resolve()),
            "native_methods_executed": [
                "add_request",
                "finish_requests",
                "_free_request",
                "_free_blocks",
            ],
            "fixture_boundaries": [
                "scheduler context",
                "request",
                "model batch/inflight completion",
                "worker cleanup completion",
            ],
            "gpu_or_network_end_to_end": False,
        }
        (args.destination / "vllm-native-fixture-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
