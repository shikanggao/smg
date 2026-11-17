#!/usr/bin/env python3
"""Apply the opt-in native hooks to an exact upstream source checkout.

No downloads, engine launch, dependency install, images, or deployment. Every
input is checked before any write; this intentionally refuses other versions.
"""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

PIN = "c9e5799bd12fd5fefb0fb375ac537a6b7d53892e"
UNCHANGED_FILES = {
    "entrypoints/openai/serving_chat.py": "c147a9db349db5478d65584fe18ff69c2300ae1f0d1711a2bde280d94b2f670d",
    "entrypoints/openai/serving_completions.py": "3ba4559636bf257006cb9de132932bc11a772533a47b274e959aab2941b9950f",
    "entrypoints/openai/serving_base.py": "f15ebb0486bc2a914f6d3e21bd9c3069106bb7e69c67bbbf73bca79d1ba87659",
    "server_args.py": "502836bfeb64adff5309c0337df79ac8eec4be770b600d4edfd7ce46fc04d06e",
    "utils/common.py": "b0e1a8ee63a91838c088e9b0a3095a7d52459ff992bc0b2f5c3776116e1a63a7",
    "managers/schedule_batch.py": "cde09ea7f562c227ba629c0bc48ce0ec24a8bc31c5f7fca43e73f65e0cc05bd6",
    "managers/schedule_policy.py": "0455a81c6de6fa591e56f3a42c6a0c7846d1a65ab8bc5e2d86c8eff8ff0c1bce",
    "managers/io_struct.py": "2f6f2318df448a3577c4f9800ef09b3209a3e04fef29e11872a526088d3f416f",
    "mem_cache/common.py": "e82da07bf1a0ac8006fa9b368e51d3723121ba06ff1ffc2c7bf58933d9e71fa3",
}
FILES = {
    "managers/scheduler.py": "40f1cea1f77daa15a5c98d49dc098fecaa13f9c95c9aa8276d1386df925b3cb3",
    "managers/tokenizer_manager.py": "779376f6e957b155269a87656770ca836f6ff73842b7f6d264c10ca906f642b0",
    "entrypoints/http_server.py": "2c12da05008d32a43bb3d85f82619f8653e938aa1361139348dd0b47d3b68233",
    "managers/scheduler_output_processor_mixin.py": "0a88aae482b9297cbcc753f03f058b9ccac6b9579ce0035a8659d96cea47f6c0",
}


def scheduler_receipt_sites(source):
    """Narrow exact-pin statement hooks; never rewrite a decision expression."""
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    lines = source.splitlines(keepends=True)
    method = methods["get_new_batch_prefill"]
    old = "".join(lines[method.lineno - 1:method.end_lineno])
    new = old
    replacements = [
        ('        ) and self.chunked_req is None:\n            return None\n',
         '        ) and self.chunked_req is None:\n            _smg_sched_note(self, "full_or_empty_no_chunk", running_full=self.running_batch.batch_is_full)\n            return None\n        _smg_sched_note(self, running_full=self.running_batch.batch_is_full)\n'),
        ('            self.running_batch.batch_is_full = True\n            return None\n',
         '            self.running_batch.batch_is_full = True\n            _smg_sched_note(self, "no_request_slots", running_full=True)\n            return None\n'),
        ('            # in the waiting queue.\n            return None\n',
         '            # in the waiting queue.\n            _smg_sched_note(self, "test_retract")\n            return None\n'),
        ('            self.priority_scheduling_preemption_threshold,\n        )\n',
         '            self.priority_scheduling_preemption_threshold,\n        )\n        _smg_bind_adder(self, adder)\n'),
        ('                self.running_batch.batch_is_full = True\n                break\n',
         '                self.running_batch.batch_is_full = True\n                _smg_sched_note(self, "lora_limit", running_full=True)\n                break\n'),
        ('            if self.running_batch.batch_is_full:\n                if not self.try_preemption:\n                    break\n                if not adder.preempt_to_schedule(req, self.server_args):\n                    break\n',
         '            if self.running_batch.batch_is_full:\n                _smg_sched_note(self, running_full=True)\n                if not self.try_preemption:\n                    _smg_sched_note(self, "running_full_no_preemption")\n                    break\n                if not adder.preempt_to_schedule(req, self.server_args):\n                    _smg_sched_note(self, "preemption_refused")\n                    break\n            else:\n                _smg_sched_note(self, running_full=False)\n'),
        ('                    # skip staging requests that are ongoing prefetch\n                    continue\n',
         '                    # skip staging requests that are ongoing prefetch\n                    _smg_prefetch_skip(self)\n                    continue\n'),
        ('                break\n\n        # Update waiting queue\n',
         '                break\n        else:\n            _smg_sched_note(self, "waiting_queue_exhausted")\n\n        # Update waiting queue\n'),
        ('        if len(can_run_list) == 0:\n            return None\n',
         '        if len(can_run_list) == 0:\n            _smg_sched_note(self, "empty_prefill_selection", only_if_unset=True)\n            return None\n'),
    ]
    for expected, replacement in replacements:
        expected, replacement = "\n" + expected, "\n" + replacement
        if new.count(expected) != 1:
            raise ValueError("Unexpected pinned scheduler receipt site: " + expected.strip().splitlines()[0])
        new = new.replace(expected, replacement, 1)
    lines[method.lineno - 1:method.end_lineno] = [new]
    source = "".join(lines)
    # Reparse because the preceding insertion changes later line numbers.
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_num_allocatable_reqs")
    lines = source.splitlines(keepends=True)
    old = "".join(lines[method.lineno - 1:method.end_lineno])
    if old.count("        return res\n") != 1:
        raise ValueError("Unexpected pinned allocatable request return")
    lines[method.lineno - 1:method.end_lineno] = [old.replace("        return res\n", "        return _smg_slots_value(self, res, running_bs)\n")]
    return "".join(lines)


def prepare(root):
    root = Path(root)
    srt = root / "python/sglang/srt"
    content = {}
    for relative, expected in {**FILES, **UNCHANGED_FILES}.items():
        path = srt / relative
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Refusing source drift: {relative}; expected {expected}, got {actual}"
            )
        if relative in FILES:
            content[relative] = data.decode()

    scheduler = content["managers/scheduler.py"]
    tree = ast.parse(scheduler)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    abort = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "abort_request"
    )
    lines = scheduler.splitlines(keepends=True)
    abort_source = "".join(lines[abort.lineno - 1 : abort.end_lineno])
    old = "req.rid.startswith(recv_req.rid)"
    count = abort_source.count(old)
    if count != 7:
        raise ValueError(f"Unexpected pinned abort matching sites: {count}")
    abort_source = re.sub(
        r"(?:decode_req\.req|req)\.rid\.startswith\(recv_req\.rid\)",
        lambda match: (
            "("
            + match.group(0).split(".rid.")[0]
            + '.rid == recv_req.rid if getattr(recv_req, "smg_exact", False) else '
            + match.group(0)
            + ")"
        ),
        abort_source,
    )
    lines[abort.lineno - 1 : abort.end_lineno] = [abort_source]
    scheduler = scheduler_receipt_sites("".join(lines))
    scheduler += (
        "\n# SMG opt-in native signals; source pinned by the patch installer.\n"
        "from sglang.srt.managers.smg_scheduler_receipt import (note as _smg_sched_note, bind_adder as _smg_bind_adder, slots_value as _smg_slots_value, prefetch_skip as _smg_prefetch_skip)\n"
        "from sglang.srt.managers.smg_native_deadline import install_scheduler, release_observer\n"
        "release_kv_cache = release_observer(release_kv_cache)\n"
        "install_scheduler(Scheduler)\n"
    )
    content["managers/scheduler.py"] = scheduler
    content["managers/tokenizer_manager.py"] += (
        "\nfrom sglang.srt.managers.smg_native_deadline import install_tokenizer\n"
        "install_tokenizer(TokenizerManager)\n"
    )
    content["managers/scheduler_output_processor_mixin.py"] += (
        "\nfrom sglang.srt.managers.smg_native_deadline import release_observer\n"
        "release_kv_cache = release_observer(release_kv_cache)\n"
    )
    content["entrypoints/http_server.py"] += (
        "\nfrom sglang.srt.managers.smg_native_deadline import install_http\n"
        "install_http(app, lambda: _global_state.tokenizer_manager)\n"
    )
    content["managers/smg_native_deadline.py"] = (
        Path(__file__).with_name("native_deadline.py").read_text()
    )
    content["managers/smg_scheduler_receipt.py"] = Path(__file__).with_name("scheduler_receipt.py").read_text()
    content["managers/smg_engine_work_progress.py"] = (
        Path(__file__).resolve().parents[2]
        / "smg_grpc_servicer/engine_work_progress.py"
    ).read_text()
    for relative, source in content.items():
        ast.parse(source, filename=relative)
    return srt, content


def apply(root, check=False):
    srt, content = prepare(root)
    manifest = {
        name: hashlib.sha256(source.encode()).hexdigest() for name, source in content.items()
    }
    if not check:
        for relative, source in content.items():
            (srt / relative).write_text(source)
    return dict(upstream_commit=PIN, check_only=check, output_sha256=manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(apply(args.checkout, args.check), indent=2))
