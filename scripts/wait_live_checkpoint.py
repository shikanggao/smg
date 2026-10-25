"""Incremental off-Pod copies of an exact-owned driver's append-only evidence.

Copies are partial evidence, never a successful run/collection marker. The host
coordinator supplies its existing UID-guarded get/exec callbacks; this module
does not authenticate, create resources, resume inference, or alter the driver.
Remote continuity checks inode/size and a trailing 256-byte anchor, not a hash
of the full remote prefix. Local copies have full-prefix SHA256 hashes. A crash
after an append but before the ledger commit fails closed on restart: preserve
the partial files/ledger instead of silently replacing or resuming them.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

FILES = {
    "requests.jsonl": "/work/output/production-shadow/requests.jsonl",
    "client-events.jsonl": "/work/output/production-shadow/client-events.jsonl",
    "metrics.jsonl": "/work/output/production-shadow/metrics.jsonl",
    "router-events.jsonl": "/work/output/router-events.jsonl",
    "native-log-events.jsonl": "/work/output/native-log-events.jsonl",
}
CHUNK_BYTES = 4 * 1024 * 1024
ANCHOR_BYTES = 256


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_owner(pod, owner):
    meta = pod.get("metadata", {})
    if any(meta.get(k) != owner[k] for k in ("name", "uid", "namespace")):
        raise ValueError("Checkpoint driver identity changed")
    if meta.get("labels", {}).get("codex.oracle.com/run-id") != owner["run_id"]:
        raise ValueError("Checkpoint run ownership changed")
    if meta.get("deletionTimestamp"):
        raise ValueError("Checkpoint driver is terminating; preserve prior partial evidence")


def remote_snapshot(offsets):
    """Executed in the owned driver; bounded reads, no arbitrary path selection."""
    import base64
    import hashlib
    import os
    from pathlib import Path

    if set(offsets) - FILES.keys():
        raise ValueError("Unknown checkpoint evidence file")
    result = {}
    for name, filename in FILES.items():
        offset = offsets.get(name, 0)
        if type(offset) is not int or offset < 0:
            raise ValueError("Invalid checkpoint offset")
        path = Path(filename)
        if not path.exists():
            result[name] = {"missing": True}
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("Checkpoint source is not a regular evidence file")
        with path.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            if stat.st_size < offset:
                raise ValueError("Checkpoint source truncated")
            stream.seek(max(0, offset - ANCHOR_BYTES))
            anchor = stream.read(min(offset, ANCHOR_BYTES))
            stream.seek(offset)
            chunk = stream.read(min(CHUNK_BYTES, stat.st_size - offset))
            # Never commit an in-flight JSON line or invent its terminal fields.
            end = chunk.rfind(b"\n") + 1
            chunk = chunk[:end]
            result[name] = {
                "offset": offset,
                "end": offset + len(chunk),
                "source_size": stat.st_size,
                "source_identity": [stat.st_dev, stat.st_ino],
                "anchor_sha256": hashlib.sha256(anchor).hexdigest(),
                "chunk_sha256": hashlib.sha256(chunk).hexdigest(),
                "data_base64": base64.b64encode(chunk).decode("ascii"),
            }
    return result


def remote_code(offsets):
    import inspect

    return (
        "import json\nFILES="
        + repr(FILES)
        + "\nCHUNK_BYTES="
        + str(CHUNK_BYTES)
        + "\nANCHOR_BYTES="
        + str(ANCHOR_BYTES)
        + "\n"
        + inspect.getsource(remote_snapshot)
        + "\nprint(json.dumps(remote_snapshot("
        + repr(offsets)
        + ")))\n"
    )


class Checkpoints:
    def __init__(self, folder, owner):
        if set(owner) != {"name", "uid", "namespace", "run_id"} or not all(owner.values()):
            raise ValueError("Exact driver ownership is required")
        self.folder, self.owner = Path(folder), dict(owner)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.ledger = self.folder / "checkpoint.json"
        self.state = (
            json.loads(self.ledger.read_text())
            if self.ledger.exists()
            else {
                "schema_version": 1,
                "owner": self.owner,
                "files": {},
                "checkpoints": 0,
                "scope": "Partial append-only evidence; not full-run validity or completion",
            }
        )
        if self.state.get("schema_version") != 1 or self.state.get("owner") != self.owner:
            raise ValueError("Checkpoint ledger belongs to a different driver")
        for name, saved in self.state["files"].items():
            path = self.folder / name
            if name not in FILES or not path.is_file() or path.stat().st_size != saved["offset"]:
                raise ValueError(
                    "Checkpoint local offset does not match ledger; preserve partial files and ledger"
                )
            if sha(path.read_bytes()) != saved["prefix_sha256"]:
                raise ValueError("Checkpoint local prefix hash changed")

    def capture(self, get_pod, execute, observed_utc):
        """Callbacks must be bounded, UID-guarded calls from the host coordinator."""
        validate_owner(get_pod(), self.owner)
        offsets = {n: s["offset"] for n, s in self.state["files"].items()}
        packet = json.loads(execute(remote_code(offsets)))
        validate_owner(get_pod(), self.owner)
        if set(packet) != FILES.keys():
            raise ValueError("Checkpoint packet file inventory changed")
        pending = []
        missing = []
        for name, value in packet.items():
            old = self.state["files"].get(name)
            if value.get("missing"):
                if old:
                    raise ValueError("Previously captured evidence disappeared")
                missing.append(name)
                continue
            offset = offsets.get(name, 0)
            path = self.folder / name
            if old and value["source_identity"] != old["source_identity"]:
                raise ValueError("Checkpoint source rotated or replaced")
            if value["offset"] != offset or value["source_size"] < value["end"]:
                raise ValueError("Checkpoint source offset mismatch")
            anchor = b""
            if offset:
                with path.open("rb") as stream:
                    stream.seek(max(0, offset - ANCHOR_BYTES))
                    anchor = stream.read()
            if sha(anchor) != value["anchor_sha256"]:
                raise ValueError("Checkpoint source prefix changed")
            chunk = base64.b64decode(value["data_base64"], validate=True)
            if (
                len(chunk) > CHUNK_BYTES
                or sha(chunk) != value["chunk_sha256"]
                or value["end"] != offset + len(chunk)
                or chunk
                and not chunk.endswith(b"\n")
            ):
                raise ValueError("Checkpoint chunk integrity failed")
            if not old and path.exists():
                raise ValueError("Checkpoint refuses to overwrite an unowned local file")
            pending.append((name, path, value, chunk))
        # Validate every source before committing any bytes.
        for name, path, value, chunk in pending:
            with path.open("ab") as stream:
                stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            self.state["files"][name] = {
                "offset": value["end"],
                "source_size_at_read": value["source_size"],
                "source_identity": value["source_identity"],
                "prefix_sha256": sha(path.read_bytes()),
                "last_chunk_sha256": value["chunk_sha256"],
            }
        self.state.update(
            checkpoints=self.state["checkpoints"] + 1,
            observed_utc=observed_utc,
            missing_not_yet_created=missing,
        )
        temp = self.ledger.with_suffix(".pending")
        with temp.open("w") as stream:
            json.dump(self.state, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(self.ledger)
        return self.state
