"""Download official pinned source for GPU-free native lifecycle validation."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.request import urlopen

module_path = Path(__file__).parents[1] / "smg_grpc_servicer/native_deadline_vllm.py"
spec = importlib.util.spec_from_file_location("native_deadline_vllm", module_path)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    paths = {
        name.replace(".", "/") + ".py": digest for name, digest in native.SOURCE_HASHES.items()
    }
    manifest = {"upstream_commit": native.UPSTREAM_COMMIT, "files": {}}
    for path, expected in paths.items():
        url = f"https://raw.githubusercontent.com/vllm-project/vllm/{native.UPSTREAM_COMMIT}/{path}"
        with urlopen(url, timeout=20) as response:
            content = response.read()
        digest = hashlib.sha256(content).hexdigest()
        if expected is not None and digest != expected:
            raise RuntimeError(f"Pinned upstream source hash mismatch: {path}")
        target = args.destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest["files"][path] = {"url": url, "sha256": digest}
    (args.destination / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(paths)} pinned upstream source files in {args.destination}")


if __name__ == "__main__":
    main()
