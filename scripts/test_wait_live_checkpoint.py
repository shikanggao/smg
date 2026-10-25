import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from scripts import wait_live_checkpoint as c
except ModuleNotFoundError:
    import wait_live_checkpoint as c


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.remote = self.root / "remote"
        self.remote.mkdir()
        self.paths = {n: str(self.remote / n) for n in c.FILES}
        self.patch = patch.object(c, "FILES", self.paths)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.owner = dict(name="driver", uid="owned-uid", namespace="default", run_id="test-run")
        self.pod = dict(metadata={k: self.owner[k] for k in ("name", "uid", "namespace")})
        self.pod["metadata"]["labels"] = {"codex.oracle.com/run-id": "test-run"}
        self.copy = c.Checkpoints(self.root / "host", self.owner)

    def execute(self, source):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(source, {})
        return out.getvalue()

    def capture(self, execute=None, get=None):
        return self.copy.capture(
            get or (lambda: self.pod), execute or self.execute, "2026-10-20T08:00:00Z"
        )

    def test_append_partial_line_recovery_and_resume_without_duplicates(self):
        source = Path(self.paths["requests.jsonl"])
        source.write_bytes(b'{"n":1}\n{"n":')
        result = self.capture()
        self.assertEqual(result["files"][source.name]["offset"], 8)
        self.assertEqual((self.copy.folder / source.name).read_bytes(), b'{"n":1}\n')
        with source.open("ab") as f:
            f.write(b"2}\n")
        self.copy = c.Checkpoints(self.copy.folder, self.owner)
        self.capture()
        self.capture()
        self.assertEqual((self.copy.folder / source.name).read_bytes(), source.read_bytes())
        self.assertIn("Partial", self.copy.state["scope"])

    def test_live_reader_is_bounded_and_only_commits_whole_records(self):
        Path(self.paths["metrics.jsonl"]).write_bytes(b"1\n2\n3\npartial")
        with patch.object(c, "CHUNK_BYTES", 5):
            first = self.capture()
            self.assertEqual(first["files"]["metrics.jsonl"]["offset"], 4)
        self.capture()
        self.assertEqual((self.copy.folder / "metrics.jsonl").read_bytes(), b"1\n2\n3\n")

    def test_uid_change_before_or_during_transfer_never_commits(self):
        Path(self.paths["requests.jsonl"]).write_bytes(b"{}\n")
        foreign = copy.deepcopy(self.pod)
        foreign["metadata"]["uid"] = "different"
        with self.assertRaisesRegex(ValueError, "identity"):
            self.capture(get=lambda: foreign)
        calls = iter([self.pod, foreign])
        with self.assertRaisesRegex(ValueError, "identity"):
            self.capture(get=lambda: next(calls))
        self.assertFalse(self.copy.ledger.exists())
        self.assertFalse((self.copy.folder / "requests.jsonl").exists())

    def test_truncation_disappearance_rotation_or_prefix_rewrite_fails(self):
        source = Path(self.paths["requests.jsonl"])
        for mutation in ["truncate", "missing", "replace", "rewrite"]:
            with self.subTest(mutation=mutation):
                source.write_bytes(b'{"n":1}\n')
                self.copy = c.Checkpoints(self.root / ("host-" + mutation), self.owner)
                self.capture()
                if mutation == "truncate":
                    source.write_bytes(b"")
                if mutation == "missing":
                    source.unlink()
                if mutation == "replace":
                    other = source.with_suffix(".replacement")
                    other.write_bytes(source.read_bytes())
                    other.replace(source)
                if mutation == "rewrite":
                    source.write_bytes(b'{"n":9}\n')
                with self.assertRaises(ValueError):
                    self.capture()
                self.assertEqual((self.copy.folder / source.name).read_bytes(), b'{"n":1}\n')

    def test_all_packet_entries_checked_before_any_write(self):
        for name in ["requests.jsonl", "metrics.jsonl"]:
            Path(self.paths[name]).write_bytes(b"{}\n")

        def tampered(source):
            packet = json.loads(self.execute(source))
            packet["metrics.jsonl"]["chunk_sha256"] = "bad"
            return json.dumps(packet)

        with self.assertRaisesRegex(ValueError, "integrity"):
            self.capture(execute=tampered)
        self.assertFalse((self.copy.folder / "requests.jsonl").exists())

    def test_local_tamper_wrong_owner_and_unowned_files_rejected(self):
        source = Path(self.paths["requests.jsonl"])
        source.write_bytes(b"{}\n")
        self.capture()
        with self.assertRaisesRegex(ValueError, "different"):
            c.Checkpoints(self.copy.folder, dict(self.owner, uid="new"))
        (self.copy.folder / source.name).write_bytes(b"[]\n")
        with self.assertRaisesRegex(ValueError, "prefix hash"):
            c.Checkpoints(self.copy.folder, self.owner)
        (self.copy.folder / source.name).write_bytes(source.read_bytes())
        source2 = Path(self.paths["metrics.jsonl"])
        source2.write_bytes(b"{}\n")
        (self.copy.folder / source2.name).write_bytes(b"preexisting")
        with self.assertRaisesRegex(ValueError, "unowned"):
            self.capture()

    def test_remote_rejects_arbitrary_paths_and_symlinks(self):
        with self.assertRaises(ValueError):
            c.remote_snapshot({"../secret": 0})
        Path(self.paths["requests.jsonl"]).symlink_to(__file__)
        with self.assertRaisesRegex(ValueError, "regular"):
            self.capture()

    def test_crash_after_append_before_ledger_preserves_partial_and_refuses_resume(self):
        source = Path(self.paths["requests.jsonl"])
        source.write_bytes(b'{"n":1}\n')
        self.capture()
        old_ledger = self.copy.ledger.read_bytes()
        with source.open("ab") as stream:
            stream.write(b'{"n":2}\n')
        with patch.object(Path, "replace", side_effect=OSError("simulated host crash")):
            with self.assertRaisesRegex(OSError, "host crash"):
                self.capture()
        self.assertEqual(self.copy.ledger.read_bytes(), old_ledger)
        self.assertEqual((self.copy.folder / source.name).read_bytes(), source.read_bytes())
        self.assertTrue(self.copy.ledger.with_suffix(".pending").exists())
        with self.assertRaisesRegex(ValueError, "preserve partial"):
            c.Checkpoints(self.copy.folder, self.owner)
        self.assertEqual(self.copy.ledger.read_bytes(), old_ledger)


if __name__ == "__main__":
    unittest.main()
