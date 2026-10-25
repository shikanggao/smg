import json
import unittest
from copy import deepcopy

from replay_wait_decision import (
    audit_schema,
    bind_native_profile,
    replay,
    verify_snapshot,
)
from wait_decision_calibration import METHOD, State
from wait_ttft_readiness import Readiness


def snapshot(now=0.0):
    return dict(
        at=now,
        profile=7,
        baseline=2.0,
        x=100.0,
        source="live_n_scalar",
        original_queue_extra=100.0,
        live_queue_extra=100.0,
        scalar_gain=1.0,
        fresh=True,
        native_ready=True,
        native_waiting=1.0,
        poll_started=now,
        eligible=True,
        context=State().context(now),
    )


class LifecycleTests(unittest.TestCase):
    def state(self):
        s = State()
        s.poll(0.0, 7, True, 5.0)
        return s

    def test_bound_upgrade_keeps_original_bank_position(self):
        s = self.state()
        s.register(1, 0.0, snapshot())
        s.label(1, 300.0, 300.0, False, False, 300.0)
        s.register(2, 10.0, snapshot(10.0))
        s.label(2, 310.0, 310.0, False, False, 300.0)
        first = deepcopy(s.rows[0])
        value = s.label(1, 400.0, 400.0, True, True)
        self.assertEqual([r["id"] for r in s.rows], [1, 2])
        self.assertEqual(value["first_label_at"], 300.0)
        self.assertEqual(value["snapshot"], first["snapshot"])
        self.assertEqual(len(s.pending), 1)

    def test_evicted_bound_cannot_resurrect(self):
        s = self.state()
        for i in range(33):
            s.register(i, 0.0, snapshot())
            s.label(i, 300.0, 300.0, False, False, 300.0)
        self.assertEqual(len(s.rows), 32)
        self.assertIsNone(s.label(0, 400.0, 400.0, True, True))

    def test_age_600_expires_even_when_feedback_is_recent(self):
        s = self.state()
        s.register(1, 0.0, snapshot())
        s.label(1, 599.0, 599.0, False, False)
        s.poll(600.0, 7, True, 5.0)
        self.assertFalse(s.rows)

    def test_growth_requires_continuity_and_complete_registry(self):
        s = self.state()
        s.poll(30.0, 7, True, 40.0)
        s.register(1, 31.0, snapshot(31.0))
        self.assertEqual(s.context(31.0)["values"][0], 1.0)
        s.poll(40.0, 7, True, 5.0)
        self.assertIsNone(s.context(40.0)["values"][0])
        s.mark_incomplete()
        self.assertIsNone(s.context(40.0)["oldest_seconds"])

    def test_zero_native_progress_is_valid(self):
        s = self.state()
        s.register(1, 0.0, snapshot())
        self.assertEqual(s.context(1.0, 0.0, 100.0, True)["values"][2], 1.0)
        self.assertIsNone(s.context(1.0, None, 100.0, True)["values"][2])

    def test_profile_reset_keeps_pending_but_rejects_old_training(self):
        s = self.state()
        s.register(1, 0.0, snapshot())
        s.poll(1.0, 8, True, 5.0)
        self.assertEqual(len(s.pending), 1)
        self.assertIsNone(s.label(1, 2.0, 2.0, True, True))

    def test_snapshot_mutations_fail(self):
        s = self.state()
        v = snapshot()
        verify_snapshot(v, s, 5.0)
        for key, value in [
            ("x", 101.0),
            ("fresh", False),
            ("source", "original_n_scalar_fallback"),
        ]:
            broken = deepcopy(v)
            broken[key] = value
            with self.assertRaises(ValueError):
                verify_snapshot(broken, s, 5.0)

    def test_method_readiness_fails_closed(self):
        e = dict(
            event="decision_poll",
            worker="w",
            calibration_epoch=1,
            decision_method=METHOD,
            decision_json=json.dumps(
                dict(
                    snapshot=None,
                    predictions=None,
                    state=State().diagnostics(),
                    fresh=False,
                    profile=None,
                    max_gap=5.0,
                    available=False,
                )
            ),
        )
        self.assertTrue(audit_schema([e])["ok"])
        self.assertFalse(audit_schema([])["ok"])
        e["decision_method"] = "old"
        self.assertFalse(audit_schema([e])["ok"])
        r = Readiness("x", expected_decision_method=METHOD)
        r.observe(e)
        self.assertTrue(r.errors)


if __name__ == "__main__":
    unittest.main()


def event_fixture():
    def event(kind, now, p):
        return dict(
            event=kind,
            worker="w",
            calibration_epoch=0,
            diagnostic_time_seconds=now,
            decision_method=METHOD,
            decision_json=json.dumps(p),
        )

    native = dict(
        expected_rank_count=1,
        ranks=[
            dict(
                dp_rank=0,
                native_request_counts_available=True,
                native_request_counts_profile=1,
                timely_prefill_progress=0.0,
                timely_prefill_progress_source="local_compute_tokens",
                timely_prefill_progress_interval=1.0,
                learned_prefill_capacity=100.0,
                prefill_capacity_sample_count=4,
                prefill_capacity_learning_active=True,
                num_waiting_reqs=1,
                num_running_reqs=0,
                num_waiting_uncached_tokens=100,
                num_waiting_uncached_tokens_available=True,
            )
        ],
    )
    n = event("decision_native_poll", 0.0, native)
    n.update(poll_started_time_seconds=0.0, outcome="success")
    state = State()
    state.poll(0.0, 7, True, 5.0)
    snap = snapshot()
    snap["context"] = state.context(0.0, 0.0, 100.0, True)
    poll = event(
        "decision_poll",
        0.0,
        dict(
            snapshot=snap,
            predictions=state.predictions(snap, 0.0, True),
            state=state.diagnostics(),
            profile=7,
            fresh=True,
            max_gap=5.0,
            available=True,
        ),
    )
    snap = snapshot(1.0)
    snap["poll_started"] = 0.0
    snap["context"] = state.context(1.0, 0.0, 100.0, True)
    selected = dict(
        event="calibration_forecast",
        worker="w",
        request_id="r",
        calibration_epoch=0,
        diagnostic_time_seconds=1.0,
        calibration_baseline_seconds=2.0,
        queue_excess_seconds=100.0,
        calibration_scale=1.0,
        native_ready=True,
        native_waiting=1.0,
    )
    admission = event(
        "decision_admission",
        1.2,
        dict(
            id=1,
            tracked=True,
            started=1.1,
            snapshot=snap,
            predictions=state.predictions(snap, 1.0, True),
        ),
    )
    admission["request_id"] = "r"
    state.register(1, 1.1, snap)
    label = state.label(1, 5.1, 5.2, True, True)
    feedback = event("decision_label", 5.2, dict(label=label))
    feedback["reason"] = "first_text"
    terminal = event("decision_terminal", 5.2, dict(id=1, observed_at=5.1, outcome="first_text"))
    outcome = dict(
        event="feature_outcome",
        worker="w",
        request_id="r",
        calibration_epoch=0,
        diagnostic_time_seconds=5.2,
        outcome="first_text",
    )
    recovery = dict(
        event="recovery_poll",
        worker="w",
        calibration_epoch=0,
        diagnostic_time_seconds=0.01,
        available=True,
        original_queue_excess_seconds=100.0,
        dispatch_debt_seconds=0.0,
        queue_excess_seconds=100.0,
        net_population_depletion_per_second=2.0,
        native_population=200.0,
        latest_poll_started_time_seconds=0.0,
        window_seconds=30.0,
        window_poll_count=5,
    )
    return [
        n,
        poll,
        recovery,
        selected,
        admission,
        feedback,
        terminal,
        outcome,
        dict(event="first_text", request_id="r", ttft_seconds=4.0),
    ]


class ReplayTests(unittest.TestCase):
    def test_independent_causal_admission_and_terminal(self):
        result = replay(event_fixture())
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["labels"]), 1)

    def test_saved_prediction_corruption(self):
        events = event_fixture()
        value = json.loads(events[4]["decision_json"])
        value["predictions"][0]["total"] += 1
        events[4]["decision_json"] = json.dumps(value)
        with self.assertRaises(ValueError):
            replay(events)

    def test_exact_label_requires_terminal_provenance(self):
        events = event_fixture()
        events[-1]["ttft_seconds"] = 300.0
        with self.assertRaises(ValueError):
            replay(events)

    def test_native_source_and_rank_completeness_fail(self):
        for field, value in [
            ("timely_prefill_progress_source", "completed_prompt"),
            ("timely_prefill_progress_interval", 10.0),
        ]:
            events = event_fixture()
            native = json.loads(events[0]["decision_json"])
            native["ranks"][0][field] = value
            events[0]["decision_json"] = json.dumps(native)
            with self.assertRaises(ValueError):
                replay(events)

    def test_missing_eligible_label_fails(self):
        events = event_fixture()
        del events[5]
        with self.assertRaises(ValueError):
            replay(events)


class ScoringTests(unittest.TestCase):
    def data(self):
        from replay_wait_decision import score_admissions

        admission = json.loads(event_fixture()[4]["decision_json"])
        admission.update(request_id="r", worker="w", epoch=0)
        paired = dict(
            opc_request_id="r",
            phase="pressure",
            first_text_late=False,
            started_elapsed_seconds=1.0,
            feature_context_candidate=dict(snapshot=dict(threshold_seconds=300.0)),
            pre_admission_recorded_extra_flags=dict(calibrated=False),
            pre_admission_total_ttft_predictions=dict(calibrated=102.0),
        )
        return score_admissions, admission, paired

    def test_missing_admission_cannot_shrink_denominator(self):
        fn, a, p = self.data()
        with self.assertRaises(ValueError):
            fn([], [p], "capture", "model")

    def test_extra_and_total_are_distinct(self):
        fn, a, p = self.data()
        a["snapshot"]["x"] = 299.0
        a["snapshot"]["baseline"] = 2.0
        rows, report = fn([a], [p], "capture", "model")
        self.assertEqual(report["extra"]["scores"]["live_n_scalar"]["all"]["fp"], 0)
        self.assertEqual(report["total_ttft_300"]["scores"]["live_n_scalar"]["all"]["fp"], 1)

    def test_rejected_owned_request_remains_unknown(self):
        fn, a, p = self.data()
        p["controlled_rejection"] = True
        rows, report = fn([], [p], "capture", "model")
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["truth"])
        self.assertEqual(report["actual_rejected"], 1)


class ProvenanceTests(unittest.TestCase):
    def test_profile_signature_is_order_invariant(self):
        evidence = json.loads(event_fixture()[0]["decision_json"])
        second = deepcopy(evidence["ranks"][0])
        second["dp_rank"] = 1
        evidence["ranks"].append(second)
        bindings = {}
        bind_native_profile(bindings, "w", 7, evidence)
        evidence["ranks"].reverse()
        bind_native_profile(bindings, "w", 7, evidence)
        evidence["ranks"][0]["native_request_counts_profile"] = 2
        with self.assertRaisesRegex(ValueError, "signature changed"):
            bind_native_profile(bindings, "w", 7, evidence)

    def test_second_poll_source_or_counter_change_cannot_keep_profile_and_history(self):
        for field, value in [
            ("timely_prefill_progress_source", "prefill_started_requests"),
            ("native_request_counts_profile", 2),
        ]:
            events = event_fixture()
            native = deepcopy(events[0])
            native["diagnostic_time_seconds"] = 6.0
            native["poll_started_time_seconds"] = 6.0
            ranks = json.loads(native["decision_json"])
            ranks["ranks"][0][field] = value
            native["decision_json"] = json.dumps(ranks)
            s = State()
            s.poll(0.0, 7, True, 5.0)
            original = json.loads(events[4]["decision_json"])
            s.register(1, 1.1, original["snapshot"])
            s.label(1, 5.1, 5.2, True, True)
            s.poll(6.0, 7, True, 5.0)
            snap = snapshot(6.0)
            snap["context"] = s.context(6.0, 0.0, 100.0, True)
            poll = deepcopy(events[1])
            poll["diagnostic_time_seconds"] = 6.0
            poll["decision_json"] = json.dumps(
                dict(
                    snapshot=snap,
                    predictions=s.predictions(snap, 6.0, True),
                    state=s.diagnostics(),
                    profile=7,
                    fresh=True,
                    max_gap=5.0,
                    available=True,
                )
            )
            with self.assertRaisesRegex(ValueError, "signature changed"):
                replay(events + [native, poll])

    def test_selected_profile_must_match_initialized_state(self):
        events = event_fixture()
        value = json.loads(events[4]["decision_json"])
        value["snapshot"]["profile"] = 8
        events[4]["decision_json"] = json.dumps(value)
        with self.assertRaisesRegex(ValueError, "profile differs"):
            replay(events)

    def test_neutral_reset_allows_retained_sample_profile(self):
        state = State()
        verify_snapshot(snapshot(), state, 5.0)

    def test_live_queue_and_predictions_cannot_be_changed_together(self):
        events = event_fixture()
        value = json.loads(events[4]["decision_json"])
        value["snapshot"]["live_queue_extra"] = 90.0
        value["snapshot"]["x"] = 90.0
        state = State()
        state.poll(0.0, 7, True, 5.0)
        value["predictions"] = state.predictions(value["snapshot"], 1.0, True)
        events[4]["decision_json"] = json.dumps(value)
        with self.assertRaisesRegex(ValueError, "Live-N source binding"):
            replay(events)

    def test_reset_before_selected_admission_keeps_neutral_prediction(self):
        events = event_fixture()
        events.insert(3, dict(event="calibration_reset", worker="w", calibration_epoch=1))
        for e in events[4:]:
            if "calibration_epoch" in e:
                e["calibration_epoch"] = 1
        events = [e for e in events if e.get("event") != "decision_label"]
        admission = next(e for e in events if e.get("event") == "decision_admission")
        value = json.loads(admission["decision_json"])
        value["snapshot"]["live_queue_extra"] = None
        value["snapshot"]["source"] = "original_n_scalar_fallback"
        admission["decision_json"] = json.dumps(value)
        result = replay(events)
        self.assertTrue(result["ok"])
        self.assertEqual(result["labels"], [])

    def test_consumed_files_must_be_explicitly_hash_bound(self):
        import hashlib
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        for missing in ("router-events.jsonl", "paired-production.jsonl"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                files = {}
                for name in ("router-events.jsonl", "paired-production.jsonl"):
                    (root / name).write_text("")
                    if name != missing:
                        files[name] = hashlib.sha256(b"").hexdigest()
                (root / "state.json").write_text(
                    json.dumps(dict(results_valid=True, result_path=str(root), result_files=files))
                )
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("replay_wait_decision.py")),
                        "--router",
                        str(root / "router-events.jsonl"),
                        "--state",
                        str(root / "state.json"),
                        "--out",
                        str(root / "out"),
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Consumed canonical sources missing", result.stderr)
