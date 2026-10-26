import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import wait_decision_calibration as legacy
import wait_decision_context_continuity as current
import wait_decision_direct as direct
import wait_decision_direct_unpenalized as previous
from replay_wait_context_continuity import verify_binding
from replay_wait_decision import audit_schema, replay
from replay_wait_joint_gains import sha
from test_wait_decision_replay import event_fixture


def supported(mask, coeff):
    value = current.rust_fit([], False, False)
    value.update(
        supported=True, mask=mask, coefficients=coeff, minimum=[0.2] * 3, maximum=[0.8] * 3
    )
    return value


def snapshot(values):
    return dict(x=100.0, baseline=2.0, scalar_gain=3.0, context=dict(values=values))


class ContinuityTests(unittest.TestCase):
    def test_intersection_preserves_partly_slewed_q_and_shared_slopes(self):
        m = current.Model()
        m.update(supported(5, [2.8, -0.4, 0.0, 0.2]), 0.0)
        before = m.applied(1.0)
        m.update(supported(6, [3.0, 0.0, -0.1, 0.25]), 1.0)
        self.assertEqual(m.anchor, [before[0], 0.0, 0.0, before[3]])
        after = m.applied(1.2)
        self.assertLessEqual(sum(abs(a - b) for a, b in zip(m.anchor, after)), 0.1 + 1e-12)

    def test_same_mask_and_queue_are_byte_identical_to_v3(self):
        a, b = current.Model(), previous.Model()
        for now, coeff in [
            (0.0, [2.0, 0.2, 0.0, 0.0]),
            (0.1, [3.0, -0.2, 0.0, 0.0]),
            (0.2, [1.5, 0.3, 0.0, 0.0]),
        ]:
            for m in (a, b):
                m.update(supported(1, coeff), now)
            self.assertEqual(a.__dict__, b.__dict__)
        for cls in (legacy.Model, previous.Model):
            self.assertIs(previous.Model, cls)

    def test_unsupported_transition_and_real_profile_gap_reset(self):
        m = current.Model()
        m.update(supported(1, [2.0, 0.2, 0.0, 0.0]), 0.0)
        m.update(current.rust_fit([], False, False), 10.0)
        self.assertEqual(m.anchor, current.PRIOR)
        m.update(supported(1, [2.0, 0.2, 0.0, 0.0]), 11.0)
        self.assertEqual(m.anchor, current.PRIOR)
        s = current.State()
        s.poll(0.0, 7, True, 5.0)
        s.models[1] = m
        s.poll(20.0, 7, False, 5.0)
        self.assertTrue(all(isinstance(x, current.Model) for x in s.models))
        self.assertTrue(all(x.anchor == current.PRIOR for x in s.models))
        s.models[1] = m
        s.poll(21.0, 8, True, 5.0)
        self.assertTrue(all(x.anchor == current.PRIOR for x in s.models))

    def state(self):
        s = current.State()
        for i in range(4):
            s.models[i].update(
                supported(i % 2, [2.0 + i / 10.0, 0.1 if i % 2 else 0.0, 0.0, 0.0]), 0.0
            )
        return s

    def test_closed_boundaries_and_outside_select_matching_queue(self):
        s = self.state()
        for x in (0.2, 0.8):
            p = s.predictions(snapshot([x, None, None]), 20.0, True)[1]
            self.assertEqual(p["source_arm"], "exact_early")
            self.assertIsNone(p["fallback_reason"])
            self.assertTrue(p["requested_within_fitted_ranges"])
        for x in (math.nextafter(0.2, 0.0), math.nextafter(0.8, 1.0)):
            p = s.predictions(snapshot([x, None, None]), 20.0, True)
            for i in (1, 3):
                self.assertEqual(p[i]["source_arm"], current.ARMS[i - 1])
                self.assertEqual(p[i]["extra"], p[i - 1]["extra"])
                self.assertEqual(p[i]["fallback_reason"], "outside_fitted_ranges")
                self.assertFalse(p[i]["requested_within_fitted_ranges"])
                self.assertTrue(p[i]["within_fitted_ranges"])

    def test_missing_invalid_unsupported_and_snapshot_priority(self):
        s = self.state()
        for x in (None, float("nan"), -1.0, 2.0):
            p = s.predictions(snapshot([x, None, None]), 20.0, True)[1]
            self.assertEqual(p["fallback_reason"], "missing_context")
            self.assertIsNone(p["requested_within_fitted_ranges"])
        s.models[0] = current.Model()
        p = s.predictions(snapshot([0.9, None, None]), 20.0, True)[1]
        self.assertTrue(p["scalar_fallback"])
        self.assertEqual(p["source_arm"], "live_n_scalar")
        self.assertEqual(p["fallback_reason"], "outside_fitted_ranges")
        self.assertEqual(p["extra"], 300.0)
        p = s.predictions(snapshot([0.9, None, None]), 20.0, False)[1]
        self.assertEqual(p["fallback_reason"], "snapshot_unavailable")
        self.assertFalse(p["available"])
        s.models[1] = current.Model()
        p = s.predictions(snapshot([0.5, None, None]), 20.0, True)[1]
        self.assertEqual(p["fallback_reason"], "early_fit_unsupported")
        self.assertIsNone(p["requested_within_fitted_ranges"])

    def test_mask_zero_is_vacuously_in_range_and_flags_do_not_mutate_v3(self):
        s = self.state()
        s.models[1].update(supported(0, [2.0, 0.0, 0.0, 0.0]), 20.0)
        p = s.predictions(snapshot([None, None, None]), 20.0, True)[1]
        self.assertTrue(p["requested_within_fitted_ranges"])
        self.assertEqual(p["source_arm"], "exact_early")
        old = previous.State()
        self.assertTrue(all(type(m) is legacy.Model for m in old.models))
        ablation = current.State(preserve_transitions=False, range_fallback=False)
        self.assertTrue(all(type(m) is legacy.Model for m in ablation.models))

    def test_historical_rebinding_requires_exact_digest_and_cannot_cover_other_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "historical.py"
            old.write_text("old planner")
            changed = root / "scripts/run_wait_production_mix.py"
            changed.parent.mkdir()
            changed.write_text("new planner")
            verify_binding(str(changed), sha(old), old)
            with self.assertRaises(ValueError):
                verify_binding(str(changed), sha(changed), old)
            other = root / "router-events.jsonl"
            other.write_text("changed capture")
            with self.assertRaises(ValueError):
                verify_binding(str(other), sha(old), old)

    def test_v4_audit_rejects_missing_provenance_and_mixed_methods(self):
        events = event_fixture()
        s = current.State()
        s.poll(0.0, 7, True, 5.0)
        for e in events:
            if "decision_json" not in e:
                continue
            e["decision_method"] = current.METHOD
            p = json.loads(e["decision_json"])
            if p.get("snapshot") is not None:
                p["snapshot"] = direct.snapshot_from_v1(p["snapshot"])
                if "predictions" in p:
                    p["predictions"] = s.predictions(p["snapshot"], p["snapshot"]["at"], True)
            if "state" in p:
                p["state"]["coefficient_ridge"] = current.COEFFICIENT_RIDGE
            if "label" in p:
                p["label"]["snapshot"] = direct.snapshot_from_v1(p["label"]["snapshot"])
            e["decision_json"] = json.dumps(p)
        self.assertTrue(replay(events, current.METHOD)["ok"])
        self.assertTrue(audit_schema(events, current.METHOD)["ok"])
        for field in ("fallback_reason", "requested_within_fitted_ranges"):
            changed = deepcopy(events)
            e = next(e for e in changed if e["event"] == "decision_admission")
            p = json.loads(e["decision_json"])
            del p["predictions"][0][field]
            e["decision_json"] = json.dumps(p)
            self.assertFalse(audit_schema(changed, current.METHOD)["ok"])
        events[1]["decision_method"] = previous.METHOD
        self.assertFalse(audit_schema(events, current.METHOD)["ok"])
        self.assertTrue(replay(event_fixture())["ok"])


if __name__ == "__main__":
    unittest.main()
