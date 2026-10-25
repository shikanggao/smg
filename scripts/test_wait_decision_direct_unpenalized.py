import json
import unittest
from copy import deepcopy

import wait_decision_calibration as legacy
import wait_decision_direct as direct
import wait_decision_direct_unpenalized as current
from replay_wait_decision import audit_schema, replay
from test_wait_decision_replay import event_fixture


class SlopeRidgeTests(unittest.TestCase):
    def samples(self, ratio, kind="exact"):
        return [
            dict(x=100.0, target=100.0 * ratio, values=[None, None, None], kind=kind)
            for _ in range(16)
        ]

    def test_queue_nonunit_gain_has_no_shrink_and_old_solver_is_unchanged(self):
        for q in (0.5, 2.0):
            rows = self.samples(q)
            result = current.fit(rows, early=False, censored=False)
            self.assertAlmostEqual(result["coefficients"][0], q, places=12)
            self.assertAlmostEqual(result["objective"], 0.0, places=20)
            self.assertAlmostEqual(
                legacy.fit(rows, early=False, censored=False)["coefficients"][0],
                (q + 0.1) / 1.1,
                places=12,
            )

    def test_censor_only_is_one_sided_not_an_exact_target(self):
        self.assertEqual(
            current.fit(self.samples(0.5, "lower_bound"), early=False, censored=True)[
                "coefficients"
            ][0],
            1.0,
        )
        self.assertAlmostEqual(
            current.fit(self.samples(2.0, "lower_bound"), early=False, censored=True)[
                "coefficients"
            ][0],
            2.0,
            places=12,
        )
        self.assertFalse(
            current.fit(self.samples(2.0, "lower_bound"), early=False, censored=False)["ready"]
        )

    def test_feature_slope_remains_penalized_and_base_absorbs_mean(self):
        rows = [
            dict(
                x=100.0,
                target=100.0 * (2.0 + 0.6 * (i % 2)),
                values=[float(i % 2), None, None],
                kind="exact",
            )
            for i in range(32)
        ]
        fit = current.fit(rows, early=True, censored=False)
        self.assertEqual(fit["mask"], 1)
        self.assertAlmostEqual(fit["coefficients"][1], 0.15 / 0.35, places=7)
        self.assertAlmostEqual(fit["coefficients"][0], 2.3 - 0.5 * (0.15 / 0.35), places=7)
        self.assertGreater(fit["objective"], 0.0)

    def test_full_v3_audit_and_mixed_missing_ridge_rejected(self):
        events = event_fixture()
        state = current.State()
        state.poll(0.0, 7, True, 5.0)
        for e in events:
            if "decision_json" not in e:
                continue
            e["decision_method"] = current.METHOD
            p = json.loads(e["decision_json"])
            if p.get("snapshot") is not None:
                p["snapshot"] = direct.snapshot_from_v1(p["snapshot"])
                if "predictions" in p:
                    p["predictions"] = state.predictions(p["snapshot"], p["snapshot"]["at"], True)
            if "state" in p:
                p["state"]["coefficient_ridge"] = current.COEFFICIENT_RIDGE
            if "label" in p:
                p["label"]["snapshot"] = direct.snapshot_from_v1(p["label"]["snapshot"])
            e["decision_json"] = json.dumps(p)
        self.assertTrue(replay(events, current.METHOD)["ok"])
        self.assertTrue(audit_schema(events, current.METHOD)["ok"])
        changed = deepcopy(events)
        poll = next(e for e in changed if e["event"] == "decision_poll")
        p = json.loads(poll["decision_json"])
        del p["state"]["coefficient_ridge"]
        poll["decision_json"] = json.dumps(p)
        self.assertFalse(audit_schema(changed, current.METHOD)["ok"])
        changed = deepcopy(events)
        changed[1]["decision_method"] = direct.METHOD
        self.assertFalse(audit_schema(changed, current.METHOD)["ok"])
        self.assertTrue(replay(event_fixture())["ok"])


if __name__ == "__main__":
    unittest.main()
