import copy
import unittest

from scripts.analyze_wait_policy import errors, replay, scorecard, sweep
from scripts.test_wait_calibration_mse import mse_trace


class PolicyTests(unittest.TestCase):
    def fixture(self):
        log = mse_trace()
        for e in log:
            if e.get("event") == "calibration_forecast":
                e["ready"] = True
        rows = [
            dict(
                opc_request_id=e["request_id"],
                started_elapsed_seconds=e["calibration_time_seconds"],
                phase="test",
                smg_ttft_seconds=1.0,
            )
            for e in log
            if e.get("event") == "calibration_forecast"
        ]
        return log, rows

    def test_live_reproduction_and_no_retroactive_fit(self):
        log, rows = self.fixture()
        replayed, _, _ = replay(log, rows)
        by_id = {r["opc_request_id"]: r for r in replayed}
        self.assertAlmostEqual(by_id["before-poll"]["mse_poll"], 4.2)
        self.assertAlmostEqual(by_id["after-poll"]["mse_poll"], 4.6)
        self.assertGreater(by_id["before-poll"]["median_feedback"], 4.2)
        self.assertEqual(by_id["0"]["mse_feedback"], by_id["0"]["raw"])

    def test_invalid_future_fit_rejected(self):
        log, rows = self.fixture()
        log[-2]["calibration_time_seconds"] = 43.0
        with self.assertRaises(ValueError):
            replay(log, rows)

    def test_epoch_reset_does_not_reuse_gain(self):
        log, rows = self.fixture()
        new = copy.deepcopy(log[0])
        new.update(request_id="reset", calibration_epoch=1, calibration_time_seconds=1.0)
        log.append(new)
        rows.append(dict(opc_request_id="reset", started_elapsed_seconds=100.0, phase="test"))
        result, _, _ = replay(log, rows)
        self.assertEqual(result[-1]["median_feedback"], result[-1]["raw"])

    def test_offered_denominator_keeps_rejects_and_timeouts(self):
        rows = [
            dict(complete=True, e2e_seconds=20, first_answer_seconds=None, finish_reason="length"),
            dict(complete=True, e2e_seconds=30, first_answer_seconds=2, finish_reason="stop"),
            dict(timed_out=True),
            dict(controlled_rejection=True),
        ]
        result = scorecard(rows, 300)
        self.assertEqual(result["completed_per_offered"], 0.5)
        self.assertEqual(result["content_completions_per_offered"], 0.25)
        self.assertEqual(result["natural_finish_with_content"], 1)

    def test_sweep_censoring_unknowns_and_readiness(self):
        rows = [
            dict(gate_predictions={"raw": 250}, gate_ready=True, completion_late=True),
            dict(gate_predictions={"raw": 250}, gate_ready=True, completion_late=False),
            dict(gate_predictions={"raw": 250}, gate_ready=False, completion_late=True),
            dict(
                gate_predictions={"raw": 400},
                gate_ready=True,
                completion_late=None,
                controlled_rejection=True,
            ),
        ]
        result = sweep(rows, 300, [60])[0]
        self.assertEqual(
            [result[k] for k in ["tp", "fp", "fn", "unknown", "unready"]], [1, 1, 1, 1, 1]
        )

    def test_reserve_never_substitutes_selected_forecast_for_gate(self):
        rows = [
            dict(raw=900, gate_predictions={"raw": 1}, gate_ready=True, completion_late=False),
            dict(raw=900, gate_ready=True, completion_late=False),
        ]
        result = sweep(rows, 300, [0])[0]
        self.assertEqual((result["tn"], result["fp"], result["unknown"]), (1, 0, 1))

    def test_signed_tail_uses_all_exact_pairs(self):
        rows = [dict(raw=2, smg_ttft_seconds=1), dict(raw=0, smg_ttft_seconds=10), dict(raw=None)]
        result = errors(rows, "raw")
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["p95_under"], 10)
        self.assertEqual(result["mae"], 5.5)


if __name__ == "__main__":
    unittest.main()
