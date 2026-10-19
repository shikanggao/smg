"""Artist regression: provisional totals stay visible without raw measured B."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import plot_wait_production_mix as plot


class TTFTPlotTests(unittest.TestCase):
    def test_independent_exact_totals_and_linear_client_scatter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = dict(measurement_target="ttft", deadline_seconds=300, maximum_p99_dispatch_lag_seconds=.25,
                cohorts=[dict(name="rag")], phases=[dict(name="pressure", seconds=10, rps=1)])
            row = dict(cohort="rag", phase="pressure", input_words=20000, max_output_tokens=32,
                started_elapsed_seconds=5, scheduled_elapsed_seconds=5, client_ttft_seconds=310,
                smg_ttft_seconds=309.9, raw_ttft_seconds=None, calibrated_ttft_seconds=999,
                pre_admission_total_ttft_predictions=dict(raw=None, queue_only_raw=350, calibrated=320),
                pre_admission_components=dict(calibration_baseline_seconds=.2, queue_excess_seconds=349.8,
                    calibration_scale=.9, calibration_epoch=1, calibration_bank_index=0),
                e2e_seconds=310.01, intentional_stream_abort=True, complete=False, timed_out=False,
                completion_late=None, first_text_late=True, dispatch_lag_seconds=.001)
            sample = dict(sample_completed_elapsed_seconds=5, raw_ttft=1234, baseline=None, data_usable=1,
                calibrated_ttft=999, native_running=2, native_waiting=3, native_kv=.2, client_active=5)
            for name, value in [("profile.json", profile), ("summary.json", dict(smoke=True, execution_status="PASS", production_coverage="INSUFFICIENT")),
                                ("run-config.json", dict(engine="mock", run_id="LOCAL_FIXTURE")), ("stages.json", [])]:
                (root/name).write_text(json.dumps(value))
            for name, value in [("plan.jsonl", row), ("paired-production.jsonl", row), ("metrics.jsonl", sample)]:
                (root/name).write_text(json.dumps(value) + "\n")
            figures = {}
            def capture(fig, filename, **unused):
                figures[Path(filename).name] = fig
            with patch("matplotlib.figure.Figure.savefig", capture):
                manifest = plot.render(root, root/"graphs", linear=True)
            self.assertEqual(manifest["plot_schema_version"], 4)
            timeline = figures["production-mix-timeline.png"].axes[0]
            series = {artist.get_label(): artist.get_offsets().tolist() for artist in timeline.collections}
            queue = next(v for k,v in series.items() if k.startswith("Queue-only uncalibrated"))
            cal = next(v for k,v in series.items() if k.startswith("Queue-only calibrated"))
            raw = next(v for k,v in series.items() if k.startswith("Raw floor-inclusive"))
            self.assertEqual(queue, [[5.0, 350.0]])
            self.assertEqual(cal, [[5.0, 320.0]])
            self.assertEqual(raw, [])
            self.assertFalse(any(line.get_label().endswith("sampled gauge") for line in timeline.lines))
            accuracy = figures["production-mix-accuracy.png"].axes[0]
            points = {artist.get_label(): artist.get_offsets().tolist() for artist in accuracy.collections}
            self.assertEqual(next(v for k,v in points.items() if k.startswith("Queue-only uncalibrated")), [[310.0, 350.0]])
            self.assertEqual(next(v for k,v in points.items() if k.startswith("Queue-only calibrated")), [[310.0, 320.0]])
            self.assertEqual(accuracy.get_xscale(), "linear")
            self.assertEqual(accuracy.get_yscale(), "linear")


if __name__ == "__main__":
    unittest.main()
