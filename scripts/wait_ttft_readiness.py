"""Fail-closed readiness for an isolated, diagnostic TTFT experiment."""
from __future__ import annotations

import math


METHOD = "mse_poll_two_wave_queue_v2"


class Readiness:
    def __init__(self, request_prefix, scenario="isolated"):
        assert scenario in {"isolated", "busy_start"}
        self.prefix = request_prefix
        self.scenario = scenario
        self.epochs, self.forecasts, self.fits, self.initial = {}, {}, set(), {}
        self.errors = []
        self.accepted = {}

    def observe(self, event):
        kind, worker = event.get("event"), event.get("worker")
        if not worker or kind not in {"forecast", "calibration_forecast", "calibration_feedback", "calibration_update", "calibration_reset"}:
            return
        if kind == "forecast" and not event.get("calibration_enabled"):
            return
        if kind != "calibration_reset" and event.get("calibration_method") != METHOD:
            self.errors.append("Unexpected calibration method")
            return
        rid = event.get("request_id")
        if rid and not rid.startswith(self.prefix):
            self.errors.append("Feedback or forecast from outside this isolated run")
        epoch = event.get("current_calibration_epoch", event.get("calibration_epoch"))
        if not isinstance(epoch, int):
            self.errors.append("Missing calibration epoch")
            return
        self.epochs[worker] = epoch
        if kind in {"forecast", "calibration_forecast"}:
            self.initial.setdefault(worker, all(event.get(k) == 0 for k in (
                "calibration_observation_sequence", "calibration_fit_sequence",
                "calibration_history_revision", "calibration_fitted_revision")))
            self.forecasts[worker] = event
        elif kind == "calibration_update":
            if (event.get("calibration_state") == "capacity_ready"
                    and event.get("sample_count_after", 0) >= 4
                    and event.get("fit_sequence_after", 0) >= 4
                    and event.get("calibration_fitted_revision", 0) > 0):
                self.fits.add((worker, epoch, event["calibration_fitted_revision"]))
        elif kind == "calibration_feedback" and event.get("reason") == "accepted":
            self.accepted[(worker, epoch)] = self.accepted.get((worker, epoch), 0) + 1

    def snapshot(self, native_idle_verified, native_busy_verified=False):
        workers = []
        for worker, epoch in self.epochs.items():
            f = self.forecasts.get(worker, {})
            baseline = f.get("calibration_baseline_seconds")
            baseline_known = (f.get("calibration_baseline_available") is True
                and f.get("calibration_baseline_source") in {"provisional", "measured"}
                and isinstance(baseline, (int, float)) and math.isfinite(baseline) and baseline >= 0)
            revision = f.get("calibration_fitted_revision", 0)
            checks = dict(empty_initial_history=self.initial.get(worker) is True,
                current_epoch=f.get("calibration_epoch") == epoch,
                baseline_with_provenance=baseline_known,
                native_capacity_ready=f.get("native_ready") is True,
                current_samples_at_least_four=f.get("calibration_sample_count", 0) >= 4,
                accepted_in_current_epoch=self.accepted.get((worker, epoch), 0) >= 4,
                actual_current_epoch_fit=(worker, epoch, revision) in self.fits)
            workers.append(dict(worker=worker, epoch=epoch, checks=checks,
                baseline_seconds=baseline, baseline_source=f.get("calibration_baseline_source"),
                samples=f.get("calibration_sample_count"), fitted_revision=revision))
        startup = native_busy_verified if self.scenario == "busy_start" else native_idle_verified
        return dict(ready=bool(workers) and startup and not self.errors
                    and all(all(w["checks"].values()) for w in workers),
                    scenario=self.scenario, initial_native_idle_verified=native_idle_verified,
                    initial_native_busy_verified=native_busy_verified, workers=workers, errors=self.errors[:20])
