"""Versioned direct queue calibration; v1 solver/lifecycle remain immutable."""

import math
from copy import deepcopy

try:
    from . import wait_decision_calibration as legacy
except ImportError:
    import wait_decision_calibration as legacy

METHOD = "decision_live_queue_early_hinge_v2"
ARMS, PRIOR = legacy.ARMS, legacy.PRIOR


def snapshot_from_v1(snapshot):
    """Change only the training basis and its eligibility at the saved clock."""
    if snapshot is None:
        return None
    value = deepcopy(snapshot)
    value["x"] = (
        value["live_queue_extra"]
        if value["live_queue_extra"] is not None
        else value["original_queue_extra"]
    )
    value["source"] = "live_n" if value["live_queue_extra"] is not None else "original_n_fallback"
    value["eligible"] = (
        value["fresh"]
        and value["native_ready"]
        and value["native_waiting"] > 0
        and math.isfinite(value["baseline"])
        and math.isfinite(value["x"])
        and value["x"] >= 0.001
    )
    return value


class State(legacy.State):
    def predictions(self, snapshot, now, available):
        result = super().predictions(snapshot, now, available)
        for prediction in result:
            fallback = not prediction["supported"] or not available
            prediction["scalar_fallback"] = fallback
            if fallback:
                gain = snapshot["scalar_gain"]
                unbounded = snapshot["x"] * gain
                extra = max(0.0, unbounded)
                prediction.update(
                    source_arm="live_n_scalar",
                    coefficients=[gain, 0.0, 0.0, 0.0],
                    contributions=[extra, 0.0, 0.0, 0.0],
                    extra=extra,
                    total=snapshot["baseline"] + extra,
                    supported=False,
                    within_fitted_ranges=True,
                    available=available and math.isfinite(unbounded),
                    unbounded_extra=unbounded,
                    contextual_clip_applied=False,
                )
        return result
