"""V4 direct gain: compatible mask transitions and in-range early prediction.

V1–V3 remain immutable. Solver, labels, real resets and independent queue arms
are unchanged. Flags exist only for fixed offline ablations, not runtime tuning.
"""

import math

try:
    from . import wait_decision_calibration as legacy
    from . import wait_decision_direct_unpenalized as previous
except ImportError:
    import wait_decision_calibration as legacy
    import wait_decision_direct_unpenalized as previous

METHOD = "decision_live_queue_context_continuity_hinge_v4"
ARMS, PRIOR = previous.ARMS, previous.PRIOR
COEFFICIENT_RIDGE = previous.COEFFICIENT_RIDGE
fit, rust_fit = previous.fit, previous.rust_fit
FALLBACK_REASONS = (
    None,
    "snapshot_unavailable",
    "early_fit_unsupported",
    "missing_context",
    "outside_fitted_ranges",
    "fit_unsupported",
)


class Model(legacy.Model):
    def update(self, value, now):
        if self.fit["supported"] and value["supported"]:
            self.anchor = self.applied(now)
            common = self.fit["mask"] & value["mask"]
            for j in range(3):
                if not common & (1 << j):
                    self.anchor[j + 1] = 0.0
        else:
            self.anchor = PRIOR.copy()
        self.target, self.at, self.fit = value["coefficients"].copy(), now, value


class State(previous.State):
    def __init__(self, *, preserve_transitions=True, range_fallback=True):
        super().__init__()
        self.preserve_transitions = preserve_transitions
        self.range_fallback = range_fallback
        self.models = self._models()

    def _models(self):
        return [(Model if self.preserve_transitions else legacy.Model)() for _ in ARMS]

    def poll(self, now, profile, fresh, max_gap):
        if profile != self.profile:
            self.rows = []
            self.models = self._models()
            self.origin = None
            self.points = []
            self.profile = profile
            self.revision += 1
            self.fitted_revision = None
        self.expire(now)
        if not fresh or self.last_poll is not None and now - self.last_poll > max_gap:
            self.points = []
            self.models = self._models()
            self.fitted_revision = None
        self.last_poll = now
        if not fresh:
            return
        if not self.incomplete:
            self.points.append((now, len(self.pending)))
        self.points = [p for p in self.points if now - p[0] <= 60][-256:]
        if self.fitted_revision != self.revision:
            for i, model in enumerate(self.models):
                model.update(rust_fit(self.rows, i % 2 == 1, i >= 2), now)
            self.fitted_revision = self.revision

    def predictions(self, snapshot, now, available):
        values = snapshot["context"]["values"]
        result = []
        for i, requested in enumerate(self.models):
            z_requested = legacy.vector(values, requested.fit["mask"])
            within_requested = None
            if requested.fit["supported"] and z_requested is not None:
                within_requested = all(
                    not requested.fit["mask"] & (1 << j)
                    or requested.fit["minimum"][j]
                    <= z_requested[j + 1]
                    <= requested.fit["maximum"][j]
                    for j in range(3)
                )
            reason = None
            index = i
            if i % 2:
                if not requested.fit["supported"]:
                    reason = "early_fit_unsupported"
                elif z_requested is None:
                    reason = "missing_context"
                elif self.range_fallback and within_requested is False:
                    reason = "outside_fitted_ranges"
                if reason is not None:
                    index = i - 1
            model = self.models[index]
            if not available:
                reason = "snapshot_unavailable"
            elif reason is None and not model.fit["supported"]:
                reason = "fit_unsupported"
            fallback = not model.fit["supported"] or not available
            if fallback:
                gain = snapshot["scalar_gain"]
                unbounded = snapshot["x"] * gain
                extra = max(0.0, unbounded)
                coeff = [gain, 0.0, 0.0, 0.0]
                contributions = [extra, 0.0, 0.0, 0.0]
                source, within, clipped = "live_n_scalar", True, False
                prediction_available = available and math.isfinite(unbounded)
            else:
                z = legacy.vector(values, model.fit["mask"]) or PRIOR.copy()
                coeff = model.applied(now)
                contributions = [snapshot["x"] * c * v for c, v in zip(coeff, z)]
                unbounded = snapshot["x"] * sum(c * v for c, v in zip(coeff, z))
                _, shift, bounded = legacy.bounded_gain(coeff, z)
                if shift:
                    for j in range(1, 4):
                        contributions[j] *= bounded / shift
                extra = max(0.0, sum(contributions))
                source = ARMS[index]
                within = all(
                    not model.fit["mask"] & (1 << j)
                    or model.fit["minimum"][j] <= z[j + 1] <= model.fit["maximum"][j]
                    for j in range(3)
                )
                clipped = shift != bounded
                prediction_available = available and math.isfinite(extra)
            result.append(
                dict(
                    arm=ARMS[i],
                    source_arm=source,
                    scalar_fallback=fallback,
                    coefficients=coeff,
                    contributions=contributions,
                    extra=extra,
                    total=snapshot["baseline"] + extra,
                    supported=not fallback,
                    within_fitted_ranges=within,
                    available=prediction_available,
                    unbounded_extra=unbounded,
                    contextual_clip_applied=clipped,
                    fallback_reason=reason,
                    requested_within_fitted_ranges=within_requested,
                )
            )
        return result
