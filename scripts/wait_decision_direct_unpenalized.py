"""Versioned direct-gain solver: only contextual slopes receive ridge.

Keep v1/v2 immutable. The conservative step bound, finite iteration count,
projection, label lifecycle and capped/slewed prediction remain unchanged.
"""

import math

try:
    from . import wait_decision_calibration as legacy
    from . import wait_decision_direct as direct
except ImportError:
    import wait_decision_calibration as legacy
    import wait_decision_direct as direct
PRIOR, RIDGE, ITERATIONS, MASKS, ARMS = (
    legacy.PRIOR,
    legacy.RIDGE,
    legacy.ITERATIONS,
    legacy.MASKS,
    legacy.ARMS,
)
project, bounded_gain, Model = legacy.project, legacy.bounded_gain, legacy.Model
snapshot_from_v1 = direct.snapshot_from_v1
COEFFICIENT_RIDGE = [0.0, 0.1, 0.1, 0.1]
METHOD = "decision_live_queue_slope_ridge_hinge_v3"


def fit(samples, *, early, censored):
    """Fit half-squared exact/hinge loss plus half-squared ridge to neutral.

    Samples: x>0, target>=0, kind exact/lower_bound, values [G,A,P].
    G/A/P may be None. Exclude unavailable features by fitting a supported subset.
    A feature arm without supported variation returns its matching queue fit.
    """
    eligible = []
    for sample in samples:
        if sample["kind"] not in ("exact", "lower_bound"):
            raise ValueError("Unknown label kind")
        if sample["kind"] == "lower_bound" and not censored:
            continue
        if (
            any(
                type(sample[k]) not in (int, float) or not math.isfinite(sample[k])
                for k in ("x", "target")
            )
            or sample["x"] <= 0
            or sample["target"] < 0
        ):
            raise ValueError("Invalid original input or target")
        if len(sample["values"]) != 3:
            raise ValueError("Expected three early inputs")
        eligible.append(sample)
    for mask in MASKS if early else (0,):
        selected = [
            r
            for r in eligible
            if all(
                not mask & (1 << j)
                or type(r["values"][j]) in (int, float)
                and math.isfinite(r["values"][j])
                and 0 <= r["values"][j] <= 1
                for j in range(3)
            )
        ]
        if len(selected) < 12:
            continue
        largest = max(r["x"] for r in selected)
        weights = [(r["x"] / largest) ** 2 for r in selected]
        total = sum(weights)
        weights = [w / total for w in weights]
        ess = 1 / sum(w * w for w in weights)
        if ess < 8:
            continue
        minimum, maximum, variance = [None] * 3, [None] * 3, [None] * 3
        for j in range(3):
            if mask & (1 << j):
                mean = sum(w * r["values"][j] for w, r in zip(weights, selected))
                minimum[j] = min(r["values"][j] for r in selected)
                maximum[j] = max(r["values"][j] for r in selected)
                variance[j] = sum(
                    w * (r["values"][j] - mean) ** 2 for w, r in zip(weights, selected)
                )
        if any(
            mask & (1 << j) and (variance[j] < 1e-4 or maximum[j] - minimum[j] < 0.05)
            for j in range(3)
        ):
            continue
        vectors = [
            [1.0] + [r["values"][j] if mask & (1 << j) else 0.0 for j in range(3)] for r in selected
        ]
        step = 1 / (
            RIDGE + sum(w * sum(z * z for z in vector) for w, vector in zip(weights, vectors))
        )

        def gradient(theta):
            result = [penalty * (c - p) for penalty, c, p in zip(COEFFICIENT_RIDGE, theta, PRIOR)]
            for w, vector, sample in zip(weights, vectors, selected):
                residual = (
                    sum(c * z for c, z in zip(theta, vector)) - sample["target"] / sample["x"]
                )
                if sample["kind"] == "lower_bound":
                    residual = min(0.0, residual)
                for j in range(4):
                    result[j] += w * residual * vector[j]
            return result

        coefficients = PRIOR.copy()
        for _ in range(ITERATIONS):
            coefficients = project(
                [c - step * g for c, g in zip(coefficients, gradient(coefficients))]
            )
        next_values = project([c - step * g for c, g in zip(coefficients, gradient(coefficients))])
        projected_gradient = max(abs((c - n) / step) for c, n in zip(coefficients, next_values))
        penalty = 0.5 * sum(
            r * (c - p) ** 2 for r, c, p in zip(COEFFICIENT_RIDGE, coefficients, PRIOR)
        )
        objectives = [penalty, penalty]
        clipped = 0
        for w, vector, sample in zip(weights, vectors, selected):
            bounded, shift, limited = bounded_gain(coefficients, vector)
            clipped += shift != limited
            for j, value in enumerate((sum(c * z for c, z in zip(coefficients, vector)), bounded)):
                residual = value - sample["target"] / sample["x"]
                if sample["kind"] == "lower_bound":
                    residual = min(0.0, residual)
                objectives[j] += 0.5 * w * residual**2
        return dict(
            ready=True,
            coefficients=coefficients,
            mask=mask,
            count=len(selected),
            ess=ess,
            minimum=minimum,
            maximum=maximum,
            variance=variance,
            exact_count=sum(r["kind"] == "exact" for r in selected),
            lower_bound_count=sum(r["kind"] == "lower_bound" for r in selected),
            iterations=ITERATIONS,
            step=step,
            objective=objectives[0],
            bounded_training_objective=objectives[1],
            training_clipped_count=clipped,
            projected_gradient=projected_gradient,
        )
    return dict(
        ready=False,
        coefficients=PRIOR.copy(),
        mask=0,
        count=0,
        ess=0.0,
        minimum=[None] * 3,
        maximum=[None] * 3,
        variance=[None] * 3,
        exact_count=0,
        lower_bound_count=0,
        iterations=0,
        step=None,
        objective=0.0,
        bounded_training_objective=0.0,
        training_clipped_count=0,
        projected_gradient=0.0,
    )


def rust_fit(labels, early, censored):
    samples = [
        dict(
            x=r["snapshot"]["x"],
            target=max(0.0, r["elapsed"] - r["snapshot"]["baseline"]),
            values=r["snapshot"]["context"]["values"],
            kind="exact" if r["exact"] else "lower_bound",
        )
        for r in labels
    ]
    value = fit(samples, early=early, censored=censored)
    return dict(
        coefficients=value["coefficients"],
        mask=value["mask"],
        count=value["count"],
        exact_count=value["exact_count"],
        bound_count=value["lower_bound_count"],
        ess=value["ess"],
        minimum=[v if v is not None else 0.0 for v in value["minimum"]],
        maximum=[v if v is not None else 0.0 for v in value["maximum"]],
        supported=value["ready"],
        **{
            k: value[k]
            for k in (
                "objective",
                "projected_gradient",
                "bounded_training_objective",
                "training_clipped_count",
            )
        },
    )


class State(direct.State):
    def poll(self, now, profile, fresh, max_gap):
        if profile != self.profile:
            self.rows = []
            self.models = [Model() for _ in ARMS]
            self.origin = None
            self.points = []
            self.profile = profile
            self.revision += 1
            self.fitted_revision = None
        self.expire(now)
        if not fresh or self.last_poll is not None and now - self.last_poll > max_gap:
            self.points = []
            self.models = [Model() for _ in ARMS]
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

    def diagnostics(self):
        result = super().diagnostics()
        result["coefficient_ridge"] = COEFFICIENT_RIDGE.copy()
        return result
