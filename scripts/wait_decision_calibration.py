"""Independent fixed solver for decision-focused shadow calibration.

Input samples retain their original admission extra estimate x and early values.
Targets are observed extra-delay values or honest lower bounds, never invented
numeric TTFT values for censored requests. This module does not publish labels.
"""

import math
from copy import deepcopy

PRIOR = [1.0, 0.0, 0.0, 0.0]
RIDGE = 0.1
ITERATIONS = 256
MASKS = (7, 6, 5, 3, 4, 2, 1, 0)
METHOD = "decision_live_scalar_early_hinge_v1"
ARMS = ("exact_queue", "exact_early", "bound_queue", "bound_early")


def project(values):
    return [
        max(0.0 if j == 0 else -2.0, min(4.0 if j == 0 else 2.0, v)) for j, v in enumerate(values)
    ]


def bounded_gain(coefficients, vector):
    shift = sum(c * z for c, z in zip(coefficients, vector)) - coefficients[0]
    bounded = max(-coefficients[0], max(-0.5, min(0.5, shift)))
    return coefficients[0] + bounded, shift, bounded


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
            result = [RIDGE * (c - p) for c, p in zip(theta, PRIOR)]
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
        penalty = 0.5 * RIDGE * sum((c - p) ** 2 for c, p in zip(coefficients, PRIOR))
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


def vector(values, mask):
    result = PRIOR.copy()
    for j, value in enumerate(values):
        if mask & (1 << j):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                return None
            result[j + 1] = value
    return result


class Model:
    def __init__(self):
        self.fit = rust_fit([], False, False)
        self.anchor, self.target, self.at = PRIOR.copy(), PRIOR.copy(), 0.0

    def applied(self, now):
        distance = sum(abs(t - a) for t, a in zip(self.target, self.anchor))
        fraction = min(1.0, 0.5 * max(0.0, now - self.at) / distance) if distance else 1.0
        return [a + fraction * (t - a) for a, t in zip(self.anchor, self.target)]

    def update(self, value, now):
        self.anchor = (
            self.applied(now)
            if (self.fit["mask"] == value["mask"] and self.fit["supported"] and value["supported"])
            else PRIOR.copy()
        )
        self.target, self.at, self.fit = value["coefficients"].copy(), now, value


class State:
    """Independent causal mirror; callers supply observed events in publication order."""

    def __init__(self):
        self.pending, self.incomplete, self.profile = {}, False, None
        self.points, self.last_poll, self.rows, self.origin = [], None, [], None
        self.models = [Model() for _ in ARMS]
        self.revision, self.fitted_revision = 0, None

    def register(self, identity, started, snapshot):
        if identity is None or snapshot is None or len(self.pending) >= 8192:
            self.incomplete = True
            return False
        self.pending[identity] = dict(
            started=started, snapshot=deepcopy(snapshot), first_label=None
        )
        return True

    def context(self, now, progress=None, capacity=None, backlog=False):
        pending = len(self.pending)
        complete = not self.incomplete
        oldest = (
            max((max(0.0, now - p["started"]) for p in self.pending.values()), default=0.0)
            if complete
            else None
        )
        reference = (
            next(((at, n) for at, n in reversed(self.points) if 30 <= now - at <= 60), None)
            if complete
            else None
        )
        growth = max(0.0, pending - reference[1]) / max(1, pending) if reference else None
        deficit = None
        if (
            complete
            and pending > 0
            and backlog
            and all(type(v) in (int, float) and math.isfinite(v) for v in (progress, capacity))
            and progress >= 0
            and capacity > 0
        ):
            deficit = max(0.0, min(1.0, 1 - progress / capacity))
        return dict(
            values=[growth, min(oldest / 300, 1.0) if oldest is not None else None, deficit],
            pending=pending,
            complete=complete,
            oldest_seconds=oldest,
            growth_reference_time=reference[0] if reference else None,
            progress_tokens_per_second=progress,
            progress_capacity=capacity,
        )

    def expire(self, now):
        if self.origin is None:
            return
        bank = math.floor((now - self.origin) / 300)
        old = len(self.rows)
        self.rows = [
            r
            for r in self.rows
            if math.floor((r["first_label_at"] - self.origin) / 300) >= bank - 1
            and now - r["snapshot"]["at"] < 600
        ]
        if old != len(self.rows):
            self.revision += 1

    def label(self, identity, observed_at, now, exact, terminal, bound=None):
        pending = self.pending.pop(identity, None) if terminal else self.pending.get(identity)
        if pending is None:
            return None
        elapsed = bound if bound is not None else observed_at - pending["started"]
        s = pending["snapshot"]
        if (
            not s["eligible"]
            or self.profile != s["profile"]
            or not 0 <= now - s["at"] < 600
            or not math.isfinite(elapsed)
            or elapsed <= 0
            or s["x"] < 0.001
        ):
            return None
        self.expire(now)
        first = pending["first_label"] if pending["first_label"] is not None else now
        prior = next((i for i, r in enumerate(self.rows) if r["id"] == identity), None)
        if pending["first_label"] is not None and prior is None:
            return None
        if prior is not None and (
            self.rows[prior]["exact"] or not exact and self.rows[prior]["elapsed"] >= elapsed
        ):
            return None
        if self.origin is None:
            self.origin = now
        label = dict(
            id=identity,
            at=now,
            observed_at=observed_at,
            elapsed=elapsed,
            exact=exact,
            first_label_at=first,
            snapshot=deepcopy(s),
        )
        if prior is not None:
            self.rows[prior] = label
        else:
            self.rows.append(label)
        if identity in self.pending:
            self.pending[identity]["first_label"] = first
        bank = math.floor((first - self.origin) / 300)
        while (
            sum(math.floor((r["first_label_at"] - self.origin) / 300) == bank for r in self.rows)
            > 32
        ):
            self.rows.pop(
                next(
                    i
                    for i, r in enumerate(self.rows)
                    if math.floor((r["first_label_at"] - self.origin) / 300) == bank
                )
            )
        self.revision += 1
        return label

    def terminal_unknown(self, identity):
        self.pending.pop(identity, None)

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

    def predictions(self, snapshot, now, available):
        values = snapshot["context"]["values"]
        result = []
        for i, requested in enumerate(self.models):
            index = (
                i - 1
                if i % 2
                and (
                    not requested.fit["supported"] or vector(values, requested.fit["mask"]) is None
                )
                else i
            )
            model = self.models[index]
            z = vector(values, model.fit["mask"]) or PRIOR.copy()
            coeff = model.applied(now)
            contributions = [snapshot["x"] * c * v for c, v in zip(coeff, z)]
            unbounded = snapshot["x"] * sum(c * v for c, v in zip(coeff, z))
            _, shift, bounded = bounded_gain(coeff, z)
            if shift:
                for j in range(1, 4):
                    contributions[j] *= bounded / shift
            extra = max(0.0, sum(contributions))
            within = all(
                not model.fit["mask"] & (1 << j)
                or model.fit["minimum"][j] <= z[j + 1] <= model.fit["maximum"][j]
                for j in range(3)
            )
            result.append(
                dict(
                    arm=ARMS[i],
                    source_arm=ARMS[index],
                    coefficients=coeff,
                    contributions=contributions,
                    extra=extra,
                    total=snapshot["baseline"] + extra,
                    supported=model.fit["supported"],
                    within_fitted_ranges=within,
                    available=available and math.isfinite(extra),
                    unbounded_extra=unbounded,
                    contextual_clip_applied=shift != bounded,
                )
            )
        return result

    def mark_incomplete(self):
        self.incomplete = True
        self.points = []

    def diagnostics(self):
        return dict(
            models=[
                dict(fit=m.fit, anchor=m.anchor, target=m.target, at=m.at) for m in self.models
            ],
            rows=len(self.rows),
            origin=self.origin,
            profile=self.profile,
            revision=self.revision,
            fitted_revision=self.fitted_revision,
            pending=len(self.pending),
            complete=not self.incomplete,
            ridge=RIDGE,
            min_x_seconds=0.001,
            iterations=ITERATIONS,
            coefficient_l1_per_second=0.5,
            contextual_gain_cap=0.5,
            domain=[0.0, 1.0],
        )
