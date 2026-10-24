"""Deterministic v3 joint per-feature gain arithmetic, independent of Rust.

Training consumes only admitted, accepted exact first-text labels. Censored
300/420-second observations are never numeric regression labels.
"""
from __future__ import annotations
import math

METHOD = "feature_joint_gains_deadline_ridge_v3"
PRIOR = [1., 0., 0., 0.]
RIDGE = .1
MAX_SHIFT = .5
RATE = .5


def empty_fit(revision=0):
    return dict(coefficients=PRIOR.copy(), beta=[0.] * 3, mean=[0.] * 3,
                minimum=[0.] * 3, maximum=[0.] * 3, support=[False] * 3,
                count=0, ess=0., revision=revision)


def solve(a, b):
    size = len(b)
    l = [[0.] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            v = a[i][j] - sum(l[i][k] * l[j][k] for k in range(j))
            if i == j:
                if not math.isfinite(v) or v <= 0:
                    return None
                l[i][j] = math.sqrt(v)
            else:
                l[i][j] = v / l[j][j]
    y, x = [0.] * size, [0.] * size
    for i in range(size):
        y[i] = (b[i] - sum(l[i][j] * y[j] for j in range(i))) / l[i][i]
    for i in reversed(range(size)):
        x[i] = (y[i] - sum(l[j][i] * x[j] for j in range(i + 1, size))) / l[i][i]
    return x if all(math.isfinite(v) for v in x) else None


def learn(samples, revision=0):
    for mask in (7, 3, 5, 6, 1, 2, 4):
        rows = [r for r in samples if r.get('context') is not None
                and math.isfinite(r['ratio']) and math.isfinite(r['n']) and r['n'] > 0
                and all(not mask & (1 << j) or r['context'][j] is not None
                        and math.isfinite(r['context'][j]) and 0 <= r['context'][j] <= 1
                        for j in range(3))]
        if len(rows) < 12:
            continue
        largest = max(r['n'] for r in rows)
        weights = [(r['n'] / largest) ** 2 for r in rows]
        total = sum(weights)
        weights = [w / total for w in weights]
        f = empty_fit(revision)
        f.update(count=len(rows), ess=1 / sum(w*w for w in weights))
        if f['ess'] < 8:
            continue
        for j in range(3):
            if mask & (1 << j):
                f['mean'][j] = sum(w*r['context'][j] for w, r in zip(weights, rows))
                f['minimum'][j] = min(r['context'][j] for r in rows)
                f['maximum'][j] = max(r['context'][j] for r in rows)
        if any(mask & (1 << j) and (
                sum(w*(r['context'][j]-f['mean'][j])**2 for w,r in zip(weights,rows)) < 1e-4
                or f['maximum'][j] - f['minimum'][j] < .05) for j in range(3)):
            continue
        a, b = [[0.] * 4 for _ in range(4)], [0.] * 4
        for w, r in zip(weights, rows):
            x = [1.] + [r['context'][j] if mask & (1 << j) else 0. for j in range(3)]
            for j in range(4):
                b[j] += w*x[j]*r['ratio']
                for k in range(4):
                    a[j][k] += w*x[j]*x[k]
        for j in range(4):
            a[j][j] += RIDGE
            b[j] += RIDGE * PRIOR[j]
        coefficients = solve(a,b)
        if coefficients is not None:
            f['coefficients'] = [max(0., coefficients[0])] + [max(-2.,min(2.,v)) for v in coefficients[1:]]
            f['beta'] = f['coefficients'][1:]
            f['support'] = [bool(mask & (1 << j)) for j in range(3)]
            return f
    return empty_fit(revision)


def active_features(fit, context):
    return [fit['support'][j] and context[j] is not None
            and math.isfinite(context[j]) and fit['minimum'][j] <= context[j] <= fit['maximum'][j]
            for j in range(3)]


def applied(trajectory, now):
    anchor, target, at = trajectory
    distance = sum(abs(b-a) for a,b in zip(anchor,target))
    fraction = min(1., RATE * max(0., now-at) / distance) if distance else 1.
    return [a + fraction*(b-a) for a,b in zip(anchor,target)]


def effective(coefficients, context):
    shift = sum(coefficients[j+1] * (context[j] or 0.) for j in range(3))
    bounded = max(-coefficients[0], max(-MAX_SHIFT, min(MAX_SHIFT, shift)))
    factor = bounded / shift if shift else 1.
    return [coefficients[0]] + [v*factor for v in coefficients[1:]]


def gain(coefficients, context):
    return max(0., coefficients[0] + sum(coefficients[j+1]*(context[j] or 0.) for j in range(3)))


class Model:
    def __init__(self):
        self.fit = empty_fit()
        self.trajectory = None
        self.active = [False] * 3

    def invalidate(self):
        self.trajectory = None
        self.active = [False] * 3

    def update(self, context, scalar, now):
        active = active_features(self.fit, context)
        supported = any(self.fit['support']) and active == self.fit['support']
        if not supported:
            self.invalidate()
            return
        anchor = (applied(self.trajectory, now) if self.trajectory and self.active == active
                  else [scalar, 0., 0., 0.])
        self.active = active
        self.trajectory = (anchor, self.fit['coefficients'].copy(), now)

    def prediction(self, context, scalar, now, usable, n):
        active = active_features(self.fit, context)
        trajectory = self.trajectory if usable and active == self.active and active == self.fit['support'] and any(active) else None
        coefficients = applied(trajectory, now) if trajectory else [scalar,0.,0.,0.]
        target = trajectory[1] if trajectory else coefficients
        eff, teff = effective(coefficients, context), effective(target, context)
        contributions = [n*eff[0]] + [n*eff[j+1]*(context[j] or 0.) for j in range(3)]
        contributions = [v if math.isfinite(v) else None for v in contributions]
        return dict(gain=gain(eff,context), target_gain=gain(teff,context),
                    contextual=trajectory is not None,
                    reason='snapshot_unavailable' if not usable else 'context_supported' if trajectory else 'legacy_fallback_no_current_support',
                    active=active, applied_coefficients=coefficients, target_coefficients=target.copy(),
                    effective_coefficients=eff, target_effective_coefficients=teff,
                    contribution_seconds=contributions,
                    trajectory_anchor_coefficients=trajectory[0].copy() if trajectory else None,
                    trajectory_anchor_gain=gain(effective(trajectory[0],context),context) if trajectory else None,
                    trajectory_anchor_time_seconds=trajectory[2] if trajectory else None)


def validate_prediction(prediction, fit, context, scalar, now, usable, n):
    """Validate logged vector trajectory and each contribution, not only its sum."""
    def close(actual, expected, path):
        if isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                raise ValueError('Joint vector mismatch: ' + path)
            for j, (a,b) in enumerate(zip(actual,expected)):
                close(a,b,path+'['+str(j)+']')
        elif expected is None or isinstance(expected, (str,bool)):
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError('Joint state mismatch: ' + path)
        elif type(actual) not in (int,float) or not math.isfinite(actual) or not math.isclose(actual,expected,rel_tol=1e-8,abs_tol=1e-8):
            raise ValueError('Joint arithmetic mismatch: ' + path)
    coefficients = fit['coefficients']
    if (not isinstance(coefficients,list) or len(coefficients) != 4
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in coefficients)
            or coefficients[0] < 0 or any(abs(v)>2 for v in coefficients[1:])):
        raise ValueError('Invalid joint fitted coefficients')
    close(fit['beta'], coefficients[1:], 'fit beta aliases')
    m = Model(); m.fit = fit
    if prediction['contextual']:
        if not usable or not any(fit['support']) or active_features(fit,context) != fit['support']:
            raise ValueError('Joint trajectory lacks complete supported context')
        at = prediction['trajectory_anchor_time_seconds']
        if type(at) not in (int,float) or not math.isfinite(at) or at > now + 1e-8:
            raise ValueError('Joint trajectory anchor follows admission')
        anchor = prediction['trajectory_anchor_coefficients']
        if (not isinstance(anchor,list) or len(anchor)!=4 or
                any(type(v) not in (int,float) or not math.isfinite(v) for v in anchor)
                or anchor[0] < 0 or any(abs(v)>2 for v in anchor[1:])):
            raise ValueError('Invalid joint trajectory anchor')
        m.active = fit['support'].copy()
        m.trajectory = (anchor,coefficients,at)
    expected = m.prediction(context,scalar,now,usable,n)
    for name,value in expected.items():
        close(prediction.get(name),value,name)
    return expected
