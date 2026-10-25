//! Decision-focused shadow ablations. These predictions never affect admission.
//! Exact labels and right-censored lower bounds share a slope-regularized objective;
//! a lower bound is never treated as a measured TTFT or rounded to a deadline.
//!
//! X = uncalibrated live N (original N fallback), independent of legacy scalar.
//! The ideal single-layer expression is T = B + X*max(0, q+a·z); the deployed
//! expression is B + X*max(0, q+clamp(a·z, -0.5, 0.5)). q is bounded to [0,4]
//! and each feature coefficient to [-2,2], with the existing L1 trajectory cap.
//! Only contextual slopes receive ridge regularization. Penalizing q toward1
//! biases even an exact, noiseless queue-only fit away from its observed gain.
//! z contains pending growth over30s, oldest pending age/300, and timely native
//! progress deficit. The declared [0,1] domain permits bounded extrapolation;
//! diagnostics distinguish predictions outside the observed training ranges.
//! The convex fit uses an unclipped linear response. Prediction retains the
//! contextual ±0.5 cap; both objectives and clipping are exported explicitly.
//! This is not an exact optimizer of the capped, time-slewed prediction.
//! Fits run on poll only when labels/retention change. Fresh polls after a gap
//! may reuse age-valid rows, but restart the trajectory and growth history.
//! All history is shadow-only and independently reset with the worker epoch.
//! Unsupported/unavailable models retain the legacy scalar*X fallback. Its
//! exported coefficients are effective output weights, not learned weights;
//! the scalar can exceed the fitted q cap and fallback transitions are not
//! constrained by the learned trajectory. Learned vectors remain in diagnostics.
use std::collections::{HashMap, VecDeque};

use serde::Serialize;

pub(super) const METHOD: &str = "decision_live_queue_slope_ridge_hinge_v3";
const PRIOR: [f64; 4] = [1.0, 0.0, 0.0, 0.0];
const RIDGE: f64 = 0.1;
const MIN_X: f64 = 0.001;
const MAX_PENDING: usize = 8192;
const ARMS: [&str; 4] = ["exact_queue", "exact_early", "bound_queue", "bound_early"];

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(super) struct Context {
    pub values: [Option<f64>; 3],
    pub pending: usize,
    pub complete: bool,
    pub oldest_seconds: Option<f64>,
    pub growth_reference_time: Option<f64>,
    pub progress_tokens_per_second: Option<f64>,
    pub progress_capacity: Option<f64>,
}
#[derive(Clone, Copy, Debug, Serialize)]
pub(super) struct Snapshot {
    pub at: f64,
    pub profile: u64,
    pub baseline: f64,
    pub x: f64,
    pub source: &'static str,
    pub original_queue_extra: f64,
    pub live_queue_extra: Option<f64>,
    pub scalar_gain: f64,
    pub fresh: bool,
    pub native_ready: bool,
    pub native_waiting: f64,
    pub poll_started: f64,
    pub eligible: bool,
    pub context: Context,
}
#[derive(Clone, Debug, Serialize)]
pub(super) struct Admission {
    pub snapshot: Snapshot,
    pub predictions: [Prediction; 4],
}

#[derive(Clone, Debug)]
struct Pending {
    started: f64,
    snapshot: Snapshot,
    first_label: Option<f64>,
}
#[derive(Clone, Copy, Debug, Serialize)]
pub(super) struct Label {
    pub id: u64,
    pub at: f64,
    pub observed_at: f64,
    pub elapsed: f64,
    pub exact: bool,
    pub first_label_at: f64,
    pub snapshot: Snapshot,
}
#[derive(Clone, Debug, Serialize)]
pub(super) struct Fit {
    pub coefficients: [f64; 4],
    pub mask: u8,
    pub count: usize,
    pub exact_count: usize,
    pub bound_count: usize,
    pub ess: f64,
    pub minimum: [f64; 3],
    pub maximum: [f64; 3],
    pub supported: bool,
    pub objective: f64,
    pub projected_gradient: f64,
    pub bounded_training_objective: f64,
    pub training_clipped_count: usize,
}
impl Default for Fit {
    fn default() -> Self {
        Self {
            coefficients: PRIOR,
            mask: 0,
            count: 0,
            exact_count: 0,
            bound_count: 0,
            ess: 0.0,
            minimum: [0.0; 3],
            maximum: [0.0; 3],
            supported: false,
            objective: 0.0,
            projected_gradient: 0.0,
            bounded_training_objective: 0.0,
            training_clipped_count: 0,
        }
    }
}
fn vector(ctx: Context, mask: u8) -> Option<[f64; 4]> {
    let mut z = PRIOR;
    for j in 0..3 {
        if mask & (1 << j) != 0 {
            z[j + 1] = ctx.values[j].filter(|v| v.is_finite() && (0.0..=1.0).contains(v))?;
        }
    }
    Some(z)
}
fn dot(a: [f64; 4], b: [f64; 4]) -> f64 {
    (0..4).map(|j| a[j] * b[j]).sum()
}
fn project(mut theta: [f64; 4]) -> [f64; 4] {
    theta[0] = theta[0].clamp(0.0, 4.0);
    for v in &mut theta[1..] {
        *v = v.clamp(-2.0, 2.0);
    }
    theta
}
impl Fit {
    fn learn(rows: &VecDeque<Label>, early: bool, censor: bool) -> Self {
        let masks: &[u8] = if early {
            &[7, 6, 5, 3, 4, 2, 1, 0]
        } else {
            &[0]
        };
        for &mask in masks {
            let selected: Vec<_> = rows
                .iter()
                .filter(|r| r.exact || censor)
                .filter_map(|r| vector(r.snapshot.context, mask).map(|z| (r, z)))
                .collect();
            let count = selected.len();
            if count < 12 {
                continue;
            }
            let scale = selected
                .iter()
                .map(|(r, _)| r.snapshot.x)
                .fold(0.0, f64::max);
            let weights: Vec<_> = selected
                .iter()
                .map(|(r, _)| (r.snapshot.x / scale).powi(2))
                .collect();
            let sum: f64 = weights.iter().sum();
            let ess = sum * sum / weights.iter().map(|w| w * w).sum::<f64>();
            if !ess.is_finite() || ess < 8.0 {
                continue;
            }
            let mut minimum = [0.0; 3];
            let mut maximum = [0.0; 3];
            let mut support = true;
            for j in 0..3 {
                if mask & (1 << j) != 0 {
                    minimum[j] = selected
                        .iter()
                        .map(|(_, z)| z[j + 1])
                        .fold(f64::INFINITY, f64::min);
                    maximum[j] = selected
                        .iter()
                        .map(|(_, z)| z[j + 1])
                        .fold(f64::NEG_INFINITY, f64::max);
                    let mean = selected
                        .iter()
                        .zip(&weights)
                        .map(|((_, z), w)| w * z[j + 1] / sum)
                        .sum::<f64>();
                    let var = selected
                        .iter()
                        .zip(&weights)
                        .map(|((_, z), w)| w * (z[j + 1] - mean).powi(2) / sum)
                        .sum::<f64>();
                    support &= maximum[j] - minimum[j] >= 0.05 && var >= 1e-4;
                }
            }
            if !support {
                continue;
            }
            // RIDGE remains the largest diagonal penalty (on slopes), so this
            // conservative Lipschitz bound is safe with an unpenalized intercept.
            let step = 1.0
                / (RIDGE
                    + selected
                        .iter()
                        .zip(&weights)
                        .map(|((_, z), w)| w * dot(*z, *z) / sum)
                        .sum::<f64>());
            let gradient = |theta: [f64; 4]| {
                let mut grad = theta.map(|_| 0.0);
                for j in 1..4 {
                    grad[j] = RIDGE * (theta[j] - PRIOR[j]);
                }
                for ((r, z), w) in selected.iter().zip(&weights) {
                    let target = (r.elapsed - r.snapshot.baseline).max(0.0) / r.snapshot.x;
                    let residual = dot(theta, *z) - target;
                    let residual = if r.exact { residual } else { residual.min(0.0) };
                    for j in 0..4 {
                        grad[j] += w / sum * residual * z[j];
                    }
                }
                grad
            };
            let mut coefficients = PRIOR;
            for _ in 0..256 {
                let grad = gradient(coefficients);
                coefficients = project(std::array::from_fn(|j| coefficients[j] - step * grad[j]));
            }
            let grad = gradient(coefficients);
            let next = project(std::array::from_fn(|j| coefficients[j] - step * grad[j]));
            let projected_gradient = (0..4)
                .map(|j| ((coefficients[j] - next[j]) / step).abs())
                .fold(0.0, f64::max);
            let objective = 0.5
                * RIDGE
                * (1..4)
                    .map(|j| (coefficients[j] - PRIOR[j]).powi(2))
                    .sum::<f64>()
                + 0.5
                    * selected
                        .iter()
                        .zip(&weights)
                        .map(|((r, z), w)| {
                            let e = dot(coefficients, *z)
                                - (r.elapsed - r.snapshot.baseline).max(0.0) / r.snapshot.x;
                            w / sum * if r.exact { e * e } else { e.min(0.0).powi(2) }
                        })
                        .sum::<f64>();
            let bounded_training_objective = 0.5
                * selected
                    .iter()
                    .zip(&weights)
                    .map(|((r, z), w)| {
                        let shift = dot(coefficients, *z) - coefficients[0];
                        let y = coefficients[0] + shift.clamp(-0.5, 0.5).max(-coefficients[0]);
                        let e = y - (r.elapsed - r.snapshot.baseline).max(0.0) / r.snapshot.x;
                        w / sum * if r.exact { e * e } else { e.min(0.0).powi(2) }
                    })
                    .sum::<f64>()
                + 0.5
                    * RIDGE
                    * (1..4)
                        .map(|j| (coefficients[j] - PRIOR[j]).powi(2))
                        .sum::<f64>();
            let training_clipped_count = selected
                .iter()
                .filter(|(_, z)| {
                    let shift = dot(coefficients, *z) - coefficients[0];
                    shift != shift.clamp(-0.5, 0.5).max(-coefficients[0])
                })
                .count();
            return Self {
                coefficients,
                mask,
                count,
                exact_count: selected.iter().filter(|(r, _)| r.exact).count(),
                bound_count: selected.iter().filter(|(r, _)| !r.exact).count(),
                ess,
                minimum,
                maximum,
                supported: true,
                objective,
                projected_gradient,
                bounded_training_objective,
                training_clipped_count,
            };
        }
        Self::default()
    }
}
#[derive(Clone, Debug, Serialize)]
struct Model {
    fit: Fit,
    anchor: [f64; 4],
    target: [f64; 4],
    at: f64,
}
impl Default for Model {
    fn default() -> Self {
        Self {
            fit: Fit::default(),
            anchor: PRIOR,
            target: PRIOR,
            at: 0.0,
        }
    }
}
impl Model {
    fn applied(&self, now: f64) -> [f64; 4] {
        let distance = (0..4)
            .map(|j| (self.target[j] - self.anchor[j]).abs())
            .sum::<f64>();
        let fraction = if distance > 0.0 {
            (0.5 * (now - self.at).max(0.0) / distance).min(1.0)
        } else {
            1.0
        };
        std::array::from_fn(|j| self.anchor[j] + fraction * (self.target[j] - self.anchor[j]))
    }
    fn update(&mut self, fit: Fit, now: f64) {
        // Unsupported/subset changes cannot retain an intercept from a different basis.
        self.anchor = if self.fit.mask == fit.mask && self.fit.supported && fit.supported {
            self.applied(now)
        } else {
            PRIOR
        };
        self.target = fit.coefficients;
        self.at = now;
        self.fit = fit;
    }
}
#[derive(Clone, Debug, Serialize)]
pub(super) struct Prediction {
    pub arm: &'static str,
    pub source_arm: &'static str,
    pub scalar_fallback: bool,
    pub coefficients: [f64; 4],
    pub contributions: [f64; 4],
    pub extra: f64,
    pub total: f64,
    pub supported: bool,
    pub within_fitted_ranges: bool,
    pub available: bool,
    pub unbounded_extra: f64,
    pub contextual_clip_applied: bool,
}
#[derive(Debug, Default)]
pub(super) struct State {
    pending: HashMap<u64, Pending>,
    incomplete: bool,
    profile: Option<u64>,
    points: VecDeque<(f64, usize)>,
    last_poll: Option<f64>,
    rows: VecDeque<Label>,
    origin: Option<f64>,
    models: [Model; 4],
    revision: u64,
    fitted_revision: Option<u64>,
}
impl State {
    pub fn register(&mut self, id: Option<u64>, started: f64, snapshot: Option<Snapshot>) -> bool {
        let Some(id) = id else {
            self.incomplete = true;
            return false;
        };
        let Some(snapshot) = snapshot else {
            self.incomplete = true;
            return false;
        };
        if self.pending.len() >= MAX_PENDING {
            self.incomplete = true;
            return false;
        }
        self.pending.insert(
            id,
            Pending {
                started,
                snapshot,
                first_label: None,
            },
        );
        true
    }
    pub fn mark_incomplete(&mut self) {
        self.incomplete = true;
        self.points.clear();
    }
    pub fn context(
        &self,
        now: f64,
        progress: Option<f64>,
        capacity: Option<f64>,
        backlog: bool,
    ) -> Context {
        let pending = self.pending.len();
        let complete = !self.incomplete;
        let oldest = complete.then(|| {
            self.pending
                .values()
                .map(|p| (now - p.started).max(0.0))
                .fold(0.0, f64::max)
        });
        let reference = self
            .points
            .iter()
            .rev()
            .find(|(at, _)| now - at >= 30.0 && now - at <= 60.0)
            .filter(|_| complete);
        let growth = reference
            .map(|(_, count)| (pending as f64 - *count as f64).max(0.0) / (pending.max(1) as f64));
        let deficit = progress
            .zip(capacity)
            .filter(|(p, c)| p.is_finite() && *p >= 0.0 && c.is_finite() && *c > 0.0)
            .filter(|_| complete && pending > 0 && backlog)
            .map(|(p, c)| (1.0 - p / c).clamp(0.0, 1.0));
        Context {
            values: [growth, oldest.map(|a| (a / 300.0).min(1.0)), deficit],
            pending,
            complete,
            oldest_seconds: oldest,
            growth_reference_time: reference.map(|(at, _)| *at),
            progress_tokens_per_second: progress,
            progress_capacity: capacity,
        }
    }
    pub fn label(
        &mut self,
        id: u64,
        observed_at: f64,
        now: f64,
        exact: bool,
        terminal: bool,
        bound: Option<f64>,
    ) -> Option<Label> {
        let pending = if terminal {
            self.pending.remove(&id)
        } else {
            self.pending.get(&id).cloned()
        }?;
        let elapsed = bound.unwrap_or(observed_at - pending.started);
        let s = pending.snapshot;
        if !s.eligible
            || self.profile != Some(s.profile)
            || now - s.at >= 600.0
            || now < s.at
            || !elapsed.is_finite()
            || elapsed <= 0.0
            || s.x < MIN_X
        {
            return None;
        }
        self.expire(now);
        let first = pending.first_label.unwrap_or(now);
        let prior = self.rows.iter().position(|r| r.id == id);
        if pending.first_label.is_some() && prior.is_none() {
            return None;
        }
        if let Some(i) = prior {
            if self.rows[i].exact || (!exact && self.rows[i].elapsed >= elapsed) {
                return None;
            }
        }
        if self.origin.is_none() {
            self.origin = Some(now);
        }
        let label = Label {
            id,
            at: now,
            observed_at,
            elapsed,
            exact,
            first_label_at: first,
            snapshot: s,
        };
        if let Some(i) = prior {
            self.rows[i] = label;
        } else {
            self.rows.push_back(label);
        }
        self.revision = self.revision.wrapping_add(1);
        if let Some(p) = self.pending.get_mut(&id) {
            p.first_label = Some(first);
        }
        // Upgrades remain in their original bank and never rejuvenate retention.
        let origin = *self.origin.get_or_insert(now);
        let bank = ((first - origin) / 300.0).floor();
        while self
            .rows
            .iter()
            .filter(|r| ((r.first_label_at - origin) / 300.0).floor() == bank)
            .count()
            > 32
        {
            if let Some(index) = self
                .rows
                .iter()
                .position(|r| ((r.first_label_at - origin) / 300.0).floor() == bank)
            {
                self.rows.remove(index);
            } else {
                break;
            }
        }
        Some(label)
    }
    pub fn terminal_unknown(&mut self, id: Option<u64>) {
        if let Some(id) = id {
            self.pending.remove(&id);
        }
    }
    fn expire(&mut self, now: f64) {
        let Some(origin) = self.origin else { return };
        let bank = ((now - origin) / 300.0).floor();
        let before = self.rows.len();
        self.rows.retain(|r| {
            ((r.first_label_at - origin) / 300.0).floor() >= bank - 1.0
                && now - r.snapshot.at < 600.0
        });
        if before != self.rows.len() {
            self.revision = self.revision.wrapping_add(1);
        }
    }
    pub fn poll(&mut self, now: f64, profile: Option<u64>, fresh: bool, max_gap: f64) {
        if profile != self.profile {
            self.rows.clear();
            self.models = Default::default();
            self.origin = None;
            self.points.clear();
            self.profile = profile;
            self.revision = self.revision.wrapping_add(1);
            self.fitted_revision = None;
        }
        self.expire(now);
        if !fresh || self.last_poll.is_some_and(|at| now - at > max_gap) {
            self.points.clear();
            self.models = Default::default();
            self.fitted_revision = None;
        }
        self.last_poll = Some(now);
        if !fresh {
            return;
        }
        if !self.incomplete {
            self.points.push_back((now, self.pending.len()));
        }
        while self.points.front().is_some_and(|(at, _)| now - at > 60.0) || self.points.len() > 256
        {
            self.points.pop_front();
        }
        if self.fitted_revision != Some(self.revision) {
            for i in 0..4 {
                self.models[i].update(Fit::learn(&self.rows, i % 2 == 1, i >= 2), now);
            }
            self.fitted_revision = Some(self.revision);
        }
    }
    pub fn predictions(&self, snapshot: Snapshot, now: f64, available: bool) -> [Prediction; 4] {
        std::array::from_fn(|i| {
            let requested = &self.models[i];
            let index = if i % 2 == 1
                && (!requested.fit.supported
                    || vector(snapshot.context, requested.fit.mask).is_none())
            {
                i - 1
            } else {
                i
            };
            let model = &self.models[index];
            if !available || !model.fit.supported {
                let unbounded_extra = snapshot.x * snapshot.scalar_gain;
                let extra = unbounded_extra.max(0.0);
                return Prediction {
                    arm: ARMS[i],
                    source_arm: "live_n_scalar",
                    scalar_fallback: true,
                    coefficients: [snapshot.scalar_gain, 0.0, 0.0, 0.0],
                    contributions: [extra, 0.0, 0.0, 0.0],
                    extra,
                    total: snapshot.baseline + extra,
                    supported: false,
                    within_fitted_ranges: true,
                    available: available && unbounded_extra.is_finite(),
                    unbounded_extra,
                    contextual_clip_applied: false,
                };
            }
            let z = vector(snapshot.context, model.fit.mask).unwrap_or(PRIOR);
            let coefficients = model.applied(now);
            let mut contributions = std::array::from_fn(|j| snapshot.x * coefficients[j] * z[j]);
            let shift = dot(coefficients, z) - coefficients[0];
            let bounded = shift.clamp(-0.5, 0.5).max(-coefficients[0]);
            if shift != 0.0 {
                for c in &mut contributions[1..] {
                    *c *= bounded / shift;
                }
            }
            let extra = contributions.iter().sum::<f64>().max(0.0);
            let within = (0..3).all(|j| {
                model.fit.mask & (1 << j) == 0
                    || (z[j + 1] >= model.fit.minimum[j] && z[j + 1] <= model.fit.maximum[j])
            });
            Prediction {
                arm: ARMS[i],
                source_arm: ARMS[index],
                scalar_fallback: false,
                coefficients,
                contributions,
                extra,
                total: snapshot.baseline + extra,
                supported: model.fit.supported,
                within_fitted_ranges: within,
                available: available && extra.is_finite(),
                unbounded_extra: snapshot.x * dot(coefficients, z),
                contextual_clip_applied: shift != bounded,
            }
        })
    }
    pub fn diagnostics(&self) -> serde_json::Value {
        serde_json::json!({"models":self.models,"rows":self.rows.len(),"origin":self.origin,"profile":self.profile,
            "revision":self.revision,"fitted_revision":self.fitted_revision,"pending":self.pending.len(),"complete":!self.incomplete,"ridge":RIDGE,"min_x_seconds":MIN_X,
            "coefficient_ridge":[0.0,RIDGE,RIDGE,RIDGE],
            "iterations":256,"coefficient_l1_per_second":0.5,"contextual_gain_cap":0.5,"domain":[0.0,1.0]})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn snapshot(at: f64, values: [Option<f64>; 3]) -> Snapshot {
        Snapshot {
            at,
            profile: 7,
            baseline: 2.0,
            x: 100.0,
            source: "live_n",
            original_queue_extra: 100.0,
            live_queue_extra: Some(100.0),
            scalar_gain: 1.0,
            fresh: true,
            native_ready: true,
            native_waiting: 1.0,
            poll_started: at,
            eligible: true,
            context: Context {
                values,
                complete: true,
                ..Default::default()
            },
        }
    }
    fn state() -> State {
        let mut s = State::default();
        s.poll(0.0, Some(7), true, 5.0);
        s
    }
    fn labels() -> VecDeque<Label> {
        (0..32)
            .map(|i| {
                let g = (i % 4) as f64 / 3.0;
                let a = (i / 4) as f64 / 7.0;
                Label {
                    id: i,
                    at: 20.0,
                    observed_at: 20.0,
                    elapsed: 2.0 + 100.0 * (1.5 + 0.8 * g - 0.4 * a),
                    exact: i % 3 != 0,
                    first_label_at: 20.0,
                    snapshot: snapshot(0.0, [Some(g), Some(a), None]),
                }
            })
            .collect()
    }
    #[test]
    fn joint_exact_and_hinge_fixture() {
        let rows = labels();
        let fit = Fit::learn(&rows, true, true);
        assert!(fit.supported);
        assert_eq!(fit.mask, 3);
        assert!(fit.coefficients[1] > 0.0);
        assert!(fit.coefficients[2] < 0.0);
        assert!(fit.projected_gradient < 1e-7);
        println!(
            "DECISION_FIXTURE={}",
            serde_json::json!({"rows":rows,"fit":fit})
        );
    }
    #[test]
    fn neutral_prior_and_one_sided_bound() {
        let mut rows = labels();
        for r in &mut rows {
            r.exact = false;
            r.elapsed = 52.0;
        }
        let fit = Fit::learn(&rows, false, true);
        assert_eq!(fit.coefficients, PRIOR);
        assert!(!Fit::learn(&rows, false, false).supported);
        for r in &mut rows {
            r.elapsed = 202.0;
        }
        let fit = Fit::learn(&rows, false, true);
        assert!(fit.coefficients[0] > 1.8);
        assert_eq!(Fit::learn(&VecDeque::new(), true, true).coefficients, PRIOR);
    }
    #[test]
    fn noiseless_queue_gain_has_no_shrinkage_above_or_below_one() {
        let mut fixtures = Vec::new();
        for gain in [0.5, 2.0] {
            let mut rows = labels();
            for row in &mut rows {
                row.exact = true;
                row.elapsed = row.snapshot.baseline + gain * row.snapshot.x;
            }
            let fit = Fit::learn(&rows, false, false);
            assert!(fit.supported);
            assert!((fit.coefficients[0] - gain).abs() < 1e-12);
            assert_eq!(&fit.coefficients[1..], &[0.0; 3]);
            assert!(fit.objective < 1e-20);
            assert!(fit.projected_gradient < 1e-12);
            fixtures.push(serde_json::json!({"name":"exact_queue","gain":gain,"rows":rows,"early":false,"censored":false,"fit":fit}));
        }
        println!(
            "UNPENALIZED_QUEUE_FIXTURE={}",
            serde_json::json!({"cases":fixtures})
        );
    }
    #[test]
    fn bound_only_queue_fit_respects_one_sided_identifiability_and_cap() {
        let mut fixtures = Vec::new();
        for (bound, expected_gain) in [(0.5, 1.0), (2.0, 2.0), (5.0, 4.0)] {
            let mut rows = labels();
            for row in &mut rows {
                row.exact = false;
                row.elapsed = row.snapshot.baseline + bound * row.snapshot.x;
            }
            let fit = Fit::learn(&rows, false, true);
            assert!(fit.supported);
            assert_eq!(fit.exact_count, 0);
            assert_eq!(fit.bound_count, rows.len());
            assert!((fit.coefficients[0] - expected_gain).abs() < 1e-12);
            assert!(fit.projected_gradient < 1e-12);
            assert!(!Fit::learn(&rows, false, false).supported);
            if bound < 4.0 {
                assert!(fit.objective < 1e-20);
            } else {
                assert!((fit.objective - 0.5).abs() < 1e-12);
            }
            // A lower bound below the prior does not identify a smaller point
            // prediction. Feasible q>=bound has zero loss, so initialization1 stays.
            fixtures.push(serde_json::json!({"name":"bound_queue","bound":bound,"rows":rows,"early":false,"censored":true,"fit":fit}));
            rows.truncate(11);
            assert!(!Fit::learn(&rows, false, true).supported);
        }
        println!(
            "BOUND_QUEUE_FIXTURE={}",
            serde_json::json!({"cases":fixtures})
        );
    }
    #[test]
    fn contextual_slopes_still_have_ridge_and_intercept_has_zero_mean_residual() {
        for slope in [-0.3, 0.3] {
            let mut rows = labels();
            for (i, row) in rows.iter_mut().enumerate() {
                let age = i as f64 / 31.0;
                row.exact = true;
                row.snapshot.context.values = [None, Some(age), None];
                row.elapsed = row.snapshot.baseline + row.snapshot.x * (2.0 + slope * age);
            }
            let fit = Fit::learn(&rows, true, false);
            let variance = (0..32)
                .map(|i| (i as f64 / 31.0 - 0.5).powi(2))
                .sum::<f64>()
                / 32.0;
            let expected_slope = slope * variance / (variance + RIDGE);
            let expected_intercept = 2.0 + (slope - expected_slope) * 0.5;
            assert_eq!(fit.mask, 2);
            assert!((fit.coefficients[2] - expected_slope).abs() < 1e-10);
            assert!((fit.coefficients[0] - expected_intercept).abs() < 1e-10);
            assert!(fit.coefficients[2].abs() < slope.abs());
            assert_eq!(fit.training_clipped_count, 0);
            assert!((fit.objective - fit.bounded_training_objective).abs() < 1e-12);
        }
    }
    #[test]
    fn single_layer_fit_and_supported_predictions_ignore_legacy_scalar() {
        let mut a = state();
        a.rows = labels();
        let mut b = state();
        b.rows = a.rows.clone();
        for (i, row) in b.rows.iter_mut().enumerate() {
            row.snapshot.scalar_gain = if i % 2 == 0 { 0.2 } else { 6.0 };
        }
        for s in [&mut a, &mut b] {
            s.revision += 1;
            s.poll(1.0, Some(7), true, 5.0);
        }
        assert_eq!(a.diagnostics(), b.diagnostics());
        let mut low = snapshot(1.0, [Some(0.5), Some(0.5), None]);
        low.scalar_gain = 0.2;
        let mut high = low;
        high.scalar_gain = 6.0;
        let before = a.predictions(low, 4.0, true);
        let after = b.predictions(high, 4.0, true);
        assert!(before.iter().all(|p| p.supported && !p.scalar_fallback));
        assert_eq!(
            serde_json::to_value(&before).unwrap(),
            serde_json::to_value(&after).unwrap()
        );
        for p in before {
            let shift = p.coefficients[1] * 0.5 + p.coefficients[2] * 0.5;
            let expected = low.x * (p.coefficients[0] + shift.clamp(-0.5, 0.5)).max(0.0);
            assert!((p.extra - expected).abs() < 1e-10);
        }
    }
    #[test]
    fn unsupported_and_unavailable_outputs_explicitly_preserve_scalar_fallback() {
        let mut s = state();
        let mut snap = snapshot(1.0, [Some(0.5), Some(0.5), None]);
        // Legacy fallback may exceed the learned intercept cap; it is not fitted.
        snap.scalar_gain = 6.0;
        for p in s.predictions(snap, 1.0, true) {
            assert!(p.scalar_fallback && p.available && !p.supported);
            assert_eq!(p.source_arm, "live_n_scalar");
            assert_eq!(p.coefficients, [6.0, 0.0, 0.0, 0.0]);
            assert_eq!(p.contributions, [600.0, 0.0, 0.0, 0.0]);
            assert_eq!(p.total, 602.0);
        }
        s.rows = labels();
        s.origin = Some(20.0);
        s.revision += 1;
        s.poll(21.0, Some(7), true, 30.0);
        assert!(s
            .predictions(snap, 24.0, true)
            .iter()
            .all(|p| !p.scalar_fallback));
        for p in s.predictions(snap, 24.0, false) {
            assert!(p.scalar_fallback && !p.available && !p.supported);
            assert_eq!(p.extra, 600.0);
            assert_eq!(p.extra, p.contributions.iter().sum::<f64>());
        }
        s.poll(600.0, Some(7), true, 5.0);
        assert!(s
            .predictions(snap, 600.0, true)
            .iter()
            .all(|p| p.scalar_fallback));
    }
    #[test]
    fn lower_bound_upgrade_preserves_age_position_and_weight() {
        let mut s = state();
        s.register(Some(1), 0.0, Some(snapshot(0.0, [None; 3])));
        let bound = s.label(1, 300.0, 300.0, false, false, Some(300.0)).unwrap();
        assert_eq!(s.pending.len(), 1);
        assert!(!bound.exact);
        let exact = s.label(1, 350.0, 350.0, true, true, None).unwrap();
        assert_eq!(s.rows.len(), 1);
        assert_eq!(exact.first_label_at, 300.0);
        assert_eq!(exact.snapshot.at, 0.0);
        assert!(s.pending.is_empty());
        assert_eq!(s.rows[0].elapsed, 350.0);
        s.expire(600.0);
        assert!(s.rows.is_empty());
    }
    #[test]
    fn near_deadline_is_not_rounded_or_exact() {
        let mut s = state();
        s.register(Some(1), 0.0, Some(snapshot(0.0, [None; 3])));
        let l = s.label(1, 299.3, 299.4, false, true, None).unwrap();
        assert_eq!(l.elapsed, 299.3);
        assert!(!l.exact);
        assert_eq!(l.at, 299.4);
        assert!(s.pending.is_empty());
    }
    #[test]
    fn unknown_and_ineligible_terminal_still_remove_pending() {
        let mut s = state();
        let mut snap = snapshot(0.0, [None; 3]);
        snap.eligible = false;
        s.register(Some(1), 0.0, Some(snap));
        assert!(s.label(1, 5.0, 5.0, false, true, None).is_none());
        s.register(Some(2), 0.0, Some(snapshot(0.0, [None; 3])));
        s.terminal_unknown(Some(2));
        assert!(s.pending.is_empty());
        assert!(s.rows.is_empty());
    }
    #[test]
    fn evicted_bound_cannot_return_as_fresh_exact() {
        let mut s = state();
        for id in 0..33 {
            s.register(Some(id), 0.0, Some(snapshot(0.0, [None; 3])));
            s.label(id, 300.0, 300.0, false, false, Some(300.0));
        }
        assert_eq!(s.rows.len(), 32);
        assert!(s.label(0, 310.0, 310.0, true, true, None).is_none());
        assert_eq!(s.rows.len(), 32);
    }
    #[test]
    fn profile_and_stale_admissions_cannot_train() {
        let mut s = state();
        s.register(Some(1), 0.0, Some(snapshot(0.0, [None; 3])));
        s.poll(2.0, Some(8), true, 5.0);
        assert!(s.label(1, 3.0, 3.0, true, true, None).is_none());
        s.poll(3.0, Some(7), true, 5.0);
        s.register(Some(2), 0.0, Some(snapshot(0.0, [None; 3])));
        assert!(s.label(2, 600.0, 600.0, true, true, None).is_none());
        assert!(s.pending.is_empty());
        assert!(s.rows.is_empty());
    }
    #[test]
    fn growth_window_uses_complete_continuous_polls_and_retains_breaches() {
        let mut s = state();
        s.register(Some(1), 0.0, Some(snapshot(0.0, [None; 3])));
        for at in 1..=30 {
            s.poll(at as f64, Some(7), true, 5.0);
        }
        let c = s.context(30.0, Some(0.0), Some(100.0), true);
        assert_eq!(c.values, [Some(1.0), Some(0.1), Some(1.0)]);
        s.label(1, 300.0, 300.0, false, false, Some(300.0));
        assert_eq!(
            s.context(301.0, None, Some(100.0), true).oldest_seconds,
            Some(301.0)
        );
        s.poll(301.0, Some(7), true, 5.0);
        assert_eq!(s.context(301.0, None, None, true).values[0], None);
    }
    #[test]
    fn progress_unknown_is_not_stall_and_overflow_is_unknown() {
        let mut s = state();
        s.register(Some(1), 0.0, Some(snapshot(0.0, [None; 3])));
        assert_eq!(s.context(1.0, None, Some(100.0), true).values[2], None);
        assert_eq!(
            s.context(1.0, Some(0.0), Some(100.0), false).values[2],
            None
        );
        for id in 2..=MAX_PENDING as u64 + 1 {
            s.register(Some(id), 0.0, Some(snapshot(0.0, [None; 3])));
        }
        let c = s.context(1.0, Some(0.0), Some(100.0), true);
        assert!(!c.complete);
        assert_eq!(c.values, [None; 3]);
    }
    #[test]
    fn movement_bound_decomposition_and_native_input_responsiveness() {
        let mut s = state();
        s.rows = labels();
        s.revision += 1;
        s.poll(1.0, Some(7), true, 5.0);
        let snap = snapshot(1.0, [Some(0.5), Some(0.5), None]);
        let a = s.predictions(snap, 1.0, true)[3].clone();
        let b = s.predictions(snap, 1.4, true)[3].clone();
        let distance = (0..4)
            .map(|j| (b.coefficients[j] - a.coefficients[j]).abs())
            .sum::<f64>();
        assert!(distance <= 0.200000001);
        assert!((b.extra - b.contributions.iter().sum::<f64>()).abs() < 1e-10);
        let mut next = snap;
        next.x = 200.0;
        let c = s.predictions(next, 1.4, true)[3].clone();
        assert!((c.extra - 2.0 * b.extra).abs() < 1e-10);
    }
    #[test]
    fn missing_feature_falls_back_whole_queue_model_and_range_is_reported() {
        let mut s = state();
        s.rows = labels();
        s.revision += 1;
        s.poll(1.0, Some(7), true, 5.0);
        let pred = s.predictions(snapshot(1.0, [None, Some(0.5), None]), 10.0, true);
        assert_eq!(pred[3].source_arm, "bound_queue");
        assert_eq!(pred[3].coefficients, pred[2].coefficients);
        assert!(!pred[3].scalar_fallback);
        // Declared [0,1] domain permits bounded extrapolation, with explicit evidence.
        for r in &mut s.rows {
            r.snapshot.context.values[0] = r.snapshot.context.values[0].map(|x| x * 0.5);
        }
        s.revision += 1;
        s.poll(2.0, Some(7), true, 5.0);
        let pred = s.predictions(snapshot(2.0, [Some(1.0), Some(0.5), None]), 10.0, true);
        assert!(!pred[3].within_fitted_ranges);
        assert!(pred[3].available);
    }
    #[test]
    fn no_feedback_poll_does_not_refit_or_restart_movement() {
        let mut s = state();
        s.rows = labels();
        s.revision += 1;
        s.poll(1.0, Some(7), true, 5.0);
        let before = s.models[3].at;
        s.poll(2.0, Some(7), true, 5.0);
        assert_eq!(s.models[3].at, before);
        s.poll(3.0, Some(7), false, 5.0);
        assert_eq!(s.models[3].target, PRIOR);
    }
    #[test]
    fn deterministic_state_fixture() {
        let mut s = State::default();
        let mut actions = Vec::new();
        s.poll(0.0, Some(7), true, 5.0);
        actions.push(serde_json::json!({"op":"poll","now":0.0,"profile":7,"fresh":true,"max_gap":5.0,"state":s.diagnostics()}));
        let mut outcomes = Vec::new();
        for i in 0..32 {
            let g = (i % 4) as f64 / 3.0;
            let a = (i / 4) as f64 / 7.0;
            let mut snap = snapshot(0.0, [Some(g * 0.5), Some(a), None]);
            snap.scalar_gain = if i % 2 == 0 { 0.2 } else { 6.0 };
            s.register(Some(i), 0.0, Some(snap));
            actions.push(serde_json::json!({"op":"register","id":i,"started":0.0,"snapshot":snap}));
            outcomes.push((102.0 + 100.0 * (0.8 * g - 0.4 * a), i));
        }
        outcomes.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (at, id) in outcomes {
            let exact = id % 3 != 0;
            let label = s.label(id, at, at, exact, exact, None);
            actions.push(serde_json::json!({"op":"label","id":id,"observed_at":at,"now":at,"exact":exact,"terminal":exact,"bound":null,"label":label}));
        }
        for now in [200.0, 200.4, 230.0] {
            if now != 200.4 {
                s.poll(now, Some(7), true, 40.0);
            }
            let mut snap = snapshot(now, [Some(1.0), Some(0.5), None]);
            snap.scalar_gain = 2.0;
            snap.context = s.context(now, Some(0.0), Some(100.0), true);
            actions.push(serde_json::json!({"op":if now==200.4{"predict"}else{"poll"},"now":now,"profile":7,"fresh":true,"max_gap":40.0,
                "snapshot":snap,"predictions":s.predictions(snap,now,true),"state":s.diagnostics()}));
        }
        let label = s.label(0, 240.0, 240.0, true, true, None);
        actions.push(serde_json::json!({"op":"label","id":0,"observed_at":240.0,"now":240.0,"exact":true,"terminal":true,"bound":null,"label":label}));
        for (now, profile, fresh) in [
            (241.0, 7, true),
            (250.0, 7, false),
            (251.0, 7, true),
            (599.0, 7, true),
            (600.0, 7, true),
            (601.0, 8, true),
        ] {
            s.poll(now, Some(profile), fresh, 40.0);
            let mut snap = snapshot(now, [Some(1.0), Some(0.5), None]);
            snap.scalar_gain = 6.0;
            actions.push(serde_json::json!({"op":"poll","now":now,"profile":profile,"fresh":fresh,"max_gap":40.0,"snapshot":snap,
                "predictions":s.predictions(snap,now,fresh),"state":s.diagnostics()}));
        }
        println!(
            "DECISION_STATE_FIXTURE={}",
            serde_json::json!({"actions":actions})
        );
    }
}
