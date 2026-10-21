//! Experimental context-conditioned queue gain. No field here affects admission.
//! Rates describe resolved local observations, not native cancellation acknowledgements.
use super::ScaleTrajectory;
use serde::Serialize;
use std::{collections::VecDeque, time::Instant};

pub(super) const METHOD: &str = "feature_context_ridge_v1";
pub(super) const RIDGE: f64 = 0.1;
const MAX_SHIFT: f64 = 0.5;
const MAX_BETA: f64 = 2.0;
const MIN_SAMPLES: usize = 12;

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(super) struct Context {
    pub values: [Option<f64>; 3],
    pub counts: Counts,
    pub bucket_second: u64,
    pub activity_origin_time_seconds: f64,
}
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(super) struct Counts {
    pub first_text: u64,
    pub upstream_timeout: u64,
    pub pretext_downstream_disconnect: u64,
    pub unknown_pretext_end: u64,
    pub admitted: u64,
    pub actual_rejected: u64,
    pub untracked_admitted: u64,
}
#[derive(Clone, Copy, Debug)]
pub(super) enum Outcome {
    FirstText,
    Timeout,
    Disconnect,
    Unknown,
    Admitted,
    Rejected,
    UntrackedAdmitted,
}
impl Outcome {
    pub fn name(self) -> &'static str {
        match self {
            Self::FirstText => "first_text",
            Self::Timeout => "upstream_timeout",
            Self::Disconnect => "pretext_downstream_disconnect",
            Self::Unknown => "unknown_pretext_end",
            Self::Admitted | Self::UntrackedAdmitted => "admitted",
            Self::Rejected => "actual_rejected",
        }
    }
}
impl Counts {
    fn add(&mut self, other: Self) {
        self.first_text = self.first_text.saturating_add(other.first_text);
        self.upstream_timeout = self.upstream_timeout.saturating_add(other.upstream_timeout);
        self.pretext_downstream_disconnect = self
            .pretext_downstream_disconnect
            .saturating_add(other.pretext_downstream_disconnect);
        self.unknown_pretext_end = self
            .unknown_pretext_end
            .saturating_add(other.unknown_pretext_end);
        self.admitted = self.admitted.saturating_add(other.admitted);
        self.actual_rejected = self.actual_rejected.saturating_add(other.actual_rejected);
        self.untracked_admitted = self
            .untracked_admitted
            .saturating_add(other.untracked_admitted);
    }
    fn record(&mut self, outcome: Outcome) {
        if matches!(outcome, Outcome::UntrackedAdmitted) {
            self.untracked_admitted = self.untracked_admitted.saturating_add(1);
        }
        let value = match outcome {
            Outcome::FirstText => &mut self.first_text,
            Outcome::Timeout => &mut self.upstream_timeout,
            Outcome::Disconnect => &mut self.pretext_downstream_disconnect,
            Outcome::Unknown => &mut self.unknown_pretext_end,
            Outcome::Admitted | Outcome::UntrackedAdmitted => &mut self.admitted,
            Outcome::Rejected => &mut self.actual_rejected,
        };
        *value = value.saturating_add(1);
    }
}
#[derive(Debug)]
pub(super) struct Activity {
    origin: Instant,
    buckets: VecDeque<(u64, Counts)>,
}
impl Activity {
    pub fn new(now: Instant) -> Self {
        Self {
            origin: now,
            buckets: VecDeque::new(),
        }
    }
    pub fn record(&mut self, now: Instant, outcome: Outcome) {
        let second = now.saturating_duration_since(self.origin).as_secs();
        // Callbacks are serialized by the ledger lock. Never insert an older bucket.
        if self.buckets.back().is_some_and(|(time, _)| *time > second) {
            return;
        }
        while self
            .buckets
            .front()
            .is_some_and(|(time, _)| time.saturating_add(60) <= second)
        {
            self.buckets.pop_front();
        }
        if self.buckets.back().is_none_or(|(time, _)| *time != second) {
            self.buckets.push_back((second, Counts::default()));
        }
        self.buckets.back_mut().unwrap().1.record(outcome);
    }
    pub fn context(&self, now: Instant, kv: Option<f64>) -> Context {
        let second = now.saturating_duration_since(self.origin).as_secs();
        let mut counts = Counts::default();
        for (time, bucket) in &self.buckets {
            if *time <= second && time.saturating_add(60) > second {
                counts.add(*bucket);
            }
        }
        let cutoff = counts
            .upstream_timeout
            .saturating_add(counts.pretext_downstream_disconnect);
        let known = counts.first_text.saturating_add(cutoff);
        let resolved = known.saturating_add(counts.unknown_pretext_end);
        let attempts = counts.admitted.saturating_add(counts.actual_rejected);
        Context {
            bucket_second: second,
            activity_origin_time_seconds: super::diagnostic_time(self.origin),
            values: [
                kv.filter(|v| v.is_finite() && (0.0..=1.0).contains(v)),
                (known >= 16
                    && counts.untracked_admitted == 0
                    && (counts.unknown_pretext_end as f64) < 0.1 * resolved as f64)
                    .then(|| cutoff as f64 / known as f64),
                (attempts >= 16).then(|| counts.actual_rejected as f64 / attempts as f64),
            ],
            counts,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub(super) struct Fit {
    pub beta: [f64; 3],
    pub mean: [f64; 3],
    pub minimum: [f64; 3],
    pub maximum: [f64; 3],
    pub support: [bool; 3],
    pub count: usize,
    pub ess: f64,
    pub revision: u64,
}
impl Fit {
    /// At most seven tiny common-row fits. Missing inputs are never imputed.
    pub fn learn(
        samples: &VecDeque<(f64, f64)>,
        contexts: &VecDeque<Option<Context>>,
        revision: u64,
        scalar_target: f64,
    ) -> Self {
        let empty = Self {
            revision,
            ..Self::default()
        };
        if samples.len() != contexts.len() {
            return empty;
        }
        // Largest supported subset first; deterministic KV, cutoff, rejection priority.
        for mask in [7u8, 3, 5, 6, 1, 2, 4] {
            let rows: Vec<_> = samples
                .iter()
                .zip(contexts)
                .filter_map(|(&(ratio, n), ctx)| {
                    let ctx = ctx.as_ref()?;
                    (ratio.is_finite()
                        && n.is_finite()
                        && n > 0.0
                        && (0..3).all(|j| {
                            mask & (1 << j) == 0
                                || ctx.values[j]
                                    .is_some_and(|x| x.is_finite() && (0.0..=1.0).contains(&x))
                        }))
                    .then_some((ratio, n, ctx))
                })
                .collect();
            if rows.len() < MIN_SAMPLES {
                continue;
            }
            let largest = rows.iter().map(|r| r.1).fold(0.0, f64::max);
            let sumw: f64 = rows.iter().map(|r| (r.1 / largest).powi(2)).sum();
            let mut fit = Self {
                count: rows.len(),
                revision,
                minimum: [f64::INFINITY; 3],
                maximum: [f64::NEG_INFINITY; 3],
                ..Self::default()
            };
            let mut sumw2 = 0.0;
            for &(_, n, ctx) in &rows {
                let w = (n / largest).powi(2) / sumw;
                sumw2 += w * w;
                for j in 0..3 {
                    if mask & (1 << j) != 0 {
                        let x = ctx.values[j].unwrap();
                        fit.mean[j] += w * x;
                        fit.minimum[j] = fit.minimum[j].min(x);
                        fit.maximum[j] = fit.maximum[j].max(x);
                    }
                }
            }
            fit.ess = 1.0 / sumw2;
            if fit.ess < 8.0 {
                continue;
            }
            let mut covariance = [[0.0; 3]; 3];
            let mut rhs = [0.0; 3];
            for &(ratio, n, ctx) in &rows {
                let w = (n / largest).powi(2) / sumw;
                let z: [f64; 3] = std::array::from_fn(|j| {
                    if mask & (1 << j) != 0 {
                        ctx.values[j].unwrap() - fit.mean[j]
                    } else {
                        0.0
                    }
                });
                for j in 0..3 {
                    rhs[j] += w * z[j] * (ratio - scalar_target);
                    for k in 0..3 {
                        covariance[j][k] += w * z[j] * z[k];
                    }
                }
            }
            if (0..3).any(|j| {
                mask & (1 << j) != 0
                    && (covariance[j][j] < 1e-4 || fit.maximum[j] - fit.minimum[j] < 0.05)
            }) {
                continue;
            }
            for j in 0..3 {
                fit.support[j] = mask & (1 << j) != 0;
                covariance[j][j] += RIDGE;
                if !fit.support[j] {
                    fit.minimum[j] = 0.0;
                    fit.maximum[j] = 0.0;
                }
            }
            if let Some(beta) = solve(covariance, rhs) {
                // Neutral frozen prior; coefficient clipping is a bounded approximation
                // to unconstrained ridge, not a constrained-ridge optimizer.
                fit.beta = beta.map(|x| x.clamp(-MAX_BETA, MAX_BETA));
                return fit;
            }
        }
        empty
    }
    fn correction(&self, context: &Context) -> (f64, [bool; 3]) {
        let active = std::array::from_fn(|j| {
            self.support[j]
                && context.values[j]
                    .is_some_and(|x| x.is_finite() && x >= self.minimum[j] && x <= self.maximum[j])
        });
        let shift = (0..3)
            .filter(|j| active[*j])
            .map(|j| self.beta[j] * (context.values[j].unwrap() - self.mean[j]))
            .sum::<f64>()
            .clamp(-MAX_SHIFT, MAX_SHIFT);
        (shift, active)
    }
}
fn solve(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let mut l = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let v = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            l[i][j] = if i == j {
                if !v.is_finite() || v <= 0.0 {
                    return None;
                }
                v.sqrt()
            } else {
                v / l[j][j]
            };
        }
    }
    let mut y = [0.0; 3];
    for i in 0..3 {
        y[i] = (b[i] - (0..i).map(|j| l[i][j] * y[j]).sum::<f64>()) / l[i][i];
    }
    let mut x = [0.0; 3];
    for i in (0..3).rev() {
        x[i] = (y[i] - ((i + 1)..3).map(|j| l[j][i] * x[j]).sum::<f64>()) / l[i][i];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}
#[derive(Debug, Default)]
pub(super) struct Model {
    pub fit: Fit,
    trajectory: Option<ScaleTrajectory>,
    active: [bool; 3],
}
#[derive(Serialize)]
pub(super) struct Prediction {
    pub gain: f64,
    pub target_gain: f64,
    pub contextual: bool,
    pub reason: &'static str,
    pub active: [bool; 3],
    pub trajectory_anchor_gain: Option<f64>,
    pub trajectory_anchor_time_seconds: Option<f64>,
}
impl Model {
    pub fn invalidate(&mut self) {
        self.trajectory = None;
        self.active = [false; 3];
    }
    pub fn update(
        &mut self,
        context: &Context,
        scalar_target: f64,
        scalar_applied: f64,
        now: Instant,
    ) {
        let (shift, active) = self.fit.correction(context);
        self.active = active;
        if !active.iter().any(|x| *x) {
            self.trajectory = None;
            return;
        }
        let target = (scalar_target + shift).max(0.0);
        let trajectory = self.trajectory.get_or_insert(ScaleTrajectory {
            anchor_scale: scalar_applied,
            target_scale: scalar_applied,
            anchor_time: now,
        });
        trajectory.retarget(target, now);
    }
    pub fn prediction(
        &self,
        context: &Context,
        scalar: f64,
        now: Instant,
        usable: bool,
    ) -> Prediction {
        let (_, active) = self.fit.correction(context);
        // If current support changes between polls, avoid applying stale contributions.
        let trajectory = self
            .trajectory
            .as_ref()
            .filter(|_| usable && active == self.active && active.iter().any(|x| *x));
        Prediction {
            gain: trajectory.map_or(scalar, |t| t.applied(now)),
            target_gain: trajectory.map_or(scalar, |t| t.target_scale),
            contextual: trajectory.is_some(),
            reason: if !usable {
                "snapshot_unavailable"
            } else if trajectory.is_some() {
                "context_supported"
            } else {
                "legacy_fallback_no_current_support"
            },
            active,
            trajectory_anchor_gain: trajectory.map(|t| t.anchor_scale),
            trajectory_anchor_time_seconds: trajectory
                .map(|t| super::diagnostic_time(t.anchor_time)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    fn data(sign: f64) -> (VecDeque<(f64, f64)>, VecDeque<Option<Context>>) {
        (0..32)
            .map(|i| {
                let x = i as f64 / 31.0;
                (
                    (2.0 + sign * x, 100.0),
                    Some(Context {
                        values: [Some(x), None, None],
                        ..Context::default()
                    }),
                )
            })
            .unzip()
    }
    #[test]
    fn joint_ridge_matches_orthogonal_closed_form_and_collinear_remains_finite() {
        let (samples, contexts): (VecDeque<_>, VecDeque<_>) = (0..32)
            .map(|i| {
                let z = [
                    u8::from(i & 1 != 0) as f64,
                    u8::from(i & 2 != 0) as f64,
                    u8::from(i & 4 != 0) as f64,
                ];
                (
                    (3.0 + z[0] - z[1] + 0.5 * z[2], 100.0),
                    Some(Context {
                        values: z.map(Some),
                        ..Context::default()
                    }),
                )
            })
            .unzip();
        let f = Fit::learn(&samples, &contexts, 1, 3.25);
        assert_eq!(f.support, [true; 3]);
        for (actual, expected) in f
            .beta
            .into_iter()
            .zip([0.25 / 0.35, -0.25 / 0.35, 0.125 / 0.35])
        {
            assert!((actual - expected).abs() < 1e-10);
        }
        let collinear: VecDeque<_> = contexts
            .into_iter()
            .map(|c| {
                c.map(|mut c| {
                    c.values[1] = c.values[0];
                    c
                })
            })
            .collect();
        let f = Fit::learn(&samples, &collinear, 2, 3.25);
        assert_eq!(f.support, [true; 3]);
        assert!(f.beta.iter().all(|v| v.is_finite()));
        assert!((f.beta[0] - f.beta[1]).abs() < 1e-10);
    }

    #[test]
    fn learned_signed_slopes_missing_rejection_does_not_block_kv() {
        for sign in [-1.0, 1.0] {
            let (s, c) = data(sign);
            let f = Fit::learn(&s, &c, 1, 2.0);
            assert_eq!(f.support, [true, false, false]);
            assert!(f.beta[0] * sign > 0.0);
            assert_eq!(f.beta[1..], [0.0, 0.0]);
            assert_eq!(f.count, 32);
        }
    }
    #[test]
    fn unsupported_constant_small_and_dominant_samples_fall_back() {
        let (mut s, mut c) = data(1.0);
        for ctx in c.iter_mut().flatten() {
            ctx.values[0] = Some(0.5);
        }
        assert!(!Fit::learn(&s, &c, 1, 1.0).support[0]);
        let (_, c) = data(1.0);
        s[0].1 = 1e9;
        assert!(!Fit::learn(&s, &c, 1, 1.0).support[0]);
        assert!(
            !Fit::learn(
                &s.into_iter().take(11).collect(),
                &c.into_iter().take(11).collect(),
                1,
                1.0
            )
            .support[0]
        );
    }
    #[test]
    fn trajectory_has_no_idle_windfall_and_missing_context_drops_stale_effect() {
        let now = Instant::now();
        let (s, c) = data(1.0);
        let mut m = Model {
            fit: Fit::learn(&s, &c, 1, 2.0),
            ..Model::default()
        };
        let ctx = c.back().unwrap().unwrap();
        m.update(&ctx, 4.0, 1.0, now);
        assert_eq!(m.prediction(&ctx, 1.0, now, true).gain, 1.0);
        assert!(
            (m.prediction(&ctx, 1.0, now + Duration::from_millis(100), true)
                .gain
                - 1.05)
                .abs()
                < 1e-8
        );
        let missing = Context::default();
        assert_eq!(m.prediction(&missing, 2.0, now, true).gain, 2.0);
        m.update(&missing, 4.0, 2.0, now);
        m.update(&ctx, 4.0, 2.0, now + Duration::from_secs(100));
        assert_eq!(
            m.prediction(&ctx, 2.0, now + Duration::from_secs(100), true)
                .gain,
            2.0
        );
    }
    #[test]
    fn causal_buckets_and_unknown_coverage() {
        let now = Instant::now();
        let mut a = Activity::new(now);
        for _ in 0..16 {
            a.record(now, Outcome::FirstText);
            a.record(now, Outcome::Admitted);
        }
        a.record(now + Duration::from_secs(1), Outcome::Timeout);
        assert_eq!(
            a.context(now, Some(0.2)).values,
            [Some(0.2), Some(0.0), Some(0.0)]
        );
        assert_eq!(
            a.context(now + Duration::from_secs(60), None).values,
            [None, None, None]
        );
        for _ in 0..3 {
            a.record(now + Duration::from_secs(1), Outcome::Unknown);
        }
        assert_eq!(
            a.context(now + Duration::from_secs(1), None).values[1],
            None
        );
    }
    #[test]
    fn untracked_admission_suppresses_cutoff_without_inventing_terminal_outcome() {
        let now = Instant::now();
        let mut activity = Activity::new(now);
        for _ in 0..16 {
            activity.record(now, Outcome::FirstText);
            activity.record(now, Outcome::Admitted);
        }
        activity.record(now, Outcome::UntrackedAdmitted);
        let context = activity.context(now, None);
        assert_eq!(context.counts.admitted, 17);
        assert_eq!(context.counts.untracked_admitted, 1);
        assert_eq!(context.counts.unknown_pretext_end, 0);
        assert_eq!(context.counts.upstream_timeout, 0);
        assert_eq!(context.values[1], None);
        assert_eq!(context.values[2], Some(0.0));
    }

    #[test]
    fn correction_is_bounded_and_out_of_range_neutral() {
        let (s, c) = data(100.0);
        let f = Fit::learn(&s, &c, 1, 1.0);
        assert_eq!(f.beta[0], 2.0);
        assert_eq!(f.correction(&c.back().unwrap().unwrap()).0, 0.5);
        assert_eq!(
            f.correction(&Context {
                values: [Some(1.1), None, None],
                ..Context::default()
            })
            .0,
            0.0
        );
    }
}
