//! Experimental jointly fitted queue/feature gains. No field here affects admission.
//! Rates describe resolved local observations, not native cancellation acknowledgements.
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU8, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use serde::Serialize;

use super::CALIBRATION_MAX_SCALE_CHANGE_PER_SECOND;

pub(super) const METHOD: &str = "feature_joint_gains_deadline_ridge_v3";
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
    pub deadline_untracked_pending: u64,
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
    pub deadline_timely_first_text: u64,
    pub deadline_breach: u64,
    pub deadline_or_cutoff_adverse: u64,
    pub deadline_or_cutoff_unknown: u64,
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
    DeadlineTimely,
    DeadlineBreach,
    ActualCutoff,
    DeadlineUnknown,
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
            Self::DeadlineTimely => "timely_first_text",
            Self::DeadlineBreach => "deadline_breach",
            Self::ActualCutoff => "actual_cutoff",
            Self::DeadlineUnknown => "unknown_pretext_end",
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
        self.deadline_timely_first_text = self
            .deadline_timely_first_text
            .saturating_add(other.deadline_timely_first_text);
        self.deadline_breach = self.deadline_breach.saturating_add(other.deadline_breach);
        self.deadline_or_cutoff_adverse = self
            .deadline_or_cutoff_adverse
            .saturating_add(other.deadline_or_cutoff_adverse);
        self.deadline_or_cutoff_unknown = self
            .deadline_or_cutoff_unknown
            .saturating_add(other.deadline_or_cutoff_unknown);
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
        if matches!(outcome, Outcome::DeadlineBreach) {
            self.deadline_breach = self.deadline_breach.saturating_add(1);
        }
        let value = match outcome {
            Outcome::FirstText => &mut self.first_text,
            Outcome::Timeout => &mut self.upstream_timeout,
            Outcome::Disconnect => &mut self.pretext_downstream_disconnect,
            Outcome::Unknown => &mut self.unknown_pretext_end,
            Outcome::Admitted | Outcome::UntrackedAdmitted => &mut self.admitted,
            Outcome::Rejected => &mut self.actual_rejected,
            Outcome::DeadlineTimely => &mut self.deadline_timely_first_text,
            Outcome::DeadlineBreach | Outcome::ActualCutoff => &mut self.deadline_or_cutoff_adverse,
            Outcome::DeadlineUnknown => &mut self.deadline_or_cutoff_unknown,
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
        let adverse = counts.deadline_or_cutoff_adverse;
        let known = counts.deadline_timely_first_text.saturating_add(adverse);
        let resolved = known.saturating_add(counts.deadline_or_cutoff_unknown);
        let attempts = counts.admitted.saturating_add(counts.actual_rejected);
        Context {
            bucket_second: second,
            deadline_untracked_pending: 0,
            activity_origin_time_seconds: super::diagnostic_time(self.origin),
            values: [
                kv.filter(|v| v.is_finite() && (0.0..=1.0).contains(v)),
                (known >= 16
                    && counts.untracked_admitted == 0
                    && (counts.deadline_or_cutoff_unknown as f64) < 0.1 * resolved as f64)
                    .then(|| adverse as f64 / known as f64),
                (attempts >= 16).then(|| counts.actual_rejected as f64 / attempts as f64),
            ],
            counts,
        }
    }
}

/// A bounded local first-text observation, independent of upstream cancellation.
pub(super) const TTFT_DEADLINE: Duration = Duration::from_secs(300);
const MAX_PENDING_DEADLINES: usize = 8192;
#[derive(Clone, Debug)]
pub(super) enum DeadlineHandle {
    Tracked(u64, Arc<AtomicU8>),
    Untracked,
}
// 0 idle, 1 scanning a received chunk, 2 terminal claimed, 3 poll breach.
// Claim before the observation timestamp so a poll cannot overtake timely text
// while its callback waits for the ledger lock. Polling never blocks on scans.
pub(super) struct ObservationClaim(Arc<AtomicU8>);
impl Drop for ObservationClaim {
    fn drop(&mut self) {
        let _ = self
            .0
            .compare_exchange(1, 0, Ordering::SeqCst, Ordering::SeqCst);
    }
}
impl DeadlineHandle {
    pub fn observing(&self) -> Option<ObservationClaim> {
        if let Self::Tracked(_, state) = self {
            state
                .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
                .ok()
                .map(|_| ObservationClaim(Arc::clone(state)))
        } else {
            None
        }
    }
    pub fn claim_terminal(&self) {
        if let Self::Tracked(_, state) = self {
            let _ =
                state.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |s| (s < 2).then_some(2));
        }
    }
}
#[derive(Debug)]
pub(super) struct DeadlineResolution {
    pub started: Instant,
    pub request_id: Option<String>,
    pub outcome: Outcome,
}
#[derive(Debug, Default)]
pub(super) struct Deadlines {
    next: u64,
    pending: HashMap<u64, (Instant, Option<String>, Arc<AtomicU8>)>,
    pub untracked: u64,
}
impl Deadlines {
    pub fn register(&mut self, started: Instant, request_id: Option<&str>) -> DeadlineHandle {
        if self.pending.len() >= MAX_PENDING_DEADLINES || self.next == u64::MAX {
            self.untracked = self.untracked.saturating_add(1);
            return DeadlineHandle::Untracked;
        }
        self.next += 1;
        let state = Arc::new(AtomicU8::new(0));
        self.pending.insert(
            self.next,
            (started, request_id.map(str::to_owned), Arc::clone(&state)),
        );
        DeadlineHandle::Tracked(self.next, state)
    }
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
    pub fn advance(&mut self, now: Instant) -> Vec<DeadlineResolution> {
        let mut resolved = Vec::new();
        self.pending.retain(|_, (started, request_id, state)| {
            if now.saturating_duration_since(*started) < TTFT_DEADLINE {
                return true;
            }
            if state
                .compare_exchange(0, 3, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
            {
                return true;
            }
            resolved.push(DeadlineResolution {
                started: *started,
                request_id: request_id.clone(),
                outcome: Outcome::DeadlineBreach,
            });
            false
        });
        resolved
    }
    pub fn resolve(
        &mut self,
        handle: DeadlineHandle,
        started: Instant,
        request_id: Option<&str>,
        now: Instant,
        terminal: Outcome,
    ) -> Option<DeadlineResolution> {
        let (started, request_id) = match handle {
            DeadlineHandle::Tracked(id, _) => {
                let (started, request_id, _) = self.pending.remove(&id)?;
                (started, request_id)
            }
            DeadlineHandle::Untracked => {
                self.untracked = self.untracked.saturating_sub(1);
                (started, request_id.map(str::to_owned))
            }
        };
        let outcome = if now.saturating_duration_since(started) >= TTFT_DEADLINE {
            Outcome::DeadlineBreach
        } else {
            match terminal {
                Outcome::FirstText => Outcome::DeadlineTimely,
                Outcome::Timeout | Outcome::Disconnect => Outcome::ActualCutoff,
                _ => Outcome::DeadlineUnknown,
            }
        };
        Some(DeadlineResolution {
            started,
            request_id,
            outcome,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct Fit {
    /// Compatibility alias for the three uncentered contextual coefficients.
    pub beta: [f64; 3],
    pub coefficients: [f64; 4],
    pub mean: [f64; 3],
    pub minimum: [f64; 3],
    pub maximum: [f64; 3],
    pub support: [bool; 3],
    pub count: usize,
    pub ess: f64,
    pub revision: u64,
}
impl Default for Fit {
    fn default() -> Self {
        Self {
            beta: [0.0; 3],
            coefficients: [1.0, 0.0, 0.0, 0.0],
            mean: [0.0; 3],
            minimum: [0.0; 3],
            maximum: [0.0; 3],
            support: [false; 3],
            count: 0,
            ess: 0.0,
            revision: 0,
        }
    }
}
impl Fit {
    /// At most seven tiny common-row fits. Missing inputs are never imputed.
    pub fn learn(
        samples: &VecDeque<(f64, f64)>,
        contexts: &VecDeque<Option<Context>>,
        revision: u64,
    ) -> Self {
        let empty = Self {
            revision,
            ..Self::default()
        };
        if samples.len() != contexts.len() {
            return empty;
        }
        // Largest supported subset first; deterministic KV, deadline-or-cutoff, rejection priority.
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
            let mut covariance = [[0.0; 4]; 4];
            let mut rhs = [0.0; 4];
            let mut variance = [0.0; 3];
            for &(ratio, n, ctx) in &rows {
                let w = (n / largest).powi(2) / sumw;
                let mut x = [1.0, 0.0, 0.0, 0.0];
                for j in 0..3 {
                    if mask & (1 << j) != 0 {
                        // Common-row filtering above guarantees a present input.
                        x[j + 1] = ctx.values[j].unwrap_or_default();
                        variance[j] += w * (x[j + 1] - fit.mean[j]).powi(2);
                    }
                }
                for j in 0..4 {
                    rhs[j] += w * x[j] * ratio;
                    for k in 0..4 {
                        covariance[j][k] += w * x[j] * x[k];
                    }
                }
            }
            if (0..3).any(|j| {
                mask & (1 << j) != 0
                    && (variance[j] < 1e-4 || fit.maximum[j] - fit.minimum[j] < 0.05)
            }) {
                continue;
            }
            // Joint ridge on [N, N*KV, N*cutoff, N*reject], normalized by sum N².
            // A neutral prior [1,0,0,0] is independent of the scalar comparator.
            for (j, row) in covariance.iter_mut().enumerate() {
                row[j] += RIDGE;
            }
            rhs[0] += RIDGE;
            for j in 0..3 {
                fit.support[j] = mask & (1 << j) != 0;
                if !fit.support[j] {
                    fit.minimum[j] = 0.0;
                    fit.maximum[j] = 0.0;
                }
            }
            if let Some(mut coefficients) = solve(covariance, rhs) {
                // Coefficient clipping is a bounded approximation to ridge,
                // not a constrained-ridge optimizer.
                coefficients[0] = coefficients[0].max(0.0);
                for value in &mut coefficients[1..] {
                    *value = value.clamp(-MAX_BETA, MAX_BETA);
                }
                fit.coefficients = coefficients;
                fit.beta.copy_from_slice(&coefficients[1..]);
                return fit;
            }
        }
        empty
    }
    fn active(&self, context: &Context) -> [bool; 3] {
        std::array::from_fn(|j| {
            self.support[j]
                && context.values[j]
                    .is_some_and(|x| x.is_finite() && x >= self.minimum[j] && x <= self.maximum[j])
        })
    }
    fn supported(&self, active: [bool; 3]) -> bool {
        // A jointly fitted intercept must not be reused after silently dropping
        // one of its fitted predictors. Fall back as a whole in that case.
        active == self.support && active.iter().any(|x| *x)
    }
}
fn solve(a: [[f64; 4]; 4], b: [f64; 4]) -> Option<[f64; 4]> {
    let mut l = [[0.0; 4]; 4];
    for i in 0..4 {
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
    let mut y = [0.0; 4];
    for i in 0..4 {
        y[i] = (b[i] - (0..i).map(|j| l[i][j] * y[j]).sum::<f64>()) / l[i][i];
    }
    let mut x = [0.0; 4];
    for i in (0..4).rev() {
        x[i] = (y[i] - ((i + 1)..4).map(|j| l[j][i] * x[j]).sum::<f64>()) / l[i][i];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// One shared interpolation fraction bounds the L1 coefficient movement.
/// Thus each coefficient AND the gain at any fixed inputs in [0,1] move by
/// at most 0.5/second. Input/queue changes themselves are never rate limited.
#[derive(Debug)]
struct Trajectory {
    anchor: [f64; 4],
    target: [f64; 4],
    anchor_time: Instant,
}
impl Trajectory {
    fn applied(&self, now: Instant) -> [f64; 4] {
        let distance: f64 = (0..4)
            .map(|j| (self.target[j] - self.anchor[j]).abs())
            .sum();
        let budget = CALIBRATION_MAX_SCALE_CHANGE_PER_SECOND
            * now
                .saturating_duration_since(self.anchor_time)
                .as_secs_f64();
        let fraction = if distance > 0.0 {
            (budget / distance).min(1.0)
        } else {
            1.0
        };
        std::array::from_fn(|j| self.anchor[j] + fraction * (self.target[j] - self.anchor[j]))
    }
    fn retarget(&mut self, target: [f64; 4], now: Instant) {
        self.anchor = self.applied(now);
        self.target = target;
        self.anchor_time = now;
    }
}

/// Return the effective coefficients after the contextual +/-0.5 safety cap.
/// Scale contextual terms together to preserve an exact signed decomposition.
fn effective(coefficients: [f64; 4], inputs: [f64; 4]) -> [f64; 4] {
    let shift: f64 = (1..4).map(|j| coefficients[j] * inputs[j]).sum();
    let bounded = shift.clamp(-MAX_SHIFT, MAX_SHIFT).max(-coefficients[0]);
    let factor = if shift == 0.0 { 1.0 } else { bounded / shift };
    let mut result = coefficients;
    for value in &mut result[1..] {
        *value *= factor;
    }
    result
}
fn gain(coefficients: [f64; 4], inputs: [f64; 4]) -> f64 {
    coefficients
        .iter()
        .zip(inputs)
        .map(|(g, x)| g * x)
        .sum::<f64>()
        .max(0.0)
}
#[derive(Debug, Default)]
pub(super) struct Model {
    pub fit: Fit,
    trajectory: Option<Trajectory>,
    active: [bool; 3],
}
#[derive(Serialize)]
pub(super) struct Prediction {
    pub gain: f64,
    pub target_gain: f64,
    pub contextual: bool,
    pub reason: &'static str,
    pub active: [bool; 3],
    pub applied_coefficients: [f64; 4],
    pub target_coefficients: [f64; 4],
    pub effective_coefficients: [f64; 4],
    pub target_effective_coefficients: [f64; 4],
    pub contribution_seconds: [f64; 4],
    pub trajectory_anchor_coefficients: Option<[f64; 4]>,
    pub trajectory_anchor_gain: Option<f64>,
    pub trajectory_anchor_time_seconds: Option<f64>,
}
impl Model {
    pub fn invalidate(&mut self) {
        self.trajectory = None;
        self.active = [false; 3];
    }
    pub fn update(&mut self, context: &Context, scalar_applied: f64, now: Instant) {
        let active = self.fit.active(context);
        if !self.fit.supported(active) {
            self.invalidate();
            return;
        }
        if self.active != active {
            self.trajectory = None;
        }
        self.active = active;
        let trajectory = self.trajectory.get_or_insert(Trajectory {
            anchor: [scalar_applied, 0.0, 0.0, 0.0],
            target: [scalar_applied, 0.0, 0.0, 0.0],
            anchor_time: now,
        });
        trajectory.retarget(self.fit.coefficients, now);
    }
    pub fn prediction(
        &self,
        context: &Context,
        scalar: f64,
        now: Instant,
        usable: bool,
        n: f64,
    ) -> Prediction {
        let active = self.fit.active(context);
        let trajectory = self
            .trajectory
            .as_ref()
            .filter(|_| usable && active == self.active && self.fit.supported(active));
        let inputs = std::array::from_fn(|j| {
            if j == 0 {
                1.0
            } else if active[j - 1] {
                context.values[j - 1].unwrap_or_default()
            } else {
                0.0
            }
        });
        let applied = trajectory.map_or([scalar, 0.0, 0.0, 0.0], |t| t.applied(now));
        let target = trajectory.map_or(applied, |t| t.target);
        let effective_coefficients = effective(applied, inputs);
        let target_effective_coefficients = effective(target, inputs);
        Prediction {
            gain: gain(effective_coefficients, inputs),
            target_gain: gain(target_effective_coefficients, inputs),
            contextual: trajectory.is_some(),
            reason: if !usable {
                "snapshot_unavailable"
            } else if trajectory.is_some() {
                "context_supported"
            } else {
                "legacy_fallback_no_current_support"
            },
            active,
            applied_coefficients: applied,
            target_coefficients: target,
            effective_coefficients,
            target_effective_coefficients,
            contribution_seconds: std::array::from_fn(|j| {
                n * effective_coefficients[j] * inputs[j]
            }),
            trajectory_anchor_coefficients: trajectory.map(|t| t.anchor),
            trajectory_anchor_gain: trajectory.map(|t| gain(effective(t.anchor, inputs), inputs)),
            trajectory_anchor_time_seconds: trajectory
                .map(|t| super::diagnostic_time(t.anchor_time)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
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
    fn deadline_poll_is_once_and_late_terminals_do_not_reclassify() {
        let start = Instant::now();
        for terminal in [
            Outcome::FirstText,
            Outcome::Timeout,
            Outcome::Disconnect,
            Outcome::Unknown,
        ] {
            let mut d = Deadlines::default();
            let h = d.register(start, Some("request"));
            assert!(d.advance(start + Duration::from_millis(299_999)).is_empty());
            let resolved = d.advance(start + TTFT_DEADLINE);
            assert_eq!(resolved.len(), 1);
            assert!(matches!(resolved[0].outcome, Outcome::DeadlineBreach));
            assert!(d.advance(start + Duration::from_secs(420)).is_empty());
            assert!(d
                .resolve(
                    h,
                    start,
                    Some("request"),
                    start + Duration::from_secs(420),
                    terminal
                )
                .is_none());
        }
    }
    #[test]
    fn terminal_resolves_deadline_without_a_poll_and_early_cutoff_is_not_breach() {
        let start = Instant::now();
        for (elapsed, terminal, expected) in [
            (299.99, Outcome::Disconnect, "actual_cutoff"),
            (299.99, Outcome::Timeout, "actual_cutoff"),
            (299.99, Outcome::Unknown, "unknown_pretext_end"),
            (299.99, Outcome::FirstText, "timely_first_text"),
            (300.0, Outcome::FirstText, "deadline_breach"),
        ] {
            let mut d = Deadlines::default();
            let h = d.register(start, None);
            h.claim_terminal();
            let r = d
                .resolve(
                    h,
                    start,
                    None,
                    start + Duration::from_secs_f64(elapsed),
                    terminal,
                )
                .unwrap();
            assert_eq!(r.outcome.name(), expected);
            assert!(d.advance(start + Duration::from_secs(420)).is_empty());
        }
    }
    #[test]
    fn timely_scan_claim_cannot_be_overtaken_by_poll_waiting_on_ledger() {
        let start = Instant::now();
        let mut d = Deadlines::default();
        let h = d.register(start, None);
        let scan = h.observing().unwrap();
        // The bytes were observed at299.99; the ledger callback arrives after300.
        let observed = start + Duration::from_millis(299_990);
        assert!(d.advance(start + TTFT_DEADLINE).is_empty());
        h.claim_terminal();
        drop(scan);
        assert!(d.advance(start + Duration::from_secs(301)).is_empty());
        assert!(matches!(
            d.resolve(h, start, None, observed, Outcome::FirstText)
                .unwrap()
                .outcome,
            Outcome::DeadlineTimely
        ));
        let h = d.register(start, None);
        let scan = h.observing().unwrap();
        drop(scan);
        // Metadata/malformed/no-text scans release the claim; next poll can resolve.
        assert_eq!(d.advance(start + TTFT_DEADLINE).len(), 1);
        assert!(h.observing().is_none());
    }
    #[test]
    fn deadline_tracking_is_bounded_and_overflow_drains() {
        let start = Instant::now();
        let mut d = Deadlines::default();
        for _ in 0..MAX_PENDING_DEADLINES {
            assert!(matches!(
                d.register(start, None),
                DeadlineHandle::Tracked(_, _)
            ));
        }
        let h = d.register(start, None);
        assert!(matches!(h, DeadlineHandle::Untracked));
        assert_eq!(d.pending_count(), MAX_PENDING_DEADLINES);
        assert_eq!(d.untracked, 1);
        assert!(matches!(
            d.resolve(
                h,
                start,
                None,
                start + Duration::from_secs(1),
                Outcome::Unknown
            )
            .unwrap()
            .outcome,
            Outcome::DeadlineUnknown
        ));
        assert_eq!(d.untracked, 0);
        assert_eq!(
            d.advance(start + TTFT_DEADLINE).len(),
            MAX_PENDING_DEADLINES
        );
        assert_eq!(d.pending_count(), 0);
        d.next = u64::MAX;
        assert!(matches!(d.register(start, None), DeadlineHandle::Untracked));
    }
    #[test]
    fn deadline_union_keeps_actual_cutoff_diagnostics_separate() {
        let start = Instant::now();
        let mut a = Activity::new(start);
        for _ in 0..15 {
            a.record(start, Outcome::DeadlineTimely);
        }
        a.record(start, Outcome::DeadlineBreach);
        a.record(start, Outcome::Timeout); // Later terminal is diagnostic only.
        a.record(start, Outcome::Unknown); // Already-breached unknown isn't union unknown.
        let c = a.context(start, None);
        assert_eq!(c.values[1], Some(1. / 16.));
        assert_eq!(c.counts.deadline_breach, 1);
        assert_eq!(c.counts.upstream_timeout, 1);
        assert_eq!(c.counts.deadline_or_cutoff_unknown, 0);
        assert_eq!(
            a.context(start + Duration::from_secs(60), None).values[1],
            None
        );
    }

    #[test]
    fn joint_ridge_satisfies_normal_equations_and_collinear_remains_finite() {
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
        let f = Fit::learn(&samples, &contexts, 1);
        assert_eq!(f.support, [true; 3]);
        // Verify all four joint normal equations, including intercept shrinkage
        // toward one. The scalar comparator is deliberately not an input.
        let mut residual = [0.0; 4];
        for ((ratio, _), context) in samples.iter().zip(&contexts) {
            let c = context.unwrap();
            let x = [
                1.0,
                c.values[0].unwrap(),
                c.values[1].unwrap(),
                c.values[2].unwrap(),
            ];
            let error = gain(f.coefficients, x) - ratio;
            for j in 0..4 {
                residual[j] += x[j] * error / 32.0;
            }
        }
        for (j, value) in residual.iter().enumerate() {
            let prior = if j == 0 { 1.0 } else { 0.0 };
            assert!((value + RIDGE * (f.coefficients[j] - prior)).abs() < 1e-10);
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
        let f = Fit::learn(&samples, &collinear, 2);
        assert_eq!(f.support, [true; 3]);
        assert!(f.beta.iter().all(|v| v.is_finite()));
        assert!((f.beta[0] - f.beta[1]).abs() < 1e-10);
    }

    #[test]
    fn learned_signed_slopes_missing_rejection_does_not_block_kv() {
        for sign in [-1.0, 1.0] {
            let (s, c) = data(sign);
            let f = Fit::learn(&s, &c, 1);
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
        assert!(!Fit::learn(&s, &c, 1).support[0]);
        let (_, c) = data(1.0);
        s[0].1 = 1e9;
        assert!(!Fit::learn(&s, &c, 1).support[0]);
        assert!(
            !Fit::learn(
                &s.into_iter().take(11).collect(),
                &c.into_iter().take(11).collect(),
                1
            )
            .support[0]
        );
    }
    #[test]
    fn trajectory_has_no_idle_windfall_and_missing_context_drops_stale_effect() {
        let now = Instant::now();
        let (s, c) = data(1.0);
        let mut m = Model {
            fit: Fit::learn(&s, &c, 1),
            ..Model::default()
        };
        let ctx = c.back().unwrap().unwrap();
        m.update(&ctx, 1.0, now);
        assert_eq!(m.prediction(&ctx, 1.0, now, true, 100.0).gain, 1.0);
        assert!(
            (m.prediction(&ctx, 1.0, now + Duration::from_millis(100), true, 100.0)
                .gain
                - 1.05)
                .abs()
                < 1e-8
        );
        let missing = Context::default();
        assert_eq!(m.prediction(&missing, 2.0, now, true, 100.0).gain, 2.0);
        m.update(&missing, 2.0, now);
        m.update(&ctx, 2.0, now + Duration::from_secs(100));
        assert_eq!(
            m.prediction(&ctx, 2.0, now + Duration::from_secs(100), true, 100.0)
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
            a.record(now, Outcome::DeadlineTimely);
            a.record(now, Outcome::Admitted);
        }
        a.record(now + Duration::from_secs(1), Outcome::Timeout);
        a.record(now + Duration::from_secs(1), Outcome::ActualCutoff);
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
            a.record(now + Duration::from_secs(1), Outcome::DeadlineUnknown);
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
            activity.record(now, Outcome::DeadlineTimely);
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
    fn joint_gain_cross_language_fixture() {
        let (samples, contexts): (VecDeque<_>, VecDeque<_>) = (0..32)
            .map(|i| {
                let x: [f64; 3] = std::array::from_fn(|j| u8::from(i & (1 << j) != 0) as f64);
                (
                    (
                        1.4 + 0.7 * x[0] - 0.4 * x[1] + 0.2 * x[2],
                        20.0 + 5.0 * i as f64,
                    ),
                    Some(Context {
                        values: x.map(Some),
                        ..Context::default()
                    }),
                )
            })
            .unzip();
        let now = Instant::now();
        let mut model = Model {
            fit: Fit::learn(&samples, &contexts, 1),
            ..Model::default()
        };
        let context = Context {
            values: [Some(0.2), Some(0.7), Some(0.4)],
            ..Context::default()
        };
        model.update(&context, 1.0, now);
        let predictions: Vec<_> = [0.0, 0.25, 2.0, 20.0]
            .into_iter()
            .map(|seconds| {
                let p = model.prediction(
                    &context,
                    1.0,
                    now + Duration::from_secs_f64(seconds),
                    true,
                    125.0,
                );
                serde_json::json!({"elapsed":seconds,"prediction":p})
            })
            .collect();
        assert_eq!(model.fit.support, [true; 3]);
        println!(
            "JOINT_GAIN_FIXTURE {}",
            serde_json::json!({"fit":model.fit,"predictions":predictions})
        );
    }

    #[test]
    fn neutral_prior_and_signed_independent_weights() {
        let (mut samples, contexts) = data(0.0);
        for sample in &mut samples {
            sample.0 = 1.0;
        }
        let fit = Fit::learn(&samples, &contexts, 1);
        for (actual, prior) in fit.coefficients.into_iter().zip([1.0, 0.0, 0.0, 0.0]) {
            assert!((actual - prior).abs() < 1e-10);
        }
        assert_eq!(Fit::default().coefficients, [1.0, 0.0, 0.0, 0.0]);
        let (samples, contexts) = data(100.0);
        let fit = Fit::learn(&samples, &contexts, 1);
        assert_eq!(fit.beta[0], 2.0);
    }

    fn supported_model(now: Instant, target: [f64; 4]) -> (Model, Context) {
        let context = Context {
            values: [Some(1.0); 3],
            ..Context::default()
        };
        let mut model = Model {
            fit: Fit {
                coefficients: target,
                support: [true; 3],
                minimum: [0.0; 3],
                maximum: [1.0; 3],
                ..Fit::default()
            },
            ..Model::default()
        };
        model.update(&context, 1.0, now);
        (model, context)
    }

    #[test]
    fn coefficient_and_combined_motion_are_bounded_for_all_current_inputs() {
        let now = Instant::now();
        let (mut model, context) = supported_model(now, [3.0, 2.0, -2.0, 1.0]);
        let later = now + Duration::from_millis(100);
        let p = model.prediction(&context, 1.0, later, true, 100.0);
        let distance: f64 = p
            .applied_coefficients
            .iter()
            .zip([1.0, 0.0, 0.0, 0.0])
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!((distance - 0.05).abs() < 1e-10);
        for mask in 0..8 {
            let c = Context {
                values: std::array::from_fn(|j| Some(f64::from(mask & (1 << j) != 0))),
                ..context
            };
            let p = model.prediction(&c, 1.0, later, true, 100.0);
            assert!((p.gain - 1.0).abs() <= 0.05 + 1e-10);
        }
        let before = p.applied_coefficients;
        model.fit.coefficients = [0.1, -2.0, 2.0, -1.0];
        model.update(&context, 1.0, later);
        assert_eq!(
            model
                .prediction(&context, 1.0, later, true, 100.0)
                .applied_coefficients,
            before
        );
        let after = model.prediction(
            &context,
            1.0,
            later + Duration::from_millis(100),
            true,
            100.0,
        );
        let distance: f64 = after
            .applied_coefficients
            .iter()
            .zip(before)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(distance <= 0.05 + 1e-10);
    }

    #[test]
    fn bounded_context_has_exact_signed_seconds_decomposition_and_responsive_queue() {
        let now = Instant::now();
        for target in [[2.0, 2.0, -0.25, 1.0], [0.1, -2.0, 0.25, -1.0]] {
            let (model, context) = supported_model(now, target);
            let p = model.prediction(&context, 1.0, now + Duration::from_secs(100), true, 100.0);
            assert!((p.contribution_seconds.iter().sum::<f64>() - p.gain * 100.0).abs() < 1e-10);
            let shift = p.gain - p.applied_coefficients[0];
            assert!(shift.abs() <= MAX_SHIFT + 1e-10 && p.gain >= 0.0);
            assert!(p.contribution_seconds[1] * target[1] > 0.0);
            assert!(p.contribution_seconds[2] * target[2] > 0.0);
            let doubled =
                model.prediction(&context, 1.0, now + Duration::from_secs(100), true, 200.0);
            for (a, b) in doubled
                .contribution_seconds
                .iter()
                .zip(p.contribution_seconds)
            {
                assert!((a - 2.0 * b).abs() < 1e-10);
            }
            let empty = model.prediction(&context, 1.0, now + Duration::from_secs(100), true, 0.0);
            assert_eq!(empty.contribution_seconds, [0.0; 4]);
        }
    }

    #[test]
    fn missing_out_of_range_stale_or_changed_subset_never_reuses_joint_intercept() {
        let now = Instant::now();
        let (mut model, context) = supported_model(now, [3.0, 1.0, -1.0, 0.5]);
        let later = now + Duration::from_secs(100);
        for missing in [None, Some(f64::NAN), Some(1.1), Some(-0.1)] {
            let mut c = context;
            c.values[1] = missing;
            let p = model.prediction(&c, 1.2, later, true, 100.0);
            assert_eq!(p.gain, 1.2);
            assert!(!p.contextual);
            assert_eq!(p.applied_coefficients, [1.2, 0.0, 0.0, 0.0]);
        }
        assert_eq!(
            model.prediction(&context, 1.2, later, false, 100.0).gain,
            1.2
        );
        let mut partial = context;
        partial.values[1] = None;
        model.update(&partial, 1.2, later);
        model.update(&context, 1.2, later + Duration::from_secs(100));
        assert_eq!(
            model
                .prediction(&context, 1.2, later + Duration::from_secs(100), true, 100.0)
                .gain,
            1.2
        );
        model.fit.support[1] = false;
        model.fit.coefficients[2] = 0.0;
        model.update(&context, 1.4, later + Duration::from_secs(200));
        assert_eq!(
            model
                .prediction(&context, 1.4, later + Duration::from_secs(200), true, 100.0)
                .gain,
            1.4
        );
        model.invalidate();
        assert!(
            !model
                .prediction(&context, 1.0, later, true, 100.0)
                .contextual
        );
    }
}
