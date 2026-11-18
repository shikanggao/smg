//! Calibrated worker-backlog admission, independent of the routing policy.
//!
//! The ledger lock spans admission and selection/credit. Polls capture a
//! watermark before I/O so admissions during a poll survive its publication.
//!
//! One TTFT predictor: a source-qualified coherent native work/service estimate,
//! with a fresh baseline + queue-work/capacity fallback when native proof is absent.
//! Learning and native provenance are internal components, not operator-selectable
//! algorithms. A configured budget defaults to shadow mode; explicit enforcement
//! uses the same immutable prediction captured for operational metrics.
//! First-text outcomes train only their original epoch and admission snapshot.
//! Cancellation intent never counts as released work until native acknowledgement.
//! Two 300-second history banks, lifecycle/profile resets, and original source
//! expiry bound retention; publication cannot refresh old evidence.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, OnceLock, Weak,
    },
    time::Instant,
};

use axum::response::Response;
use openai_protocol::worker::{RuntimeType, WorkerLoadResponse};
use parking_lot::{Mutex, MutexGuard};
use serde::{Deserialize, Serialize};

use super::{
    expected_wait::{
        ExpectedWait, DEFAULT_DISPATCH_BLOCKING_FACTOR, DEFAULT_MEAN_PREFILL_TOKENS,
        DEFAULT_THROUGHPUT,
    },
    ttft_model::basis,
    Worker, WorkerRegistry,
};
use crate::{
    config::{ConfigError, ConfigResult},
    routers::common::overload,
};
mod coherent_pair;
use super::{
    ttft_accounting::{deadline_recovery, feature, native_deadline},
    ttft_model::{decision, pending_work, recovery},
};
mod survival;
pub(crate) use survival::SgRawEvidence;
static NATIVE_GATEWAY_EPOCH: OnceLock<uuid::Uuid> = OnceLock::new();

struct NativeDeadlineObservation {
    transport_timeout: bool,
}
use feature::{DeadlineHandle, DeadlineResolution, Deadlines, Outcome as FeatureOutcome};
use recovery::{RecoveryForecast, RecoveryTracker};

pub(crate) use super::ttft_model::prediction::SelectedPrediction;

/// All defaults are starting points for calibration, not capacity guarantees.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct EstimatedWaitConfig {
    pub max_estimated_wait_secs: Option<f64>,
    /// Deprecated compatibility field; the admission formula has no KV term.
    /// Only zero is accepted so old disabled-KV configurations still load.
    pub estimated_wait_kv_pressure_weight: f64,
    pub estimated_wait_dispatch_blocking_factor: f64,
    pub estimated_wait_mean_prefill_tokens: u32,
    #[serde(
        alias = "estimated_wait_default_throughput",
        alias = "estimated_wait_min_prefill_throughput"
    )]
    /// Aggregate prefill-capacity floor/fallback; the field name is retained for compatibility.
    pub estimated_wait_fallback_prefill_throughput: f64,
    pub estimated_wait_prompt_size_prior_samples: u32,
    /// Zero disables the waiting-request compatibility proxy.
    pub estimated_wait_queue_tokens_per_request: u32,
    pub estimated_wait_max_snapshot_age_secs: f64,
    /// Record would-reject decisions without enforcing estimated-wait budgets.
    pub estimated_wait_shadow: bool,
    /// Internal learning switch used by component tests; not an operator option.
    #[doc(hidden)]
    #[serde(skip, default = "internal_learning_enabled")]
    pub estimated_wait_calibration_shadow: bool,
    /// Internal context switch used by component tests; not an operator option.
    #[doc(hidden)]
    #[serde(skip, default = "internal_learning_enabled")]
    pub estimated_wait_feature_shadow: bool,
    /// Opt-in native first-output budget for regular HTTP feature streams.
    /// Sent as remaining time at dispatch; requires a verified enforcing engine.
    /// Independent of admission thresholds; absent means no native deadline.
    pub estimated_wait_native_hard_ttft_timeout_secs: Option<f64>,
    /// Same-model deployment baseline for diagnostic calibration until measured.
    pub estimated_wait_provisional_baseline_secs: Option<f64>,
}

fn internal_learning_enabled() -> bool {
    true
}

impl Default for EstimatedWaitConfig {
    fn default() -> Self {
        Self {
            max_estimated_wait_secs: None,
            estimated_wait_kv_pressure_weight: 0.0,
            estimated_wait_dispatch_blocking_factor: DEFAULT_DISPATCH_BLOCKING_FACTOR,
            estimated_wait_mean_prefill_tokens: DEFAULT_MEAN_PREFILL_TOKENS,
            estimated_wait_fallback_prefill_throughput: DEFAULT_THROUGHPUT,
            estimated_wait_prompt_size_prior_samples: 32,
            estimated_wait_queue_tokens_per_request: 0,
            estimated_wait_max_snapshot_age_secs: 30.0,
            estimated_wait_shadow: true,
            estimated_wait_calibration_shadow: true,
            estimated_wait_feature_shadow: true,
            estimated_wait_native_hard_ttft_timeout_secs: None,
            estimated_wait_provisional_baseline_secs: None,
        }
    }
}

impl EstimatedWaitConfig {
    pub fn validate(&self) -> ConfigResult<()> {
        if self
            .estimated_wait_native_hard_ttft_timeout_secs
            .is_some_and(|value| {
                !self.estimated_wait_feature_shadow
                    || !value.is_finite()
                    || value <= 0.0
                    || value > 86400.0
            })
        {
            return Err(ConfigError::InvalidValue { field: "estimated_wait_native_hard_ttft_timeout_secs".into(),
                value: format!("{:?}", self.estimated_wait_native_hard_ttft_timeout_secs),
                reason: "Requires feature shadow and a finite native first-output budget in (0,86400] seconds; omitted by default".into() });
        }
        if self.estimated_wait_feature_shadow && !self.estimated_wait_calibration_shadow {
            return Err(ConfigError::InvalidValue {
                field: "estimated_wait_feature_shadow".into(),
                value: "true".into(),
                reason: "Requires estimated_wait_calibration_shadow".into(),
            });
        }
        if self.estimated_wait_kv_pressure_weight != 0.0 {
            return Err(ConfigError::InvalidValue {
                field: "estimated_wait_kv_pressure_weight".to_string(),
                value: self.estimated_wait_kv_pressure_weight.to_string(),
                reason: "KV pressure was removed from estimated-wait admission; use zero"
                    .to_string(),
            });
        }
        if self
            .estimated_wait_provisional_baseline_secs
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(ConfigError::InvalidValue {
                field: "estimated_wait_provisional_baseline_secs".to_owned(),
                value: self
                    .estimated_wait_provisional_baseline_secs
                    .unwrap_or_default()
                    .to_string(),
                reason: "Must be a finite same-model baseline > 0; omit when unknown".to_owned(),
            });
        }
        for (field, value, allow_zero) in [
            (
                "max_estimated_wait_secs",
                self.max_estimated_wait_secs.unwrap_or(1.0),
                false,
            ),
            (
                "estimated_wait_dispatch_blocking_factor",
                self.estimated_wait_dispatch_blocking_factor,
                true,
            ),
            (
                "estimated_wait_fallback_prefill_throughput",
                self.estimated_wait_fallback_prefill_throughput,
                false,
            ),
            (
                "estimated_wait_mean_prefill_tokens",
                f64::from(self.estimated_wait_mean_prefill_tokens),
                false,
            ),
            (
                "estimated_wait_max_snapshot_age_secs",
                self.estimated_wait_max_snapshot_age_secs,
                false,
            ),
        ] {
            if !value.is_finite() || value < 0.0 || (!allow_zero && value == 0.0) {
                return Err(ConfigError::InvalidValue {
                    field: field.to_owned(),
                    value: value.to_string(),
                    reason: if allow_zero {
                        "Must be finite and >= 0"
                    } else {
                        "Must be finite and > 0"
                    }
                    .to_owned(),
                });
            }
        }
        if !self.estimated_wait_dispatch_blocking_factor.is_finite()
            || !(0.0..=1.0).contains(&self.estimated_wait_dispatch_blocking_factor)
        {
            return Err(ConfigError::InvalidValue {
                field: "estimated_wait_dispatch_blocking_factor".to_owned(),
                value: self.estimated_wait_dispatch_blocking_factor.to_string(),
                reason: "Must be finite and in [0, 1]".to_owned(),
            });
        }
        Ok(())
    }

    /// Resolve worker overrides using the same metadata as the static guard.
    fn threshold(&self, worker: &Arc<dyn Worker>) -> Option<f64> {
        worker
            .metadata()
            .overload
            .max_estimated_wait_secs
            .or(self.max_estimated_wait_secs)
    }

    /// Runs only at load ingestion, never while selecting a worker.
    fn prepare(&self, load: &WorkerLoadResponse) -> Option<PreparedWait> {
        if load.loads.is_empty()
            || (load.dp_rank_count > 0 && load.dp_rank_count as usize != load.loads.len())
            || load
                .loads
                .iter()
                .any(|rank| rank.num_waiting_reqs < 0 || rank.num_waiting_uncached_tokens < 0)
        {
            return None;
        }
        #[cfg(test)]
        let mut proxy = false;
        let mut blended_prompt_size = None;
        let mut queued = 0.0;
        for rank in &load.loads {
            queued += match rank.num_waiting_uncached_tokens_available {
                Some(false) => {
                    let observed = rank
                        .avg_request_prefill_kv_computed_tokens
                        .or(rank.median_request_prefill_kv_computed_tokens)
                        .filter(|value| value.is_finite() && *value >= 0.0)
                        .zip(rank.prefill_size_sample_count.filter(|count| *count > 0));
                    let configured_prior = if self.estimated_wait_queue_tokens_per_request > 0 {
                        f64::from(self.estimated_wait_queue_tokens_per_request)
                    } else {
                        f64::from(self.estimated_wait_mean_prefill_tokens)
                    };
                    let calibrated_backend = rank.prefill_capacity_learning_active.is_some();
                    let tokens_per_request = observed
                        .map(|(median, count)| {
                            let observed_weight = count as f64;
                            let prior_weight =
                                f64::from(self.estimated_wait_prompt_size_prior_samples);
                            if prior_weight + observed_weight > 0.0 {
                                (configured_prior * prior_weight + median * observed_weight)
                                    / (prior_weight + observed_weight)
                            } else {
                                configured_prior
                            }
                        })
                        .or_else(|| {
                            if calibrated_backend {
                                None
                            } else {
                                rank.avg_request_prefill_kv_computed_tokens
                                    .filter(|value| value.is_finite() && *value > 0.0)
                            }
                        })
                        .unwrap_or_else(|| {
                            if calibrated_backend {
                                configured_prior
                            } else {
                                f64::from(self.estimated_wait_queue_tokens_per_request)
                            }
                        });
                    if tokens_per_request == 0.0 {
                        return None;
                    }
                    blended_prompt_size = Some(tokens_per_request);
                    #[cfg(test)]
                    {
                        proxy = true;
                    }
                    f64::from(rank.num_waiting_reqs) * tokens_per_request
                }
                _ => f64::from(rank.num_waiting_uncached_tokens),
            };
        }
        let capacity_learning_backend = load
            .loads
            .iter()
            .any(|rank| rank.prefill_capacity_learning_active.is_some());
        let learned = load.loads.iter().try_fold(0.0, |sum, rank| {
            rank.learned_prefill_capacity
                .filter(|value| value.is_finite() && *value > 0.0)
                .map(|value| sum + value)
        });
        // Prometheus backends publish learned capacity after the monitor has collected
        // enough saturated intervals. Keep producing estimates before then so
        // operators can observe calibration, but do not let those cold-start
        // estimates drive hard admission decisions.
        let baseline = load.loads.iter().try_fold(0.0_f64, |value, rank| {
            rank.low_load_ttft_seconds
                .filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
                .map(|seconds| value.max(seconds))
        });
        // The extra-delay budget removes baseline from native queue arithmetic.
        // Baseline is still required for total TTFT and observed-delay learning,
        // but its absence must not disable a calibrated queue-capacity signal.
        let capacity_ready = !capacity_learning_backend
            || learned.is_some_and(|value| value.is_finite() && value > 0.0);
        // Queue work is uncached prompt tokens, so its rate must measure the
        // same work. Output generation throughput never selects this divisor.
        let active_prefill = load.loads.iter().try_fold(0.0, |sum, rank| {
            (rank.prefill_capacity_learning_active == Some(true))
                .then_some(rank.prefill_throughput)
                .flatten()
                .filter(|rate| rate.is_finite() && *rate > 0.0)
                .map(|rate| sum + rate)
        });
        let capacity = learned
            .filter(|rate| rate.is_finite() && *rate > 0.0)
            .or(active_prefill);
        let (throughput, fallback) = match capacity {
            Some(value) if value >= self.estimated_wait_fallback_prefill_throughput => {
                (value, false)
            }
            _ => (self.estimated_wait_fallback_prefill_throughput, true),
        };
        let estimate = ExpectedWait::calibrated(
            queued,
            throughput,
            baseline.unwrap_or(0.0),
            self.estimated_wait_dispatch_blocking_factor,
        );
        estimate.seconds(0).is_finite().then_some(PreparedWait {
            size_basis: basis::provenance(
                load,
                f64::from(if self.estimated_wait_queue_tokens_per_request > 0 {
                    self.estimated_wait_queue_tokens_per_request
                } else {
                    self.estimated_wait_mean_prefill_tokens
                }),
                self.estimated_wait_prompt_size_prior_samples,
            ),
            estimate,
            native_estimate: estimate,
            native_waiting: load
                .loads
                .iter()
                .map(|rank| f64::from(rank.num_waiting_reqs))
                .sum(),
            native_running: load
                .loads
                .iter()
                .map(|rank| f64::from(rank.num_running_reqs.max(0)))
                .sum(),
            decision_profile: {
                use std::{
                    collections::hash_map::DefaultHasher,
                    hash::{Hash, Hasher},
                };
                let mut h = DefaultHasher::new();
                let mut ranks: Vec<_> = load
                    .loads
                    .iter()
                    .map(|r| {
                        (
                            r.dp_rank,
                            r.native_request_counts_profile,
                            r.timely_prefill_progress_source.as_deref(),
                        )
                    })
                    .collect();
                ranks.sort_unstable();
                ranks.hash(&mut h);
                h.finish()
            },
            progress_capacity: learned.filter(|v| v.is_finite() && *v > 0.0),
            timely_progress: (!load.loads.is_empty()
                && load.dp_rank_count as usize == load.loads.len()
                && load
                    .loads
                    .iter()
                    .map(|r| r.dp_rank)
                    .collect::<HashSet<_>>()
                    .len()
                    == load.loads.len())
            .then(|| {
                load.loads.iter().try_fold(0.0, |sum, r| {
                    if r.native_request_counts_available != Some(true)
                        || r.native_request_counts_profile.is_none()
                        || !matches!(
                            r.timely_prefill_progress_source.as_deref(),
                            Some("local_compute_tokens" | "prefill_started_requests")
                        )
                        || !r.timely_prefill_progress_interval.is_some_and(|s| {
                            s.is_finite()
                                && s > 0.0
                                && s <= self.estimated_wait_max_snapshot_age_secs
                        })
                    {
                        return None;
                    }
                    r.timely_prefill_progress
                        .filter(|v| v.is_finite() && *v >= 0.0)
                        .map(|v| sum + v)
                })
            })
            .flatten(),
            pending_engine: 0.0,
            #[cfg(test)]
            proxy,
            fallback,
            capacity_ready,
            blended_prompt_size,
            effective_prefill_capacity: throughput,
            native_prefill_capacity: throughput,
            started_capacity_source_key: survival::started_source_key(load),
            backlog_size: load
                .loads
                .iter()
                .map(|rank| f64::from(rank.num_waiting_reqs))
                .sum::<f64>()
                .max(if queued > 0.0 { 1.0 } else { 0.0 }),
            observed_delay_floor: 0.0,
            observed_delay_available: false,
            native_queue_delay: load
                .loads
                .iter()
                .filter_map(|rank| {
                    rank.recent_queue_delay_sample_count
                        .filter(|count| *count >= TTFT_MIN_SAMPLES as u64)?;
                    rank.recent_queue_delay_seconds
                        .filter(|delay| delay.is_finite() && *delay >= 0.0)
                })
                .max_by(f64::total_cmp),
            baseline,
            low_load: queued == 0.0
                && load
                    .loads
                    .iter()
                    .all(|rank| rank.num_waiting_reqs == 0 && rank.num_running_reqs >= 0)
                && load
                    .loads
                    .iter()
                    .map(|rank| i64::from(rank.num_running_reqs))
                    .sum::<i64>()
                    <= 1,
            capacity_learned_ready: !capacity_learning_backend
                || learned.is_some_and(|value| value.is_finite() && value > 0.0),
            capacity_profile_sample_count: load.loads.iter().try_fold(0_u64, |sum, rank| {
                rank.prefill_capacity_sample_count
                    .map(|count| sum + u64::from(count))
            }),
            visibility_delay: 0.0,
            has_backlog: queued > 0.0 || load.loads.iter().any(|rank| rank.num_waiting_reqs > 0),
        })
    }

    #[cfg(test)]
    fn score(&self, load: &WorkerLoadResponse, dispatched: u64) -> Option<(f64, bool, bool)> {
        let prepared = self.prepare(load)?;
        let seconds = prepared.seconds(dispatched);
        seconds
            .is_finite()
            .then_some((seconds, prepared.proxy, prepared.fallback))
    }
}

#[derive(Clone, Copy, Debug)]
struct PreparedWait {
    size_basis: Option<basis::Provenance>,
    estimate: ExpectedWait,
    native_estimate: ExpectedWait,
    native_waiting: f64,
    native_running: f64,
    decision_profile: u64,
    timely_progress: Option<f64>,
    progress_capacity: Option<f64>,
    pending_engine: f64,
    #[cfg(test)]
    proxy: bool,
    fallback: bool,
    capacity_ready: bool,
    blended_prompt_size: Option<f64>,
    effective_prefill_capacity: f64,
    native_prefill_capacity: f64,
    started_capacity_source_key: Option<survival::StartedSourceKey>,
    backlog_size: f64,
    observed_delay_floor: f64,
    observed_delay_available: bool,
    native_queue_delay: Option<f64>,
    baseline: Option<f64>,
    low_load: bool,
    capacity_learned_ready: bool,
    capacity_profile_sample_count: Option<u64>,
    visibility_delay: f64,
    has_backlog: bool,
}

impl PreparedWait {
    /// Queue arithmetic only: no observed floor or visibility adjustment.
    fn queue_excess_seconds(self, dispatched_tokens: u64) -> f64 {
        self.native_estimate.token_work(dispatched_tokens) / self.native_prefill_capacity
    }

    fn measured_delay_only(self) -> bool {
        self.fallback && !self.capacity_learned_ready && self.observed_delay_available
    }

    fn seconds(self, dispatched_tokens: u64) -> f64 {
        let baseline = self.baseline.unwrap_or(0.0);
        if self.measured_delay_only() {
            // Measured delay is useful before rate calibration. Do not let an
            // untrusted default rate amplify that evidence into a huge TTFT.
            return baseline + self.observed_delay_floor;
        }
        baseline
            + (self.estimate.seconds(dispatched_tokens) - baseline).max(self.observed_delay_floor)
    }

    fn excess_seconds(self, dispatched_tokens: u64) -> f64 {
        (self.seconds(dispatched_tokens) - self.baseline.unwrap_or(0.0)).max(0.0)
    }
}

fn invalidate_diagnostic_metrics(worker_url: &str) {
    let worker = worker_url.to_owned();

    metrics::gauge!("smg_estimated_wait_snapshot_age_seconds", "worker" => worker.clone())
        .set(-1.0);

    metrics::gauge!("smg_estimated_wait_capacity_ready", "worker" => worker.clone()).set(-1.0);

    metrics::gauge!("smg_estimated_wait_data_usable", "worker" => worker).set(0.0);
}

const TTFT_WINDOW_SAMPLES: usize = 32;
const TTFT_MIN_SAMPLES: usize = 4;
const MAX_TTFT_EVENT_BYTES: usize = 64 * 1024;
// A diagnostic validity horizon, not a latency SLO or an accuracy guarantee.
// Snapshot age is separate from bank retention, which is 300–600 seconds
// from feedback acceptance, subject to the 32-observation per-bank cap.

// Shared across workers/epochs; diagnostic clock starts at its first use.
pub(super) use super::ttft_accounting::diagnostic_time;

fn poll_outcome(
    generation_matches: bool,
    previous: Option<Instant>,
    started: Instant,
    now: Instant,
    has_load: bool,
    prepared: bool,
    max_age: f64,
) -> &'static str {
    if !generation_matches {
        "generation_changed"
    } else if previous.is_some_and(|last| last > started) {
        "out_of_order"
    } else if previous == Some(started) {
        "duplicate"
    } else if !has_load {
        "failed"
    } else if !prepared {
        "invalid_load"
    } else if now.saturating_duration_since(started).as_secs_f64() > max_age {
        "stale"
    } else {
        "success"
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
enum BaselineSource {
    #[default]
    Unknown,
    Provisional,
    Measured,
}

impl BaselineSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Provisional => "provisional",
            Self::Measured => "measured",
        }
    }
}

fn materially_different_baseline(provisional: f64, measured: f64) -> bool {
    (measured - provisional).abs() > 0.1_f64.max(0.2 * provisional)
}

/// Admission-time inputs travel with the request, not with the mutable load sample.
#[derive(Clone, Debug)]
pub(crate) struct CalibrationForecast {
    generation: Arc<()>,
    calibration_generation: Arc<()>,
    #[cfg(test)]
    epoch: u64,
    baseline: Option<f64>,
    raw_extra: f64,
    raw_fallback_work: survival::FrozenRawWork,
    queue_extra: f64,
    #[cfg(test)]
    raw_baseline: Option<f64>,
    baseline_source: BaselineSource,
    native_capacity_ready: bool,
    raw_ready: bool,
    fresh: bool,
    captured_at: Instant,
    feature_tracking_supported: bool,
    decision_admission: Option<Box<decision::Admission>>,
    basis_admission: Option<Box<basis::Admission>>,
    survival_admission: Option<Box<survival::Admission>>,
    pending_work_admission: Option<Box<pending_work::Admission>>,
    deadline_dispatch: Option<Arc<deadline_recovery::DispatchLease>>,
    native_hard_ttft_timeout_secs: Option<f64>,
    native_deadline_capability_valid_until: Option<Instant>,
    prompt_work: pending_work::Estimate,
}

impl CalibrationForecast {
    pub(crate) fn native_dispatch_headers(
        &self,
        request_started: Instant,
        now: Instant,
    ) -> Result<Option<(&str, Option<u64>)>, &'static str> {
        let Some(lease) = &self.deadline_dispatch else {
            return if self.native_hard_ttft_timeout_secs.is_some() {
                Err("native_deadline_request_not_tracked")
            } else {
                Ok(None)
            };
        };
        let timeout = if let Some(seconds) = self.native_hard_ttft_timeout_secs {
            if self
                .native_deadline_capability_valid_until
                .is_none_or(|expiry| now > expiry)
            {
                return Err("native_deadline_not_verified_or_stale");
            }
            let remaining = seconds - now.saturating_duration_since(request_started).as_secs_f64();
            if remaining < 0.001 {
                return Err("native_deadline_budget_exhausted_before_dispatch");
            }
            Some((remaining * 1000.0).floor() as u64)
        } else {
            None
        };
        Ok(Some((lease.native_attempt(), timeout)))
    }

    pub(crate) fn set_prompt_work(&mut self, hint: openai_protocol::common::PromptWorkHint) {
        self.prompt_work = pending_work::Estimate::from_hint(hint);
    }

    fn seconds(&self) -> f64 {
        self.baseline
            .map_or(-1.0, |baseline| baseline + self.queue_extra)
    }

    fn raw_total_for_fallback(&self) -> Option<survival::RawFallback> {
        if !self.raw_ready || !self.fresh || !self.native_capacity_ready {
            return None;
        }
        survival::RawFallback::new(
            self.baseline?,
            self.baseline_source.as_str(),
            self.raw_extra,
        )
    }

    pub(crate) fn selected_prediction(&self) -> Option<SelectedPrediction> {
        if let Some(pair) = self
            .survival_admission
            .as_ref()
            .and_then(|admission| admission.paired_native_candidate.as_ref())
            .filter(|p| p.selected)
        {
            return pair.total_ttft_seconds.map(|total| SelectedPrediction {
                total,
                evidence_valid: pair
                    .original_raw_expiry
                    .is_some_and(|e| pair.guarded_cutoff <= e)
                    && pair
                        .original_native_expiry
                        .is_some_and(|e| pair.guarded_cutoff <= e)
                    && pair
                        .original_native_receipt
                        .is_some_and(|r| r <= pair.guarded_cutoff),
            });
        }
        // One fallback: current qualified baseline + raw work/capacity estimate.
        // Do not use another fitted scalar/context arm when native proof is absent.
        (self.fresh && self.raw_ready && self.native_capacity_ready)
            .then(|| {
                self.baseline.map(|b| SelectedPrediction {
                    total: b + self.raw_extra,
                    evidence_valid: true,
                })
            })
            .flatten()
    }
}

fn push_ttft_sample(samples: &mut VecDeque<f64>, value: f64) {
    samples.push_back(value);
    while samples.len() > TTFT_WINDOW_SAMPLES {
        samples.pop_front();
    }
}

fn ttft_median(samples: &VecDeque<f64>) -> Option<f64> {
    if samples.len() < TTFT_MIN_SAMPLES {
        return None;
    }
    let mut values: Vec<_> = samples.iter().copied().collect();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    })
}

fn observation_quantile(samples: &VecDeque<f64>, quantile: f64) -> Option<f64> {
    if samples.len() < TTFT_MIN_SAMPLES {
        return None;
    }
    let mut values: Vec<_> = samples.iter().copied().collect();
    values.sort_by(f64::total_cmp);
    Some(values[((values.len() - 1) as f64 * quantile).floor() as usize])
}

struct TtftProbe {
    worker_url: String,
    generation: Arc<()>,
    dispatched: u64,
    low_load: bool,
    backlog_generation: u64,
    backlogged: bool,
    baseline: Option<f64>,
    observation_id: u64,
    backlog_size: f64,
    calibration: Option<CalibrationForecast>,
    request_id: Option<String>,
    pending_work: Option<pending_work::Handle>,
}

/// Observe standard OpenAI SSE without altering or delaying relayed bytes.
/// Empty role/metadata events are not the first visible reasoning/answer text.
pub(crate) struct StreamTtftObserver {
    registry: Weak<WorkerRegistry>,
    probe: Option<TtftProbe>,
    started: Instant,
    first_event: Option<Instant>,
    pending: Vec<u8>,
    done: bool,
    worker_url: String,
    request_id: Option<String>,
    feature_identity: Option<CalibrationForecast>,
    feature_resolved: bool,
    feature_deadline: Option<DeadlineHandle>,
    native_deadline_pending: bool,
    metric_observation: Option<crate::observability::ttft::Observation>,
}

impl StreamTtftObserver {
    /// Do not allocate or parse an additional SSE observer for unconfigured workers.
    pub(crate) fn for_worker(
        registry: &Arc<WorkerRegistry>,
        worker: &dyn Worker,
        started: Instant,
    ) -> Option<Self> {
        worker
            .metadata()
            .overload
            .max_estimated_wait_secs
            .or(registry
                .estimated_wait()
                .config
                .get()
                .and_then(|c| c.max_estimated_wait_secs))
            .is_some()
            .then(|| Self::new(registry, worker.url(), started))
    }

    pub(crate) fn upstream_send_started(&self) {
        if let Some(lease) = self.deadline_dispatch() {
            lease.forwarded();
        }
    }

    fn deadline_dispatch(&self) -> Option<&Arc<deadline_recovery::DispatchLease>> {
        self.feature_identity
            .as_ref()
            .and_then(|f| f.deadline_dispatch.as_ref())
            .or_else(|| {
                self.probe
                    .as_ref()?
                    .calibration
                    .as_ref()?
                    .deadline_dispatch
                    .as_ref()
            })
    }

    pub(crate) fn cancel(&mut self) {
        // Observation stopped by backpressure is unknown, not proof of cutoff.
        self.feature_outcome(FeatureOutcome::Unknown);
        self.finish_probe();
        self.pending.clear();
        self.done = true;
    }
    fn finish_probe(&mut self) {
        if let (Some(registry), Some(probe)) = (self.registry.upgrade(), self.probe.take()) {
            registry.estimated_wait().finish_ttft_probe(&probe);
        }
    }

    pub(crate) fn new(registry: &Arc<WorkerRegistry>, worker_url: &str, started: Instant) -> Self {
        Self {
            registry: Arc::downgrade(registry),
            probe: registry.estimated_wait().ttft_probe(worker_url),
            started,
            first_event: None,
            pending: Vec::new(),
            done: false,
            worker_url: worker_url.to_owned(),
            request_id: None,
            feature_identity: None,
            feature_resolved: false,
            feature_deadline: None,
            native_deadline_pending: false,
            metric_observation: None,
        }
    }

    pub(crate) fn with_metric_observation(
        mut self,
        observation: Option<crate::observability::ttft::Observation>,
    ) -> Self {
        self.metric_observation = observation;
        self
    }

    pub(crate) fn with_request_id(mut self, request_id: Option<&str>) -> Self {
        self.request_id = request_id.map(|id| id.chars().take(256).collect());
        if let Some(probe) = &mut self.probe {
            probe.request_id.clone_from(&self.request_id);
        }
        self
    }

    pub(crate) fn with_calibration(mut self, forecast: Option<CalibrationForecast>) -> Self {
        if let (Some(registry), Some(probe)) = (self.registry.upgrade(), self.probe.as_mut()) {
            registry
                .estimated_wait()
                .register_pending_work(probe, forecast.as_ref());
        }
        self.feature_identity = forecast
            .as_ref()
            .filter(|f| f.feature_tracking_supported)
            .cloned();
        if let (Some(registry), Some(identity)) =
            (self.registry.upgrade(), self.feature_identity.as_ref())
        {
            self.feature_deadline = registry.estimated_wait().register_feature_deadline(
                &self.worker_url,
                self.request_id.as_deref(),
                identity,
                self.started,
            );
        }
        if let Some(probe) = &mut self.probe {
            probe.calibration = forecast;
        }
        self
    }

    pub(crate) fn push(&mut self, bytes: &[u8]) {
        if self.metric_observation.is_none()
            && self.probe.is_none()
            && (self.feature_identity.is_none() || self.feature_resolved)
        {
            return;
        }
        let _observation_claim = self
            .feature_deadline
            .as_ref()
            .and_then(DeadlineHandle::observing);
        let observed_at = Instant::now();
        if let Some((ttft, visibility)) = self.scan(bytes, observed_at) {
            if let Some(lease) = self.deadline_dispatch() {
                lease.first_text();
            }
            self.feature_outcome_at(FeatureOutcome::FirstText, observed_at);

            if let (Some(registry), Some(probe)) = (self.registry.upgrade(), self.probe.take()) {
                registry
                    .estimated_wait()
                    .observe_ttft(probe, ttft, visibility);
            }
        }
    }

    fn feature_outcome(&mut self, outcome: FeatureOutcome) {
        if let Some(deadline) = &self.feature_deadline {
            deadline.claim_terminal();
        }
        self.feature_outcome_at(outcome, Instant::now());
    }
    fn feature_outcome_at(&mut self, outcome: FeatureOutcome, observed_at: Instant) {
        if self.feature_resolved {
            return;
        }
        self.feature_resolved = true;
        if let Some(observation) = &self.metric_observation {
            use crate::observability::ttft::Outcome;
            observation.resolve(
                match outcome {
                    FeatureOutcome::FirstText => Outcome::FirstText,
                    FeatureOutcome::Timeout
                    | FeatureOutcome::Disconnect
                    | FeatureOutcome::NativeDeadline => Outcome::Interrupted,
                    _ => Outcome::Unknown,
                },
                observed_at,
            );
        }
        if let Some(deadline) = &self.feature_deadline {
            deadline.claim_terminal();
        }
        if let (Some(registry), Some(identity)) =
            (self.registry.upgrade(), self.feature_identity.as_ref())
        {
            registry.estimated_wait().record_feature_outcome(
                &self.worker_url,
                self.request_id.as_deref(),
                identity,
                outcome,
                self.feature_deadline.take(),
                self.started,
                observed_at,
            );
        }
    }
    pub(crate) fn upstream_timeout(&mut self) {
        self.feature_outcome(FeatureOutcome::Timeout);
    }
    pub(crate) fn downstream_disconnected(&mut self) {
        self.feature_outcome(FeatureOutcome::Disconnect);
    }

    fn scan(&mut self, bytes: &[u8], now: Instant) -> Option<(f64, f64)> {
        if self.done {
            return None;
        }
        if self.pending.len().saturating_add(bytes.len()) > MAX_TTFT_EVENT_BYTES {
            self.feature_outcome(FeatureOutcome::Unknown);
            self.done = true;
            self.pending.clear();
            return None;
        }
        self.pending.extend_from_slice(bytes);
        while let Some(end) = self.pending.iter().position(|byte| *byte == b'\n') {
            let line: Vec<_> = self.pending.drain(..=end).collect();
            let Some(data) = line.trim_ascii().strip_prefix(b"data:") else {
                continue;
            };
            let data = data.trim_ascii();
            if data == b"[DONE]" {
                if self.native_deadline_pending {
                    self.feature_outcome_at(FeatureOutcome::NativeDeadline, now);
                    self.finish_probe();
                    self.done = true;
                    self.pending.clear();
                    return None;
                }
                continue;
            }
            let Ok(value) = serde_json::from_slice::<serde_json::Value>(data) else {
                continue;
            };
            let first_event = *self.first_event.get_or_insert(now);
            let has_text = value
                .get("choices")
                .and_then(|v| v.as_array())
                .is_some_and(|choices| {
                    choices.iter().any(|choice| {
                        choice
                            .get("text")
                            .and_then(|v| v.as_str())
                            .is_some_and(|v| !v.is_empty())
                            || choice.get("delta").is_some_and(|delta| {
                                ["content", "reasoning", "reasoning_content"]
                                    .iter()
                                    .any(|field| {
                                        delta
                                            .get(*field)
                                            .and_then(|v| v.as_str())
                                            .is_some_and(|v| !v.is_empty())
                                    })
                            })
                    })
                });
            if has_text {
                self.done = true;
                self.pending.clear();
                return Some((
                    now.duration_since(self.started).as_secs_f64(),
                    now.duration_since(first_event).as_secs_f64(),
                ));
            }
            let native_deadline_expected = self
                .feature_identity
                .as_ref()
                .or_else(|| {
                    self.probe
                        .as_ref()
                        .and_then(|probe| probe.calibration.as_ref())
                })
                .is_some_and(|forecast| {
                    forecast.native_hard_ttft_timeout_secs.is_some()
                        && forecast.deadline_dispatch.is_some()
                });
            if native_deadline_expected
                && value.get("error").is_some_and(|error| {
                    error.get("type").and_then(|v| v.as_str()) == Some("native_ttft_deadline")
                        && error
                            .get("message")
                            .and_then(|v| v.as_str())
                            .is_some_and(|v| !v.is_empty())
                })
            {
                self.native_deadline_pending = true;
            }
        }
        None
    }
}

impl Drop for StreamTtftObserver {
    fn drop(&mut self) {
        self.feature_outcome(FeatureOutcome::Unknown);
        self.finish_probe();
    }
}

#[derive(Clone, Debug)]
struct Sample {
    prepared: PreparedWait,
    started: Instant,
    watermark: u64,
    threshold: f64,
    // Latched on publication and dispatch, just like the static overload bit.
    // The request path only reads it and checks freshness.
    overloaded: Option<bool>,
    pending_work_queue: Option<pending_work::Queue>,
    pending_work_capacity_progress_source: Option<String>,
}

#[derive(Debug)]
struct Entry {
    source: Weak<dyn Worker>,
    generation: Arc<()>,
    total_dispatched: u64,
    total_dispatched_requests: u64,
    // Keep ordering even for failed polls, which must supersede older successes.
    last_published: Option<Instant>,
    sample: Option<Sample>,
    low_load_ttfts: VecDeque<f64>,
    low_load_visibility: VecDeque<f64>,
    busy_visibility: VecDeque<f64>,
    observed_delays: VecDeque<(f64, f64)>,
    backlog_generation: u64,
    next_observation_id: u64,
    pending_first_text: HashSet<u64>,
    backlog_peak_size: f64,
    pipeline_idle: bool,
    calibration_generation: Arc<()>,
    calibration_epoch: u64,
    calibration_profile: Option<(bool, Option<u64>)>,
    calibration_empty_polls: u8,
    calibration_baseline: (Option<f64>, BaselineSource),
    recovery: RecoveryTracker,
    feature_deadlines: Deadlines,
    decision: decision::State,
    basis_decision: basis::State,
    survival: survival::State,
    pending_work: pending_work::Inventory,
    pending_work_profile: Option<u64>,
    deadline_work: Arc<std::sync::Mutex<deadline_recovery::AcceptedWork>>,
    native_deadline: native_deadline::State,
    coherent_pair: coherent_pair::State,
}

impl Entry {
    fn new(worker: &Arc<dyn Worker>) -> Self {
        Self {
            source: Arc::downgrade(worker),
            generation: Arc::new(()),
            total_dispatched: 0,
            total_dispatched_requests: 0,
            last_published: None,
            sample: None,
            low_load_ttfts: VecDeque::new(),
            low_load_visibility: VecDeque::new(),
            busy_visibility: VecDeque::new(),
            observed_delays: VecDeque::new(),
            backlog_generation: 0,
            next_observation_id: 0,
            pending_first_text: HashSet::new(),
            backlog_peak_size: 0.0,
            pipeline_idle: true,
            calibration_generation: Arc::new(()),
            calibration_epoch: 0,
            calibration_profile: None,
            calibration_empty_polls: 0,
            calibration_baseline: (None, BaselineSource::Unknown),
            recovery: RecoveryTracker::default(),
            feature_deadlines: Deadlines::default(),
            decision: decision::State::default(),
            basis_decision: basis::State::default(),
            survival: survival::State::default(),
            pending_work: pending_work::Inventory::default(),
            pending_work_profile: None,
            deadline_work: Arc::new(std::sync::Mutex::new(
                deadline_recovery::AcceptedWork::default(),
            )),
            native_deadline: native_deadline::State::default(),
            coherent_pair: coherent_pair::State::default(),
        }
    }

    fn invalidate(&mut self) {
        self.generation = Arc::new(());
        self.total_dispatched = 0;
        self.total_dispatched_requests = 0;
        self.last_published = None;
        self.sample = None;
        self.low_load_ttfts.clear();
        self.low_load_visibility.clear();
        self.busy_visibility.clear();
        self.observed_delays.clear();
        self.pending_first_text.clear();
        self.backlog_peak_size = 0.0;
        self.pipeline_idle = true;
        self.survival.clear_started_history();
        self.reset_calibration("worker_invalidated");
        self.deadline_work
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .reset();
        self.native_deadline = native_deadline::State::default();
        self.calibration_profile = None;
        self.calibration_empty_polls = 0;
        self.backlog_generation = self.backlog_generation.wrapping_add(1);
    }

    fn reset_calibration(&mut self, reason: &'static str) {
        self.coherent_pair.revoke("coherent_calibration_reset");
        self.recovery.reset(reason);
        self.feature_deadlines = Deadlines::default();
        self.decision = decision::State::default();
        self.basis_decision = basis::State::default();
        self.survival.reset_calibration_preserving_started_history();
        self.pending_work.reset();
        self.calibration_baseline = (None, BaselineSource::Unknown);
        self.calibration_generation = Arc::new(());
        self.calibration_epoch = self.calibration_epoch.wrapping_add(1);
    }

    fn record_deadline_resolution(
        &mut self,
        resolved: DeadlineResolution,
        observed_at: Instant,
        now: Instant,
        source: &str,
    ) {
        if source == "poll" {
            if let Some(label) = resolved.tracking_id.and_then(|id| {
                self.decision.label(
                    id,
                    diagnostic_time(observed_at),
                    diagnostic_time(now),
                    false,
                    false,
                    Some(300.0),
                )
            }) {
                self.train_label(&label, false);
            }
        }
    }

    fn advance_feature_deadlines(&mut self, now: Instant) {
        for resolved in self.feature_deadlines.advance(now) {
            self.record_deadline_resolution(resolved, now, now, "poll");
        }
    }

    fn decision_snapshot(
        &self,
        config: &EstimatedWaitConfig,
        now: Instant,
    ) -> Option<decision::Snapshot> {
        let sample = self.sample.as_ref()?;
        let fresh = now.saturating_duration_since(sample.started).as_secs_f64()
            <= config.estimated_wait_max_snapshot_age_secs;
        let baseline = self.calibration_baseline.0?;
        let scalar = 1.0;
        let original = sample
            .prepared
            .queue_excess_seconds(self.total_dispatched.saturating_sub(sample.watermark));
        let live = self
            .recovery_forecast(config, now)
            .filter(|r| r.available)
            .map(|r| r.queue_extra);
        let x = live.unwrap_or(original);
        let context = self.decision.context(
            diagnostic_time(now),
            sample.prepared.timely_progress.filter(|_| fresh),
            sample
                .prepared
                .capacity_learned_ready
                .then_some(sample.prepared.progress_capacity)
                .flatten(),
            sample.prepared.has_backlog,
        );
        Some(decision::Snapshot {
            size_basis: self.basis_decision.stamp(sample.prepared.size_basis),
            at: diagnostic_time(now),
            profile: sample.prepared.decision_profile,
            baseline,
            x,
            source: if live.is_some() {
                "live_n"
            } else {
                "original_n_fallback"
            },
            context,
            original_queue_extra: original,
            live_queue_extra: live,
            scalar_gain: scalar,
            fresh,
            native_ready: sample.prepared.capacity_learned_ready,
            native_waiting: sample.prepared.native_waiting,
            poll_started: diagnostic_time(sample.started),
            eligible: fresh
                && sample.prepared.capacity_learned_ready
                && sample.prepared.native_waiting > 0.0
                && baseline.is_finite()
                && x.is_finite()
                && x >= 0.001,
        })
    }

    fn pending_work_admission(
        &self,
        config: &EstimatedWaitConfig,
        original: &decision::Admission,
        now: Instant,
    ) -> Option<pending_work::Admission> {
        let sample = self.sample.as_ref()?;
        let mut queue = sample.pending_work_queue?;
        let snapshot = original.snapshot;
        if queue.summary.generation != self.pending_work.generation() {
            queue.applied = false;
            queue.candidate_tokens = queue.original_tokens;
            queue.reason = "inventory_generation_changed";
        }
        if !snapshot.fresh || !snapshot.native_ready {
            queue.applied = false;
            queue.candidate_tokens = queue.original_tokens;
            queue.reason = if snapshot.fresh {
                "capacity_unready"
            } else {
                "stale_native_sample"
            };
        }
        let capacity = sample.prepared.native_prefill_capacity;
        let debt = (snapshot.original_queue_extra - queue.original_tokens / capacity).max(0.0);
        let available = snapshot.fresh && snapshot.native_ready;
        let recovery_cap_seconds = if available && snapshot.source == "live_n" {
            self.recovery_forecast(config, now)
                .filter(|r| r.available)
                .and_then(|r| r.population.zip(r.rate))
                .and_then(|(population, rate)| {
                    let cap = population / rate;
                    (rate > 0.0 && cap.is_finite() && cap >= 0.0).then_some(cap)
                })
        } else {
            None
        };
        let (control_x, recovery_fraction) =
            pending_work::candidate_x(queue, capacity, debt, snapshot.x, recovery_cap_seconds);
        // The source belongs to the same native poll as the queue and capacity.
        // Request-start capacity does not measure the pending full-prompt cohort.
        let capacity_progress_source = sample.pending_work_capacity_progress_source.clone();
        let selected_pending_applied =
            queue.applied && capacity_progress_source.as_deref() == Some("local_compute_tokens");
        let source_gate_reason = if queue.applied {
            match capacity_progress_source.as_deref() {
                Some("local_compute_tokens") => "local_compute_tokens",
                Some(_) => "capacity_progress_source_unsupported",
                None => "capacity_progress_source_missing",
            }
        } else {
            "pending_queue_fallback"
        };
        let candidate_x = if selected_pending_applied {
            control_x
        } else {
            snapshot.x
        };
        let candidate = decision::Snapshot {
            x: candidate_x,
            ..snapshot
        };
        Some(pending_work::Admission {
            method: pending_work::METHOD,
            current_inventory_generation: self.pending_work.generation(),
            queue,
            capacity_progress_source,
            selected_pending_applied,
            source_gate_reason,
            poll_started: diagnostic_time(sample.started),
            at: snapshot.at,
            debt_seconds: debt,
            capacity,
            recovery_fraction,
            recovery_cap_seconds,
            original_x: snapshot.x,
            control_x,
            candidate_x,
            original_raw_total: snapshot.baseline + snapshot.x,
            control_raw_total: snapshot.baseline + control_x,
            candidate_raw_total: snapshot.baseline + candidate_x,
            predictions: if selected_pending_applied {
                self.decision
                    .predictions(candidate, diagnostic_time(now), available)
            } else {
                original.predictions.clone()
            },
            available,
        })
    }

    fn train_label(&mut self, label: &decision::Label, terminal: bool) {
        let accepted = self.basis_decision.label(label, terminal);
        if accepted.is_some() {
            self.survival.label(label, terminal);
        }
    }

    fn recovery_forecast(
        &self,
        config: &EstimatedWaitConfig,
        now: Instant,
    ) -> Option<RecoveryForecast> {
        if !config.estimated_wait_calibration_shadow {
            return None;
        }
        let sample = self.sample.as_ref()?;
        let dispatched = self.total_dispatched.saturating_sub(sample.watermark);
        let queue = sample.prepared.queue_excess_seconds(0);
        let debt = (sample.prepared.queue_excess_seconds(dispatched) - queue).max(0.0);
        let gain = 1.0;
        Some(self.recovery.forecast(
            now,
            config.estimated_wait_max_snapshot_age_secs,
            queue,
            debt,
            self.calibration_baseline.0,
            gain,
        ))
    }

    fn update_calibration_baseline(&mut self, config: &EstimatedWaitConfig) {
        if !config.estimated_wait_calibration_shadow {
            return;
        }
        // A failed scrape removes the raw sample, not calibration evidence.
        // Re-evaluate provenance only when a usable load sample exists.
        let Some(sample) = self.sample.as_ref() else {
            return;
        };
        let measured = sample.prepared.baseline;
        let next = if let Some(value) = measured {
            (Some(value), BaselineSource::Measured)
        } else if let Some(value) = config.estimated_wait_provisional_baseline_secs {
            (Some(value), BaselineSource::Provisional)
        } else {
            (None, BaselineSource::Unknown)
        };
        let previous = self.calibration_baseline;
        let material_promotion = previous.1 == BaselineSource::Provisional
            && next.1 == BaselineSource::Measured
            && previous
                .0
                .zip(next.0)
                .is_some_and(|(old, new)| materially_different_baseline(old, new));
        let changed_fallback = previous.1 != BaselineSource::Unknown
            && next.1 != BaselineSource::Measured
            && previous != next;
        if material_promotion || changed_fallback {
            self.reset_calibration(if material_promotion {
                "baseline_promoted"
            } else {
                "baseline_fallback_changed"
            });
        }
        self.calibration_baseline = next;
    }

    fn calibration_forecast(
        &mut self,
        config: &EstimatedWaitConfig,
        now: Instant,
    ) -> Option<CalibrationForecast> {
        if !config.estimated_wait_calibration_shadow {
            return None;
        }
        self.update_calibration_baseline(config);
        let sample = self.sample.as_ref()?;
        let fresh = now.saturating_duration_since(sample.started).as_secs_f64()
            <= config.estimated_wait_max_snapshot_age_secs;
        Some(CalibrationForecast {
            generation: Arc::clone(&self.generation),
            calibration_generation: Arc::clone(&self.calibration_generation),
            #[cfg(test)]
            epoch: self.calibration_epoch,
            baseline: self.calibration_baseline.0,
            baseline_source: self.calibration_baseline.1,
            #[cfg(test)]
            raw_baseline: sample.prepared.baseline,
            queue_extra: sample
                .prepared
                .queue_excess_seconds(self.total_dispatched.saturating_sub(sample.watermark)),
            raw_extra: sample
                .prepared
                .excess_seconds(self.total_dispatched.saturating_sub(sample.watermark)),
            raw_fallback_work: survival::FrozenRawWork {
                poll_started: diagnostic_time(sample.started),
                source: sample.prepared.size_basis.map(|p| p.progress_source),
                key: sample.prepared.started_capacity_source_key,
                token_work: sample
                    .prepared
                    .estimate
                    .token_work(self.total_dispatched.saturating_sub(sample.watermark)),
                original_capacity: sample.prepared.effective_prefill_capacity,
                observed_delay_floor: sample.prepared.observed_delay_floor,
                measured_delay_only: sample.prepared.measured_delay_only(),
            },
            native_capacity_ready: sample.prepared.capacity_learned_ready,
            raw_ready: fresh && sample.prepared.capacity_ready,
            fresh,
            captured_at: now,
            decision_admission: None,
            basis_admission: None,
            survival_admission: None,
            pending_work_admission: None,
            deadline_dispatch: None,
            native_hard_ttft_timeout_secs: config.estimated_wait_native_hard_ttft_timeout_secs,
            native_deadline_capability_valid_until: self
                .native_deadline
                .deadline_valid_until(now, config.estimated_wait_max_snapshot_age_secs),
            prompt_work: pending_work::Estimate::default(),
            feature_tracking_supported: false,
        })
    }

    fn refresh(&mut self, worker: &Arc<dyn Worker>, config: &EstimatedWaitConfig) {
        let now = Instant::now();
        if config.estimated_wait_calibration_shadow {
            self.update_calibration_baseline(config);
        }
        if let Some(sample) = &mut self.sample {
            let excess = sample
                .prepared
                .excess_seconds(self.total_dispatched.saturating_sub(sample.watermark));
            sample.overloaded = excess.is_finite().then_some(excess >= sample.threshold);

            metrics::gauge!("smg_estimated_wait_snapshot_age_seconds", "worker" => worker.url().to_owned()).set(now.saturating_duration_since(sample.started).as_secs_f64());
            metrics::gauge!("smg_estimated_wait_capacity_ready", "worker" => worker.url().to_owned())
                .set(u8::from(sample.prepared.capacity_ready));
        }
        metrics::gauge!("smg_estimated_wait_data_usable", "worker" => worker.url().to_owned()).set(
            u8::from(self.sample.as_ref().is_some_and(|s| {
                s.overloaded.is_some()
                    && now.saturating_duration_since(s.started).as_secs_f64()
                        <= config.estimated_wait_max_snapshot_age_secs
            })),
        );
    }
}

#[derive(Debug, Default)]
pub(crate) struct EstimatedWaitAdmission {
    config: OnceLock<EstimatedWaitConfig>,
    // Registration/removal hooks maintain this even while monitoring is stopped.
    // Disabling the gateway flag must not disable a worker's own override.
    worker_overrides: AtomicUsize,
    entries: Mutex<HashMap<String, Entry>>,
}

impl EstimatedWaitAdmission {
    fn ttft_probe(&self, worker_url: &str) -> Option<TtftProbe> {
        if !self.enabled() {
            return None;
        }
        let mut entries = self.entries.lock();
        let entry = entries.get_mut(worker_url)?;
        let worker = entry.source.upgrade()?;
        let sample = entry.sample.as_ref()?;
        entry.next_observation_id = entry.next_observation_id.wrapping_add(1);
        let probe = TtftProbe {
            worker_url: worker_url.to_owned(),
            generation: Arc::clone(&entry.generation),
            dispatched: entry.total_dispatched,
            low_load: sample.prepared.low_load && worker.load() == 1,
            backlog_generation: entry.backlog_generation,
            backlogged: sample.prepared.has_backlog,
            baseline: ttft_median(&entry.low_load_ttfts),
            observation_id: entry.next_observation_id,
            pending_work: None,
            backlog_size: sample.prepared.backlog_size,
            calibration: None,
            request_id: None,
        };
        entry.pending_first_text.insert(probe.observation_id);
        Self::apply_observations(entry);
        entry.refresh(&worker, self.config());
        Some(probe)
    }

    fn register_feature_deadline(
        &self,
        worker: &str,
        request_id: Option<&str>,
        identity: &CalibrationForecast,
        started: Instant,
    ) -> Option<DeadlineHandle> {
        if !self.config().estimated_wait_feature_shadow {
            return None;
        }
        let mut entries = self.entries.lock();
        let entry = entries.get_mut(worker);
        let valid = entry.as_ref().is_some_and(|entry| {
            Arc::ptr_eq(&entry.generation, &identity.generation)
                && Arc::ptr_eq(
                    &entry.calibration_generation,
                    &identity.calibration_generation,
                )
        });
        if !valid {
            return None;
        }
        let entry = entry?;
        let handle = entry.feature_deadlines.register(started, request_id);
        let now = Instant::now();
        let tracking_id = match &handle {
            DeadlineHandle::Tracked(id, _) => Some(*id),
            DeadlineHandle::Untracked => None,
        };
        entry.decision.register(
            tracking_id,
            diagnostic_time(started),
            identity.decision_admission.as_ref().map(|a| a.snapshot),
        );
        entry.basis_decision.register(
            tracking_id,
            diagnostic_time(started),
            identity.decision_admission.as_ref().map(|a| a.snapshot),
        );
        entry.survival.register(
            tracking_id,
            diagnostic_time(started),
            identity.survival_admission.as_deref(),
        );
        let native_attempt_id = identity
            .deadline_dispatch
            .as_ref()
            .map(|lease| lease.native_attempt());
        entry.survival.native.register(
            native_attempt_id,
            diagnostic_time(now),
            identity.prompt_work,
        );
        Some(handle)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "outcome accounting requires worker identity and distinct admission/observation times"
    )]
    fn record_feature_outcome(
        &self,
        worker: &str,
        request_id: Option<&str>,
        identity: &CalibrationForecast,
        outcome: FeatureOutcome,
        deadline: Option<DeadlineHandle>,
        started: Instant,
        observed_at: Instant,
    ) {
        if !self.config().estimated_wait_feature_shadow {
            return;
        }
        let mut entries = self.entries.lock();
        let Some(entry) = entries.get_mut(worker) else {
            return;
        };
        if !Arc::ptr_eq(&entry.generation, &identity.generation)
            || !Arc::ptr_eq(
                &entry.calibration_generation,
                &identity.calibration_generation,
            )
        {
            return;
        }
        let now = Instant::now();
        let id = deadline.as_ref().and_then(|h| match h {
            DeadlineHandle::Tracked(id, _) => Some(*id),
            DeadlineHandle::Untracked => None,
        });
        if matches!(outcome, FeatureOutcome::FirstText) {
            entry.survival.native.first_text(
                identity
                    .deadline_dispatch
                    .as_ref()
                    .map(|lease| lease.native_attempt()),
                diagnostic_time(observed_at),
            );
        }
        if !matches!(outcome, FeatureOutcome::FirstText) {
            if let Some(lease) = identity
                .deadline_dispatch
                .as_ref()
                .filter(|lease| lease.before_send())
            {
                entry
                    .survival
                    .native
                    .local_pre_send_rollback(lease.native_attempt());
            }
        }
        let label = match outcome {
            FeatureOutcome::FirstText
            | FeatureOutcome::Timeout
            | FeatureOutcome::Disconnect
            | FeatureOutcome::NativeDeadline => id.and_then(|id| {
                entry.decision.label(
                    id,
                    diagnostic_time(observed_at),
                    diagnostic_time(now),
                    matches!(outcome, FeatureOutcome::FirstText),
                    true,
                    None,
                )
            }),
            _ => {
                entry.decision.terminal_unknown(id);
                None
            }
        };
        if let Some(label) = label {
            entry.train_label(&label, true);
        } else {
            entry.basis_decision.terminal_unknown(id);
            entry.survival.terminal_unknown(id);
        }

        if let Some(resolved) = deadline.and_then(|handle| {
            entry
                .feature_deadlines
                .resolve(handle, started, request_id, observed_at, outcome)
        }) {
            entry.record_deadline_resolution(resolved, observed_at, now, outcome.name());
        }
    }

    fn register_pending_work(&self, probe: &mut TtftProbe, forecast: Option<&CalibrationForecast>) {
        if !self.config().estimated_wait_feature_shadow || probe.pending_work.is_some() {
            return;
        }
        let mut entries = self.entries.lock();
        let Some(entry) = entries
            .get_mut(&probe.worker_url)
            .filter(|e| Arc::ptr_eq(&e.generation, &probe.generation))
        else {
            return;
        };
        let estimate = forecast
            .filter(|f| {
                Arc::ptr_eq(&entry.calibration_generation, &f.calibration_generation)
                    && f.feature_tracking_supported
            })
            .map_or_else(pending_work::Estimate::default, |f| f.prompt_work);
        let now = Instant::now();
        probe.pending_work = Some(
            entry
                .pending_work
                .register(probe.observation_id, now, estimate),
        );
    }

    fn resolve_pending_work(entry: &mut Entry, probe: &TtftProbe) {
        if let Some(handle) = probe.pending_work {
            entry.pending_work.remove(handle);
        }
    }

    fn finish_ttft_probe(&self, probe: &TtftProbe) {
        let mut entries = self.entries.lock();
        let Some(entry) = entries.get_mut(&probe.worker_url) else {
            return;
        };
        if !Arc::ptr_eq(&entry.generation, &probe.generation) {
            return;
        }
        if entry.pending_first_text.remove(&probe.observation_id) {
            Self::resolve_pending_work(entry, probe);
        }
        Self::apply_observations(entry);
        if let Some(worker) = entry.source.upgrade() {
            entry.refresh(&worker, self.config());
        }
    }

    fn observe_ttft(&self, probe: TtftProbe, ttft: f64, visibility: f64) {
        if !ttft.is_finite()
            || !visibility.is_finite()
            || ttft <= 0.0
            || visibility < 0.0
            || visibility > ttft
        {
            self.finish_ttft_probe(&probe);
            return;
        }
        let mut entries = self.entries.lock();
        let Some(entry) = entries.get_mut(&probe.worker_url) else {
            return;
        };
        if !Arc::ptr_eq(&entry.generation, &probe.generation) {
            return;
        }
        if !entry.pending_first_text.remove(&probe.observation_id) {
            return;
        }
        Self::resolve_pending_work(entry, &probe);
        if probe.calibration.is_some() {
            entry.update_calibration_baseline(self.config());
        }

        let Some(worker) = entry.source.upgrade() else {
            return;
        };
        let Some(sample) = entry.sample.as_ref() else {
            return;
        };
        if sample.started.elapsed().as_secs_f64()
            > self.config().estimated_wait_max_snapshot_age_secs
        {
            return;
        }
        if probe.low_load
            && entry.total_dispatched == probe.dispatched
            && worker.load() == 1
            && sample.prepared.low_load
        {
            push_ttft_sample(&mut entry.low_load_ttfts, ttft);
            push_ttft_sample(&mut entry.low_load_visibility, visibility);
        } else if probe.backlogged && probe.backlog_generation == entry.backlog_generation {
            push_ttft_sample(&mut entry.busy_visibility, visibility);
            // First-SSE-to-text delay can include engine queue time. Learn
            // total extra delay as an independent floor instead of subtracting
            // that gap or fitting a throughput to a changing token-work proxy.
            if let Some(baseline) = probe.baseline {
                let backlog = sample.prepared.backlog_size.max(probe.backlog_size);
                if backlog > 0.0 {
                    entry
                        .observed_delays
                        .push_back(((ttft - baseline).max(0.0), backlog));
                    while entry.observed_delays.len() > TTFT_WINDOW_SAMPLES {
                        entry.observed_delays.pop_front();
                    }
                }
            }
        }
        Self::apply_observations(entry);
        entry.refresh(&worker, self.config());
    }

    fn apply_observations(entry: &mut Entry) {
        let baseline = ttft_median(&entry.low_load_ttfts);
        let busy_visibility = observation_quantile(&entry.busy_visibility, 0.75);
        let low_visibility = ttft_median(&entry.low_load_visibility);
        let Some(sample) = entry.sample.as_mut() else {
            return;
        };
        let prepared = &mut sample.prepared;
        prepared.capacity_ready = prepared.capacity_learned_ready;
        // Bound pre-text stream telemetry by engine concurrency. This is a
        // diagnostic upper bound, not measured remaining prefill tokens.
        let pending = (entry.pending_first_text.len() as f64)
            .min(prepared.native_waiting + prepared.native_running);
        prepared.pending_engine = (pending - prepared.native_waiting).max(0.0);
        // A stream awaiting text may have partly completed prefill or may be
        // formatting reasoning. Keep its count diagnostic; never treat every
        // such stream as a full remaining prompt in the token-work numerator.
        prepared.backlog_size =
            prepared
                .native_waiting
                .max(if prepared.native_estimate.queued_work() > 0.0 {
                    1.0
                } else {
                    0.0
                });
        prepared.has_backlog =
            prepared.native_estimate.queued_work() > 0.0 || prepared.native_waiting > 0.0;
        if let Some(baseline) = baseline {
            prepared.estimate = prepared.estimate.with_base_overhead(baseline);
            prepared.baseline = Some(baseline);
            prepared.capacity_ready = prepared.capacity_learned_ready;
        }
        // Scale old observations downward as the queue drains. Never amplify
        // them when it grows: the native backlog term already models that work.
        let adjusted: VecDeque<_> = entry
            .observed_delays
            .iter()
            .map(|(delay, backlog)| delay * (prepared.backlog_size / backlog).min(1.0))
            .collect();
        let floor = prepared
            .has_backlog
            .then(|| observation_quantile(&adjusted, 0.75))
            .flatten();
        prepared.observed_delay_floor = floor.unwrap_or(0.0);
        prepared.observed_delay_available = floor.is_some();
        prepared.visibility_delay = if prepared.has_backlog {
            busy_visibility
                .zip(low_visibility)
                .map(|(busy, low)| (busy - low).max(0.0))
                .unwrap_or(0.0)
        } else {
            0.0
        };
        // Observed over-budget delay can establish overload without guessing a
        // cold processing rate. A small observed delay cannot validate that rate.
        prepared.capacity_ready |= floor.is_some_and(|delay| delay >= sample.threshold);
    }

    pub(crate) fn configure(&self, config: EstimatedWaitConfig) {
        let _ = self.config.set(config);
    }

    fn config(&self) -> &EstimatedWaitConfig {
        self.config.get_or_init(EstimatedWaitConfig::default)
    }

    pub(crate) fn is_shadow(&self) -> bool {
        self.config().estimated_wait_shadow
    }

    pub(crate) fn pending_work_enabled(&self) -> bool {
        self.enabled() && self.config().estimated_wait_feature_shadow
    }

    pub(crate) fn native_deadline_enabled(&self) -> bool {
        self.enabled() && self.config().estimated_wait_feature_shadow
    }

    pub(crate) fn native_deadline_timeout_configured(&self) -> bool {
        self.config()
            .estimated_wait_native_hard_ttft_timeout_secs
            .is_some()
    }

    /// A separate opt-in HTTP side channel cannot change raw load preparation,
    /// estimator readiness, or dispatch watermarks.
    pub(crate) fn publish_native_deadline(
        &self,
        worker: &Arc<dyn Worker>,
        native: Option<&WorkerLoadResponse>,
        watermark: Option<PollWatermark>,
        raw_started: Instant,
        native_started: Instant,
    ) {
        self.publish_native_deadline_observation(
            worker,
            native,
            watermark,
            raw_started,
            native_started,
            NativeDeadlineObservation {
                transport_timeout: false,
            },
        );
    }

    pub(crate) fn publish_native_deadline_timeout(
        &self,
        worker: &Arc<dyn Worker>,
        watermark: Option<PollWatermark>,
        raw_started: Instant,
        native_started: Instant,
    ) {
        self.publish_native_deadline_observation(
            worker,
            None,
            watermark,
            raw_started,
            native_started,
            NativeDeadlineObservation {
                transport_timeout: true,
            },
        );
    }

    fn publish_native_deadline_observation(
        &self,
        worker: &Arc<dyn Worker>,
        native: Option<&WorkerLoadResponse>,
        watermark: Option<PollWatermark>,
        raw_started: Instant,
        native_started: Instant,
        observation: NativeDeadlineObservation,
    ) {
        if !self.native_deadline_enabled() {
            return;
        }
        let Some(watermark) = watermark else {
            return;
        };
        let mut entries = self.entries.lock();
        let Some(entry) = entries.get_mut(worker.url()).filter(|entry| {
            entry.source.ptr_eq(&Arc::downgrade(worker))
                && Arc::ptr_eq(&entry.generation, &watermark.generation)
                && entry
                    .last_published
                    .is_some_and(|published| published >= raw_started)
        }) else {
            drop(entries);
            return;
        };
        let mut ledger = entry
            .deadline_work
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let received = Instant::now();
        let duplicate_facts_changed = entry
            .survival
            .native
            .candidate_duplicate_facts_changed(native);
        if observation.transport_timeout {
            entry.native_deadline.observe_transport_timeout(
                received,
                self.config().estimated_wait_max_snapshot_age_secs,
            );
        } else {
            entry.native_deadline.observe(
                native,
                worker.dp_rank(),
                native_started,
                received,
                self.config().estimated_wait_max_snapshot_age_secs,
                &mut ledger,
            );
        }
        drop(ledger);
        entry.survival.native.observe(
            native,
            entry
                .native_deadline
                .available(received, self.config().estimated_wait_max_snapshot_age_secs),
            diagnostic_time(native_started),
            diagnostic_time(received),
            diagnostic_time(raw_started),
            self.config().estimated_wait_max_snapshot_age_secs,
        );
        if worker.metadata().spec.runtime_type == RuntimeType::Sglang {
            // A newer native commit replaces or invalidates the pair, including
            // callbacks without a usable proof. Local ledger state stays live.
            entry.survival.native.commit_sg_raw(
                watermark.sg_raw_evidence,
                diagnostic_time(raw_started),
                entry.calibration_epoch,
                diagnostic_time(received),
            );
        }
        coherent_pair::native_publication(
            entry,
            watermark.coherent_raw,
            diagnostic_time(received),
            observation.transport_timeout,
            worker.metadata().spec.runtime_type == RuntimeType::Sglang,
            duplicate_facts_changed,
        );
    }

    pub(crate) fn enabled(&self) -> bool {
        self.config
            .get()
            .is_some_and(|c| c.max_estimated_wait_secs.is_some())
            || self.worker_overrides.load(Ordering::Acquire) > 0
    }

    pub(crate) fn needs_load(&self, worker: &Arc<dyn Worker>) -> bool {
        worker.metadata().overload.max_estimated_wait_secs.is_some()
            || self
                .config
                .get()
                .is_some_and(|c| c.max_estimated_wait_secs.is_some())
    }

    /// Called under the registry's per-worker mutation lock, before publication.
    pub(crate) fn worker_added(&self, worker: &Arc<dyn Worker>) {
        if !self.needs_load(worker) {
            return;
        }
        // Only registration can install a different incarnation for a URL.
        if self
            .entries
            .lock()
            .insert(worker.url().to_owned(), Entry::new(worker))
            .is_some()
        {
            invalidate_diagnostic_metrics(worker.url());
        }
        if worker.metadata().overload.max_estimated_wait_secs.is_some() {
            self.worker_overrides.fetch_add(1, Ordering::Release);
        }
    }

    pub(crate) fn worker_removed(&self, worker: &Arc<dyn Worker>) {
        if !self.needs_load(worker) {
            return;
        }
        let mut entries = self.entries.lock();
        if entries
            .get(worker.url())
            .is_some_and(|entry| entry.source.ptr_eq(&Arc::downgrade(worker)))
        {
            entries.remove(worker.url());
            invalidate_diagnostic_metrics(worker.url());
        }
        if worker.metadata().overload.max_estimated_wait_secs.is_some() {
            self.worker_overrides.fetch_sub(1, Ordering::Release);
        }
    }

    fn entry<'a>(
        entries: &'a mut HashMap<String, Entry>,
        worker: &Arc<dyn Worker>,
    ) -> Option<&'a mut Entry> {
        let entry = entries
            .entry(worker.url().to_owned())
            .or_insert_with(|| Entry::new(worker));
        entry
            .source
            .ptr_eq(&Arc::downgrade(worker))
            .then_some(entry)
    }

    pub(crate) fn poll_started(&self, worker: &Arc<dyn Worker>) -> Option<PollWatermark> {
        if !self.needs_load(worker) {
            return None;
        }
        if tracing::enabled!(target: "smg::estimated_wait::observation", tracing::Level::DEBUG) {
            diagnostic_time(Instant::now());
        }
        let mut entries = self.entries.lock();
        let entry = Self::entry(&mut entries, worker)?;
        Some(PollWatermark {
            generation: Arc::clone(&entry.generation),
            dispatched: entry.total_dispatched,
            dispatched_requests: entry.total_dispatched_requests,
            pending_work: entry.pending_work.summary(Instant::now()),
            sg_raw_evidence: None,
            coherent_raw: None,
        })
    }

    #[cfg(test)]
    pub(crate) fn publish(
        &self,
        worker: &Arc<dyn Worker>,
        load: Option<&WorkerLoadResponse>,
        watermark: Option<PollWatermark>,
        started: Instant,
    ) {
        let _ = self.publish_with_sg_raw(worker, load, watermark, started);
    }

    pub(crate) fn publish_with_sg_raw(
        &self,
        worker: &Arc<dyn Worker>,
        load: Option<&WorkerLoadResponse>,
        watermark: Option<PollWatermark>,
        started: Instant,
    ) -> Option<SgRawEvidence> {
        let Some(watermark) = watermark else {
            return None;
        };
        // Rank aggregation, validation and the queue/KV formula are report work,
        // outside the admission transaction lock.
        let config = self.config();
        let prepared = load.and_then(|load| config.prepare(load));
        let threshold = config.threshold(worker);
        let mut entries = self.entries.lock();
        let Some(entry) = entries.get_mut(worker.url()) else {
            return None;
        };
        let outcome = poll_outcome(
            Arc::ptr_eq(&entry.generation, &watermark.generation),
            entry.last_published,
            started,
            Instant::now(),
            load.is_some(),
            prepared.is_some(),
            config.estimated_wait_max_snapshot_age_secs,
        );
        if !Arc::ptr_eq(&entry.generation, &watermark.generation)
            || entry.last_published.is_some_and(|last| last >= started)
        {
            // Superseded/duplicate publications preserve the existing state.
            // Only admitted later negatives may invalidate the SG pair.

            return None;
        }
        let new_poll = entry.last_published.is_none_or(|last| started > last);
        if config.estimated_wait_feature_shadow {
            entry.advance_feature_deadlines(Instant::now());
        }
        entry.last_published = Some(started);
        // Native capacity samples only shrink when the monitor resets its
        // counter/profile history. Do not let pre-reset responses train the
        // new profile, and leave all raw estimator state untouched.
        if let Some(new) = prepared.filter(|_| config.estimated_wait_calibration_shadow && new_poll)
        {
            if entry
                .calibration_profile
                .is_some_and(|(old_ready, old_count)| {
                    (old_ready && !new.capacity_learned_ready)
                        || old_count
                            .zip(new.capacity_profile_sample_count)
                            .is_some_and(|(old, new)| new < old)
                })
            {
                entry.reset_calibration("native_profile_changed");
            }
            // A failed scrape invalidates the raw report, not this reset
            // watermark. Otherwise a counter reset across one failed poll
            // could silently retain the preceding calibration history.
            entry.calibration_profile = Some((
                new.capacity_learned_ready,
                new.capacity_profile_sample_count,
            ));
        }
        if config.estimated_wait_calibration_shadow && new_poll {
            if outcome == "success" {
                if let Some(load) = load {
                    entry.recovery.observe(
                        load,
                        started,
                        config.estimated_wait_max_snapshot_age_secs,
                        watermark.dispatched_requests,
                    );
                }
            } else {
                entry.recovery.reset(outcome);
            }
        }
        if config.estimated_wait_calibration_shadow && new_poll {
            // Two consecutive fresh queue-empty reports, with no local pre-text
            // observers or admissions during the poll, delimit a new workload.
            // Decoding may continue; this confirms queue drain, not engine idle.
            let drained = prepared.is_some_and(|p| !p.has_backlog)
                && entry.pending_first_text.is_empty()
                && entry.total_dispatched == watermark.dispatched
                && started.elapsed().as_secs_f64() <= config.estimated_wait_max_snapshot_age_secs;
            if drained {
                if entry.calibration_empty_polls < 2 {
                    entry.calibration_empty_polls += 1;
                    if entry.calibration_empty_polls == 2 {
                        entry.reset_calibration("queue_drained");
                    }
                }
            } else {
                entry.calibration_empty_polls = 0;
            }
        }

        // A verified idle worker begins a new queue epoch. Responses from the
        // preceding busy epoch cannot recalibrate a later workload after drain.
        if prepared.is_some_and(|prepared| prepared.low_load) && entry.pending_first_text.len() <= 1
        {
            if !entry.pipeline_idle {
                entry.backlog_generation = entry.backlog_generation.wrapping_add(1);
            }
            entry.pipeline_idle = true;
            entry.backlog_peak_size = 0.0;
            entry.observed_delays.clear();
            entry.busy_visibility.clear();
        } else {
            entry.pipeline_idle = false;
        }
        if let Some(prepared) = prepared {
            entry.backlog_peak_size = entry.backlog_peak_size.max(prepared.backlog_size);
            if let Some(delay) = prepared.native_queue_delay.filter(|_| prepared.has_backlog) {
                entry
                    .observed_delays
                    .push_back((delay, entry.backlog_peak_size.max(1.0)));
                while entry.observed_delays.len() > TTFT_WINDOW_SAMPLES {
                    entry.observed_delays.pop_front();
                }
            }
        }
        if config.estimated_wait_feature_shadow {
            if let Some(new) = prepared {
                if entry
                    .pending_work_profile
                    .is_some_and(|old| old != new.decision_profile)
                {
                    entry.pending_work.reset();
                }
                entry.pending_work_profile = Some(new.decision_profile);
            }
        }
        entry.sample = prepared.zip(threshold).map(|(prepared, threshold)| Sample {
            prepared,
            started,
            watermark: watermark.dispatched,
            threshold,
            overloaded: None,
            pending_work_capacity_progress_source: load
                .filter(|l| l.loads.len() == 1)
                .and_then(|l| l.loads[0].timely_prefill_progress_source.clone()),
            pending_work_queue: config.estimated_wait_feature_shadow.then(|| {
                let native_tokens = load.is_some_and(|l| {
                    l.loads
                        .iter()
                        .all(|r| r.num_waiting_uncached_tokens_available != Some(false))
                });
                let eligible = load.is_some_and(|l| {
                    l.loads.len() == 1
                        && l.dp_rank_count == 1
                        && l.loads[0].dp_rank == 0
                        && l.loads[0].native_request_counts_available == Some(true)
                        && l.loads[0].native_request_counts_profile.is_some()
                        && l.loads[0].num_running_reqs >= 0
                });
                let same_generation =
                    watermark.pending_work.generation == entry.pending_work.generation();
                let mut queue = pending_work::Queue::new(
                    watermark.pending_work,
                    prepared.native_estimate.queued_work(),
                    prepared.native_waiting,
                    prepared.native_running,
                    prepared.blended_prompt_size,
                    eligible && same_generation,
                    native_tokens,
                );
                if native_tokens
                    && load.is_some_and(|l| {
                        l.loads
                            .iter()
                            .any(|r| r.num_waiting_uncached_tokens_available.is_none())
                    })
                {
                    // Preserve the legacy numeric path without claiming exact
                    // native-token provenance when availability is unspecified.
                    queue.reason = "native_token_availability_unknown";
                } else if !same_generation && !native_tokens {
                    queue.reason = "inventory_generation_changed";
                }
                queue
            }),
        });
        Self::apply_observations(entry);
        entry.refresh(worker, config);
        if config.estimated_wait_feature_shadow {
            let now = Instant::now();
            let profile = entry.sample.as_ref().map(|s| s.prepared.decision_profile);
            let fresh = outcome == "success"
                && entry.sample.as_ref().is_some_and(|s| {
                    now.saturating_duration_since(s.started).as_secs_f64()
                        <= config.estimated_wait_max_snapshot_age_secs
                });

            entry.decision.track_poll(
                diagnostic_time(now),
                profile,
                fresh,
                config.estimated_wait_max_snapshot_age_secs,
            );
            entry.basis_decision.poll(
                diagnostic_time(now),
                profile,
                entry.sample.as_ref().and_then(|s| s.prepared.size_basis),
                fresh,
                config.estimated_wait_max_snapshot_age_secs,
            );
            entry.survival.poll(
                diagnostic_time(now),
                profile,
                entry.basis_decision.epoch,
                fresh
                    && entry
                        .sample
                        .as_ref()
                        .is_some_and(|s| s.prepared.size_basis.is_some()),
                config.estimated_wait_max_snapshot_age_secs,
            );
            entry
                .survival
                .native
                .raw_poll(load, diagnostic_time(started), outcome == "success");
        }
        // Freeze the successful raw evidence in the same transaction as its
        // publication. The bounded native task carries this exact proof; its
        // callback must never reconstruct a rate from a newer raw sample.
        let sg_proof = if config.estimated_wait_feature_shadow
            && worker.metadata().spec.runtime_type == RuntimeType::Sglang
            && outcome == "success"
        {
            let prepared = entry.sample.as_ref().map(|sample| sample.prepared);
            entry.survival.native.sg_raw_evidence(
                diagnostic_time(started),
                config.estimated_wait_max_snapshot_age_secs,
                entry.calibration_epoch,
                prepared.and_then(|p| p.size_basis),
                prepared.map(|p| p.effective_prefill_capacity),
                prepared.map(|p| p.native_waiting),
                config.estimated_wait_fallback_prefill_throughput,
            )
        } else {
            None
        };
        if config.estimated_wait_feature_shadow {
            coherent_pair::raw_publication(
                entry,
                config,
                load,
                &watermark,
                started,
                outcome == "success",
                sg_proof.clone(),
            );
        }
        sg_proof
    }

    pub(crate) fn coherent_raw_context(
        &self,
        worker: &Arc<dyn Worker>,
        started: Instant,
    ) -> Option<coherent_pair::RawContext> {
        let entries = self.entries.lock();
        let entry = entries
            .get(worker.url())
            .filter(|e| e.source.ptr_eq(&Arc::downgrade(worker)))?;
        coherent_pair::issued(entry, started)
    }

    pub(crate) fn evict(&self, worker: &Arc<dyn Worker>) {
        if !self.needs_load(worker) {
            return;
        }
        let mut entries = self.entries.lock();
        if entries
            .get(worker.url())
            .is_some_and(|entry| entry.source.ptr_eq(&Arc::downgrade(worker)))
        {
            if let Some(entry) = entries.get_mut(worker.url()) {
                entry.invalidate();
            }
            invalidate_diagnostic_metrics(worker.url());
        }
    }

    pub(crate) fn clear(&self) {
        let mut entries = self.entries.lock();
        for (url, entry) in entries.iter_mut() {
            entry.invalidate();
            invalidate_diagnostic_metrics(url);
        }
    }

    /// Keep admission, selection and final credit atomic. Disabled fleets take
    /// no lock. Report aggregation and validation are never done under this guard.
    pub(crate) fn begin(&self) -> Option<AdmissionGuard<'_>> {
        if !self.enabled() {
            return None;
        }
        Some(AdmissionGuard {
            config: self.config(),
            entries: self.entries.lock(),
        })
    }
}

#[derive(Clone)]
pub(crate) struct PollWatermark {
    generation: Arc<()>,
    dispatched: u64,
    dispatched_requests: u64,
    pending_work: pending_work::Summary,
    pub(crate) sg_raw_evidence: Option<SgRawEvidence>,
    pub(crate) coherent_raw: Option<coherent_pair::RawContext>,
}

pub(crate) struct AdmissionGuard<'a> {
    config: &'a EstimatedWaitConfig,
    entries: MutexGuard<'a, HashMap<String, Entry>>,
}

impl AdmissionGuard<'_> {
    pub(crate) fn feature_scope(
        candidates: &[Arc<dyn Worker>],
        enabled_path: bool,
    ) -> Option<&Arc<dyn Worker>> {
        // Scope is a single-worker pool, not a transient availability snapshot:
        // health changes must not attribute a cohort rejection to one worker.
        if !enabled_path || candidates.len() != 1 {
            return None;
        }
        (candidates[0].is_healthy() && candidates[0].circuit_breaker_can_execute())
            .then_some(&candidates[0])
    }

    pub(crate) fn attach_feature_context(
        &mut self,
        worker: &Arc<dyn Worker>,
        forecast: &mut Option<CalibrationForecast>,
    ) -> bool {
        if !self.config.estimated_wait_feature_shadow {
            return false;
        }
        if let (Some(entry), Some(forecast)) = (
            self.entries
                .get_mut(worker.url())
                .filter(|e| e.source.ptr_eq(&Arc::downgrade(worker))),
            forecast.as_mut(),
        ) {
            forecast.decision_admission = entry
                .decision_snapshot(self.config, forecast.captured_at)
                .map(|snapshot| {
                    Box::new(decision::Admission {
                        snapshot,
                        predictions: entry.decision.predictions(
                            snapshot,
                            diagnostic_time(forecast.captured_at),
                            forecast.fresh && forecast.native_capacity_ready,
                        ),
                    })
                });
            forecast.pending_work_admission = forecast
                .decision_admission
                .as_ref()
                .and_then(|original| {
                    entry.pending_work_admission(self.config, original, forecast.captured_at)
                })
                .map(Box::new);
            forecast.basis_admission =
                forecast.decision_admission.as_ref().map(|original| {
                    let control = forecast.pending_work_admission.as_ref().map(|pending| {
                        decision::Admission {
                            snapshot: decision::Snapshot {
                                x: pending.candidate_x,
                                ..original.snapshot
                            },
                            predictions: pending.predictions.clone(),
                        }
                    });
                    Box::new(entry.basis_decision.admission(
                        original,
                        control.as_ref().unwrap_or(original),
                        diagnostic_time(forecast.captured_at),
                        forecast.fresh && forecast.native_capacity_ready,
                    ))
                });
            forecast.survival_admission = forecast.basis_admission.as_ref().map(|basis| {
                let total = forecast.seconds();
                let scalar = (forecast.baseline.is_some()
                    && forecast.fresh
                    && total.is_finite()
                    && total >= 0.0)
                    .then_some(total);
                let mut admission = entry.survival.admission(
                    basis,
                    forecast.pending_work_admission.as_deref(),
                    scalar,
                );
                admission.select_raw_fallback(forecast.raw_total_for_fallback());
                entry.survival.native.rebase_selected_raw_fallback(
                    &mut admission,
                    forecast.raw_fallback_work,
                    self.config.estimated_wait_fallback_prefill_throughput,
                );
                if worker.metadata().spec.runtime_type == RuntimeType::Sglang {
                    // This optional output preserves the original projection
                    // and its training/expiry bookkeeping as the control.
                    entry
                        .survival
                        .native
                        .select_sg_raw_basis(&mut admission, entry.calibration_epoch);
                }
                admission.paired_native_candidate = Some(coherent_pair::select(
                    entry,
                    self.config,
                    forecast.captured_at,
                    &admission,
                    worker.metadata().spec.runtime_type == RuntimeType::Sglang,
                ));
                Box::new(admission)
            });
            forecast.feature_tracking_supported = true;
            true
        } else {
            false
        }
    }

    pub(crate) fn record_feature_decision(
        &mut self,
        worker: &Arc<dyn Worker>,
        rejected: bool,
        outcome_tracking_supported: bool,
    ) {
        if !self.config.estimated_wait_feature_shadow {
            return;
        }
        let Some(entry) = self
            .entries
            .get_mut(worker.url())
            .filter(|e| e.source.ptr_eq(&Arc::downgrade(worker)))
        else {
            return;
        };
        if !rejected && !outcome_tracking_supported {
            entry.decision.mark_incomplete();
        }
    }

    pub(crate) fn threshold(&self, worker: &Arc<dyn Worker>) -> Option<f64> {
        self.config.threshold(worker)
    }

    pub(crate) fn capture_calibration(
        &mut self,
        worker: &Arc<dyn Worker>,
    ) -> Option<CalibrationForecast> {
        let forecast = self
            .entries
            .get_mut(worker.url())
            .filter(|e| e.source.ptr_eq(&Arc::downgrade(worker)))?
            .calibration_forecast(self.config, Instant::now())?;

        Some(forecast)
    }

    /// Separate accepted-work shadow. Keep raw dispatch watermarks untouched.
    /// Only the regular HTTP feature-observation path owns the lifecycle lease.
    pub(crate) fn register_deadline_dispatch(
        &self,
        worker: &Arc<dyn Worker>,
        forecast: &mut Option<CalibrationForecast>,
    ) {
        if !self.config.estimated_wait_feature_shadow {
            return;
        }
        let (Some(entry), Some(forecast)) = (
            self.entries
                .get(worker.url())
                .filter(|e| e.source.ptr_eq(&Arc::downgrade(worker))),
            forecast.as_mut().filter(|f| f.feature_tracking_supported),
        ) else {
            return;
        };
        if forecast.deadline_dispatch.is_none() {
            forecast.deadline_dispatch = deadline_recovery::DispatchLease::with_native_attempt(
                Arc::clone(&entry.deadline_work),
                Instant::now(),
                format!(
                    "{}.{}",
                    NATIVE_GATEWAY_EPOCH.get_or_init(uuid::Uuid::now_v7),
                    uuid::Uuid::now_v7()
                ),
            );
        }
    }

    pub(crate) fn check(
        &self,
        candidates: &[Arc<dyn Worker>],
        model: &str,
    ) -> Result<(), Response> {
        if self.check_cohort(candidates, model) {
            return Err(Self::shed(model));
        }
        Ok(())
    }

    pub(crate) fn shed(model: &str) -> Response {
        metrics::counter!("smg_estimated_wait_rejections_total", "model" => model.to_owned())
            .increment(1);
        overload::shed_estimated_wait(model)
    }

    /// Probe a cohort without recording an enforced rejection or building a response.
    /// Shadow mode still counts would-reject pool checks and permits selection.
    pub(crate) fn check_cohort(&self, candidates: &[Arc<dyn Worker>], model: &str) -> bool {
        let mut eligible = false;
        for worker in candidates
            .iter()
            .filter(|w| w.is_healthy() && w.circuit_breaker_can_execute())
        {
            eligible = true;
            // One permissive worker is sufficient. In particular, an unprotected
            // worker or unknown report must not allow its peers to prove saturation.
            if self.config.threshold(worker).is_none() {
                return false;
            }
            let entry = self
                .entries
                .get(worker.url())
                .filter(|e| e.source.ptr_eq(&Arc::downgrade(worker)));
            let sample = entry.and_then(|entry| entry.sample.as_ref());
            let verdict =
                match sample {
                    Some(sample)
                        if sample.started.elapsed().as_secs_f64()
                            > self.config.estimated_wait_max_snapshot_age_secs =>
                    {
                        Err("stale")
                    }
                    Some(sample) if !sample.prepared.capacity_ready => Err("capacity_unready"),
                    Some(sample) => sample
                        .prepared
                        .baseline
                        .or(self.config.estimated_wait_provisional_baseline_secs)
                        .map(|baseline| {
                            baseline
                                + sample.prepared.excess_seconds(entry.map_or(0, |e| {
                                    e.total_dispatched.saturating_sub(sample.watermark)
                                }))
                        })
                        .filter(|total| total.is_finite() && *total >= 0.0)
                        .map(|total| total >= sample.threshold)
                        .ok_or("baseline_unavailable"),
                    None => Err(if entry.is_some_and(|e| e.last_published.is_some()) {
                        "unusable"
                    } else {
                        "missing"
                    }),
                };
            match verdict {
                Ok(true) => {}
                Ok(false) => return false,
                Err(reason) => {
                    metrics::counter!("smg_estimated_wait_unknown_total", "model" => model.to_owned(), "reason" => reason).increment(1);
                    if reason != "capacity_unready" {
                        metrics::gauge!("smg_estimated_wait_data_usable", "worker" => worker.url().to_owned()).set(0.0);
                    }
                    return false;
                }
            }
        }
        if !eligible {
            return false;
        }
        if self.config.estimated_wait_shadow {
            metrics::counter!("smg_estimated_wait_shadow_rejections_total", "model" => model.to_owned())
                .increment(1);
            return false;
        }
        true
    }

    pub(crate) fn credit(&mut self, worker: &Arc<dyn Worker>, tokens: Option<&[u32]>) {
        if self.config.threshold(worker).is_none() {
            return;
        }
        let count = tokens.map_or_else(
            || u64::from(self.config.estimated_wait_mean_prefill_tokens),
            |t| t.len() as u64,
        );
        let Some(entry) = EstimatedWaitAdmission::entry(&mut self.entries, worker) else {
            return;
        };
        entry.total_dispatched = entry.total_dispatched.saturating_add(count);
        entry.total_dispatched_requests = entry.total_dispatched_requests.saturating_add(1);
        entry.refresh(worker, self.config);
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Barrier, time::Duration};

    use axum::http::{header::RETRY_AFTER, StatusCode};
    use openai_protocol::worker::{SchedulerLoadSnapshot, WorkerStatus};

    use super::*;
    use crate::{
        config::PolicyConfig,
        policies::PolicyRegistry,
        routers::{
            common::{
                placement::{self, PlacementFailure, PlacementInputs},
                retry::is_retryable_response,
            },
            error,
        },
        worker::{
            monitor::WorkerMonitor, BasicWorkerBuilder, ConnectionMode, WorkerRegistry, WorkerType,
        },
    };

    fn recovery_registry(shadow: bool) -> (Arc<WorkerRegistry>, Arc<dyn Worker>) {
        let registry = Arc::new(WorkerRegistry::new());
        registry.estimated_wait().configure(EstimatedWaitConfig {
            max_estimated_wait_secs: Some(300.0),
            estimated_wait_max_snapshot_age_secs: 60.0,
            estimated_wait_calibration_shadow: true,
            estimated_wait_shadow: shadow,
            ..config()
        });
        let worker = worker("http://recovery:1");
        registry.register(worker.clone()).unwrap();
        let start = Instant::now() - Duration::from_secs(30);
        for i in 0..7 {
            let mut report = load(Some(100_000), 100 - i * 10, 100.0, 0.0);
            report.loads[0].num_running_reqs = 10;
            report.loads[0].native_request_counts_available = Some(true);
            report.loads[0].low_load_ttft_seconds = Some(2.0);
            let stamp = registry.estimated_wait().poll_started(&worker);
            registry.estimated_wait().publish(
                &worker,
                Some(&report),
                stamp,
                start + Duration::from_secs(i as u64 * 5),
            );
        }
        (registry, worker)
    }

    #[test]
    fn live_recovery_keeps_original_fit_and_gate_and_preserves_post_snapshot_debt() {
        let (registry, worker) = recovery_registry(false);
        let admission = registry.estimated_wait();
        {
            let entry = &admission.entries.lock()[worker.url()];

            let f = entry
                .recovery_forecast(admission.config(), Instant::now())
                .unwrap();
            assert!(f.available);
            assert_eq!(f.queue_extra, 25.0);
            assert_eq!(
                entry
                    .sample
                    .as_ref()
                    .unwrap()
                    .prepared
                    .queue_excess_seconds(0),
                1000.0
            );
        }
        let mut guard = admission.begin().unwrap();
        assert!(guard.check(std::slice::from_ref(&worker), "m").is_err());
        guard.credit(&worker, Some(&[0; 100]));
        let entry = &guard.entries[worker.url()];
        let f = entry
            .recovery_forecast(admission.config(), Instant::now())
            .unwrap();
        assert_eq!(f.queue_extra, 26.0);
        assert_eq!(f.debt, 1.0);
    }

    #[test]
    fn recovery_arrival_credit_is_one_request_and_frozen_at_poll_start() {
        for credit_before_start in [false, true] {
            let (registry, worker) = recovery_registry(false);
            let admission = registry.estimated_wait();
            if credit_before_start {
                admission.begin().unwrap().credit(&worker, Some(&[0; 100]));
            }
            let stamp = admission.poll_started(&worker);
            if !credit_before_start {
                admission.begin().unwrap().credit(&worker, Some(&[0; 100]));
            }
            let mut report = load(Some(100_000), 41, 100.0, 0.0);
            report.loads[0].num_running_reqs = 10;
            report.loads[0].native_request_counts_available = Some(true);
            report.loads[0].low_load_ttft_seconds = Some(2.0);
            let at = admission.entries.lock()[worker.url()]
                .last_published
                .unwrap()
                + Duration::from_secs(5);
            admission.publish(&worker, Some(&report), stamp, at);
            let entries = admission.entries.lock();
            let entry = &entries[worker.url()];
            assert_eq!(entry.total_dispatched_requests, 1);
            assert_eq!(entry.total_dispatched, 100);
            let f = entry.recovery_forecast(admission.config(), at).unwrap();
            assert_eq!(f.available, credit_before_start);
            assert_eq!(
                f.poll_request_watermark,
                Some(u64::from(credit_before_start))
            );
            assert_eq!(f.debt, if credit_before_start { 0.0 } else { 1.0 });
        }
    }

    #[test]
    fn recovery_request_counter_saturates_and_worker_invalidation_resets_it() {
        let (registry, worker) = recovery_registry(false);
        let admission = registry.estimated_wait();
        admission
            .entries
            .lock()
            .get_mut(worker.url())
            .unwrap()
            .total_dispatched_requests = u64::MAX;
        admission.begin().unwrap().credit(&worker, Some(&[]));
        let mut entries = admission.entries.lock();
        let entry = entries.get_mut(worker.url()).unwrap();
        assert_eq!(entry.total_dispatched_requests, u64::MAX);
        entry.invalidate();
        assert_eq!(entry.total_dispatched_requests, 0);
    }

    #[test]
    fn live_recovery_failed_poll_resets_but_duplicates_and_older_reports_do_not_mix_inputs() {
        let (registry, worker) = recovery_registry(false);
        let admission = registry.estimated_wait();
        let last = admission.entries.lock()[worker.url()]
            .last_published
            .unwrap();
        for at in [last, last - Duration::from_secs(1)] {
            let stamp = admission.poll_started(&worker);
            admission.publish(&worker, Some(&load(Some(1), 1, 100.0, 0.0)), stamp, at);
            let entries = admission.entries.lock();
            let entry = &entries[worker.url()];
            assert_eq!(
                entry
                    .sample
                    .as_ref()
                    .unwrap()
                    .prepared
                    .queue_excess_seconds(0),
                1000.0
            );
            assert!(
                entry
                    .recovery_forecast(admission.config(), Instant::now())
                    .unwrap()
                    .available
            );
        }
        let stamp = admission.poll_started(&worker);
        admission.publish(&worker, None, stamp, Instant::now());
        let entries = admission.entries.lock();
        let entry = &entries[worker.url()];
        assert!(entry.sample.is_none());
        assert_eq!(entry.recovery.reason, "failed");
    }

    #[test]
    fn diagnostic_poll_classification_and_monotonic_precision() {
        let started = Instant::now();
        let now = started + Duration::from_millis(100);
        let outcome = |generation, previous, has_load, prepared, age| {
            poll_outcome(generation, previous, started, now, has_load, prepared, age)
        };
        assert_eq!(outcome(true, None, true, true, 1.0), "success");
        assert_eq!(outcome(true, None, false, false, 1.0), "failed");
        assert_eq!(outcome(true, None, true, false, 1.0), "invalid_load");
        assert_eq!(outcome(true, None, true, true, 0.05), "stale");
        assert_eq!(outcome(true, Some(started), true, true, 1.0), "duplicate");
        assert_eq!(outcome(true, Some(now), true, true, 1.0), "out_of_order");
        assert_eq!(outcome(false, None, true, true, 1.0), "generation_changed");
        let delta = diagnostic_time(now) - diagnostic_time(started);
        assert!((delta - 0.1).abs() < 1e-8);
    }

    #[test]
    fn metric_outcome_survives_missing_learning_probe_and_metadata_events() {
        use crate::observability::ttft::{Mode, Observation};
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let registry = Arc::new(WorkerRegistry::new());
            let start = Instant::now();
            let observation = Observation::new("test", Mode::Shadow, start);
            observation.freeze(None, Some(300.0), "vllm");
            let mut observer = StreamTtftObserver::new(&registry, "http://missing", start)
                .with_metric_observation(Some(observation));
            observer.push(b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n");
            assert!(!handle.render().contains("smg_ttft_outcomes_total"));
            observer
                .push(b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hello\"}}]}\n\n");
            observer.downstream_disconnected();
            drop(observer);
            let metrics = handle.render();
            assert!(metrics.contains("outcome=\"exact\"} 1"), "{metrics}");
            assert!(!metrics.contains("outcome=\"unknown\""), "{metrics}");
            assert!(
                !metrics.contains("smg_ttft_classifications_total"),
                "{metrics}"
            );
        });
    }

    #[test]
    fn lost_observation_is_unknown_even_after_the_budget() {
        use crate::observability::ttft::{Mode, Observation, Prediction};
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let registry = Arc::new(WorkerRegistry::new());
            let start = Instant::now() - Duration::from_secs(301);
            let observation = Observation::new("test", Mode::Shadow, start);
            observation.freeze(
                Some(Prediction {
                    total: 400.0,
                    evidence_valid: true,
                }),
                Some(300.0),
                "vllm",
            );
            let mut observer = StreamTtftObserver::new(&registry, "http://missing", start)
                .with_metric_observation(Some(observation));
            observer.cancel();
            drop(observer);
            let metrics = handle.render();
            assert!(metrics.contains("outcome=\"unknown\"} 1"), "{metrics}");
            assert!(
                !metrics.contains("smg_ttft_classifications_total"),
                "{metrics}"
            );
        });
    }

    fn config() -> EstimatedWaitConfig {
        EstimatedWaitConfig {
            max_estimated_wait_secs: Some(2.0),
            estimated_wait_shadow: false,
            estimated_wait_kv_pressure_weight: 0.0,
            estimated_wait_mean_prefill_tokens: 100,
            estimated_wait_fallback_prefill_throughput: 100.0,
            // Preserve the original arithmetic in tests that exercise other
            // dimensions; the correction has dedicated coverage below.
            estimated_wait_dispatch_blocking_factor: 1.0,
            ..Default::default()
        }
    }

    fn worker(url: &str) -> Arc<dyn Worker> {
        Arc::new(
            BasicWorkerBuilder::new(url)
                .status(WorkerStatus::Ready)
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Http)
                .build(),
        )
    }

    fn load(tokens: Option<i32>, waiting: i32, throughput: f64, kv: f64) -> WorkerLoadResponse {
        WorkerLoadResponse {
            loads: vec![SchedulerLoadSnapshot {
                num_waiting_uncached_tokens: tokens.unwrap_or_default(),
                num_waiting_uncached_tokens_available: Some(tokens.is_some()),
                num_waiting_reqs: waiting,
                gen_throughput: throughput,
                learned_prefill_capacity: Some(throughput),
                low_load_ttft_seconds: Some(0.0),
                token_usage: kv,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn publish(admission: &EstimatedWaitAdmission, worker: &Arc<dyn Worker>, tokens: i32) {
        publish_load(admission, worker, &load(Some(tokens), 0, 100.0, 0.0));
    }

    fn publish_load(
        admission: &EstimatedWaitAdmission,
        worker: &Arc<dyn Worker>,
        report: &WorkerLoadResponse,
    ) {
        let stamp = admission.poll_started(worker);
        admission.publish(worker, Some(report), stamp, Instant::now());
    }

    fn feature_registry() -> (Arc<WorkerRegistry>, Arc<dyn Worker>) {
        let registry = Arc::new(WorkerRegistry::new());
        registry.estimated_wait().configure(EstimatedWaitConfig {
            estimated_wait_calibration_shadow: true,
            estimated_wait_feature_shadow: true,
            ..config()
        });
        let worker = worker("http://feature:1");
        registry.register(worker.clone()).unwrap();
        let mut report = load(Some(1000), 2, 100.0, 0.4);
        report.dp_rank_count = 1;
        report.loads[0].low_load_ttft_seconds = Some(0.2);
        report.loads[0].prefill_capacity_learning_active = Some(false);
        report.loads[0].prefill_capacity_sample_count = Some(8);
        publish_load(registry.estimated_wait(), &worker, &report);
        (registry, worker)
    }

    fn deadline_shadow_placement(
        registry: &Arc<WorkerRegistry>,
        worker: &Arc<dyn Worker>,
        supported: bool,
    ) -> Result<Option<CalibrationForecast>, Response> {
        let capture = Mutex::new(None);
        placement::select_from(
            registry,
            &PolicyRegistry::new(PolicyConfig::RoundRobin),
            "m",
            std::slice::from_ref(worker),
            PlacementInputs {
                calibration: Some(&capture),
                feature_context_supported: supported,
                ..Default::default()
            },
        )?;
        Ok(capture.into_inner())
    }

    #[test]
    fn c2_raw_fallback_uses_current_calibration_baseline_and_uncalibrated_raw_extra() {
        let (registry, worker) = feature_registry();
        publish(registry.estimated_wait(), &worker, 0);
        let mut forecast = deadline_shadow_placement(&registry, &worker, true)
            .unwrap()
            .unwrap();
        forecast.baseline = Some(2.0);
        forecast.baseline_source = BaselineSource::Provisional;
        forecast.raw_baseline = Some(100.0);
        forecast.raw_extra = 42.0;
        forecast.queue_extra = 99.0;
        forecast.raw_ready = true;
        forecast.fresh = true;
        forecast.native_capacity_ready = true;
        let raw = forecast.raw_total_for_fallback().unwrap();
        assert_eq!(raw.total_ttft_seconds, 44.0);
        assert_eq!(forecast.selected_prediction().unwrap().total, 44.0);
        assert_eq!(raw.baseline_source, "provisional");
        forecast.baseline = Some(1.5);
        forecast.baseline_source = BaselineSource::Measured;
        assert_eq!(
            forecast
                .raw_total_for_fallback()
                .unwrap()
                .total_ttft_seconds,
            43.5
        );
        for field in 0..4 {
            let mut unavailable = forecast.clone();
            match field {
                0 => unavailable.raw_ready = false,
                1 => unavailable.fresh = false,
                2 => unavailable.native_capacity_ready = false,
                _ => unavailable.baseline = None,
            }
            assert!(unavailable.raw_total_for_fallback().is_none());
            assert!(unavailable.selected_prediction().is_none());
        }
    }

    #[test]
    fn c2_current_basis_raw_inputs_are_frozen_before_a_later_poll() {
        let (registry, worker) = feature_registry();
        // The fixture intentionally exceeds the admission budget. Capture the
        // same constructor under its guard without asserting a placement succeeds.
        let first = {
            let mut guard = registry.estimated_wait().begin().unwrap();
            guard.capture_calibration(&worker).unwrap()
        };
        let frozen = first.raw_fallback_work;
        let original_extra = first.raw_extra;
        let expected_extra = if frozen.measured_delay_only {
            frozen.observed_delay_floor
        } else {
            (frozen.token_work / frozen.original_capacity).max(frozen.observed_delay_floor)
        };
        assert!((first.raw_extra - expected_extra).abs() < 1e-9);
        let mut newer = load(Some(50000), 20, 100.0, 0.4);
        newer.dp_rank_count = 1;
        publish_load(registry.estimated_wait(), &worker, &newer);
        let later = {
            let mut guard = registry.estimated_wait().begin().unwrap();
            guard.capture_calibration(&worker).unwrap()
        };
        assert!(later.raw_fallback_work.poll_started > frozen.poll_started);
        assert_ne!(later.raw_fallback_work.token_work, frozen.token_work);
        assert_eq!(first.raw_fallback_work, frozen);
        assert_eq!(first.raw_extra, original_extra);
    }

    fn entry_started_history_load(second: u32) -> WorkerLoadResponse {
        serde_json::from_value(serde_json::json!({"dp_rank_count":1,"loads":[{
            "dp_rank":0,"native_request_counts_available":true,"native_request_counts_profile":7,
            "timely_prefill_progress_source":"prefill_started_requests",
            "prefill_capacity_learning_active":second>0,"prefill_capacity_sample_count":second.min(32),
            "learned_prefill_capacity":(second>=8).then_some(2000.0),
            "prefill_metric_provenance":{"schema_version":1,"counter_epoch":0,
                "progress_source":"prefill_started_requests","raw_counter":2.0*f64::from(second),
                "previous_raw_counter":(second>0).then(||2.0*f64::from(second-1)),
                "interval_seconds":(second>0).then_some(1.0),
                "interval_started_time_seconds":(second>0).then(||f64::from(second-1)),
                "interval_ended_time_seconds":f64::from(second),
                "raw_rate":(second>0).then_some(2.0),"previous_conversion_size_mean":1000.0,
                "capacity_interval_eligible":second>0,"counter_reset":false}
        }]})).unwrap()
    }

    #[test]
    fn sg_raw_proof_is_frozen_only_for_successful_opted_in_sglang_polls() {
        for (runtime, feature_enabled) in [
            (RuntimeType::Sglang, true),
            (RuntimeType::Vllm, true),
            (RuntimeType::Sglang, false),
        ] {
            let registry = Arc::new(WorkerRegistry::new());
            registry.estimated_wait().configure(EstimatedWaitConfig {
                estimated_wait_calibration_shadow: true,
                estimated_wait_feature_shadow: feature_enabled,
                ..config()
            });
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new("http://sg-pair-publication:1")
                    .status(WorkerStatus::Ready)
                    .runtime_type(runtime)
                    .worker_type(WorkerType::Regular)
                    .connection_mode(ConnectionMode::Http)
                    .build(),
            );
            registry.register(worker.clone()).unwrap();
            let admission = registry.estimated_wait();
            let mut proof = None;
            for second in 0..=8 {
                let mut report = entry_started_history_load(second);
                let rank = &mut report.loads[0];
                rank.num_waiting_reqs = 1;
                rank.num_waiting_uncached_tokens_available = Some(false);
                rank.avg_request_prefill_kv_computed_tokens = Some(1000.0);
                rank.prefill_size_sample_count = Some(8);
                let provenance = rank.prefill_metric_provenance.as_mut().unwrap();
                provenance.queue_size_observed_sample_count = Some(8);
                provenance.current_unshrunk_size_mean = Some(1000.0);
                let watermark = admission.poll_started(&worker);
                proof = admission.publish_with_sg_raw(
                    &worker,
                    Some(&report),
                    watermark,
                    Instant::now(),
                );
            }
            assert_eq!(
                proof.is_some(),
                runtime == RuntimeType::Sglang && feature_enabled
            );
            if proof.is_some() {
                let (last, before) = {
                    let entries = admission.entries.lock();
                    let entry = entries.get(worker.url()).unwrap();
                    (
                        entry.last_published.unwrap(),
                        format!("{:?}", entry.survival.native),
                    )
                };
                let watermark = admission.poll_started(&worker);
                assert!(admission
                    .publish_with_sg_raw(
                        &worker,
                        Some(&entry_started_history_load(8)),
                        watermark,
                        last - Duration::from_millis(1)
                    )
                    .is_none());
                let entries = admission.entries.lock();
                assert_eq!(
                    format!("{:?}", entries.get(worker.url()).unwrap().survival.native),
                    before
                );
                drop(entries);

                // Keep rate/counter history coherent on genuinely newer polls.
                // Current-size proof is independent of the previous conversion
                // mean and must not be filled from that historical value.
                for (second, current_mean) in [(9, None), (10, Some(999.0)), (11, Some(1000.0))] {
                    let mut report = entry_started_history_load(second);
                    let rank = &mut report.loads[0];
                    rank.num_waiting_reqs = 1;
                    rank.num_waiting_uncached_tokens_available = Some(false);
                    rank.avg_request_prefill_kv_computed_tokens = Some(1000.0);
                    rank.prefill_size_sample_count = Some(8);
                    let provenance = rank.prefill_metric_provenance.as_mut().unwrap();
                    provenance.queue_size_observed_sample_count = Some(8);
                    provenance.current_unshrunk_size_mean = current_mean;
                    let watermark = admission.poll_started(&worker);
                    let started = Instant::now();
                    assert!(
                        admission.entries.lock()[worker.url()]
                            .last_published
                            .unwrap()
                            < started
                    );
                    let proof =
                        admission.publish_with_sg_raw(&worker, Some(&report), watermark, started);
                    let entries = admission.entries.lock();
                    let entry = &entries[worker.url()];
                    assert_eq!(entry.last_published, Some(started));
                    let capacity = entry.survival.started_history_selection_for_test(
                        diagnostic_time(started),
                        survival::started_source_key(&report),
                    );
                    assert_eq!(capacity.status, "available");
                    assert!(capacity.history_complete);
                    assert_eq!(capacity.history_samples, second as usize);
                    let current = capacity.same_poll_observed_size.unwrap();
                    let expected = current_mean == Some(1000.0);
                    assert_eq!(proof.is_some(), expected);
                    assert_eq!(
                        current.reason,
                        if expected {
                            "same_poll_observed_completed_size"
                        } else {
                            "current_observed_size_mean_unavailable_or_mismatch"
                        }
                    );
                }
            }
            let watermark = admission.poll_started(&worker);
            assert!(admission
                .publish_with_sg_raw(&worker, None, watermark, Instant::now())
                .is_none());
        }
    }

    #[test]
    fn entry_calibration_reset_preserves_capacity_history_but_invalidation_clears_it() {
        let worker = worker("http://entry-lifecycle:1");
        let mut entry = Entry::new(&worker);
        for second in 0..=8 {
            entry.survival.native.raw_poll(
                Some(&entry_started_history_load(second)),
                f64::from(second),
                true,
            );
        }
        let key = survival::started_source_key(&entry_started_history_load(8));
        let before = entry.survival.started_history_selection_for_test(8.0, key);
        assert_eq!(before.status, "available");
        assert_eq!(before.lower_quartile_request_start_rate, Some(2.0));
        for reason in [
            "queue_drained",
            "baseline_changed",
            "native_profile_changed",
        ] {
            entry.reset_calibration(reason);
            let after = entry.survival.started_history_selection_for_test(8.0, key);
            assert_eq!(
                serde_json::to_value(after).unwrap(),
                serde_json::to_value(before).unwrap()
            );
        }
        // The next source interval still has exactly the old counter/time lineage.
        entry
            .survival
            .native
            .raw_poll(Some(&entry_started_history_load(9)), 9.0, true);
        let continued = entry.survival.started_history_selection_for_test(9.0, key);
        assert_eq!(continued.status, "available");
        assert_eq!(continued.history_samples, 9);
        assert_eq!(
            continued.lower_quartile_request_start_rate,
            before.lower_quartile_request_start_rate
        );
        assert_eq!(
            serde_json::to_value(continued.source).unwrap(),
            serde_json::to_value(before.source).unwrap()
        );
        let mut changed_source = entry_started_history_load(10);
        changed_source.loads[0].native_request_counts_profile = Some(8);
        changed_source.loads[0]
            .prefill_metric_provenance
            .as_mut()
            .unwrap()
            .counter_epoch = 1;
        let changed_key = survival::started_source_key(&changed_source);
        entry
            .survival
            .native
            .raw_poll(Some(&changed_source), 10.0, true);
        let changed = entry
            .survival
            .started_history_selection_for_test(10.0, changed_key);
        assert_eq!(changed.status, "unavailable");
        assert!(!changed.history_complete);
        // A late enabled, incomplete prefix is not turned into a proven empty window.
        entry.survival.clear_started_history();
        entry
            .survival
            .native
            .raw_poll(Some(&entry_started_history_load(11)), 11.0, true);
        let unknown = entry.survival.started_history_selection_for_test(11.0, key);
        assert!(!unknown.history_complete);
        assert_eq!(unknown.history_samples, 1);
        entry.reset_calibration("queue_drained");
        assert_eq!(
            serde_json::to_value(entry.survival.started_history_selection_for_test(11.0, key))
                .unwrap(),
            serde_json::to_value(unknown).unwrap()
        );
        entry.invalidate();
        let invalidated = entry.survival.started_history_selection_for_test(11.0, key);
        assert!(!invalidated.history_complete);
        assert_eq!(invalidated.history_samples, 0);
        entry
            .survival
            .native
            .raw_poll(Some(&entry_started_history_load(12)), 12.0, true);
        let new_worker = entry.survival.started_history_selection_for_test(12.0, key);
        assert_eq!(new_worker.status, "unavailable");
        assert!(!new_worker.history_complete);
        assert_eq!(new_worker.history_samples, 1);
    }

    #[test]
    fn deadline_shadow_rejected_placement_does_not_register_or_credit() {
        let (registry, worker) = feature_registry();
        assert!(deadline_shadow_placement(&registry, &worker, true).is_err());
        let entries = registry.estimated_wait().entries.lock();
        let entry = &entries[worker.url()];
        assert_eq!(entry.total_dispatched, 0);
        assert_eq!(entry.total_dispatched_requests, 0);
        assert_eq!(entry.deadline_work.lock().unwrap().summary().pending, 0);
    }

    #[test]
    fn native_deadline_budget_defaults_validation_and_remaining_dispatch_time() {
        assert!(EstimatedWaitConfig::default()
            .estimated_wait_native_hard_ttft_timeout_secs
            .is_none());
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY, 86401.0] {
            assert!(EstimatedWaitConfig {
                estimated_wait_native_hard_ttft_timeout_secs: Some(invalid),
                estimated_wait_feature_shadow: true,
                estimated_wait_calibration_shadow: true,
                ..config()
            }
            .validate()
            .is_err());
        }
        assert!(EstimatedWaitConfig {
            estimated_wait_native_hard_ttft_timeout_secs: Some(10.0),
            estimated_wait_feature_shadow: false,
            ..config()
        }
        .validate()
        .is_err());
        let (registry, worker) = feature_registry();
        publish(registry.estimated_wait(), &worker, 0);
        let mut forecast = deadline_shadow_placement(&registry, &worker, true)
            .unwrap()
            .unwrap();
        let now = Instant::now();
        forecast.native_hard_ttft_timeout_secs = Some(10.0);
        assert_eq!(
            forecast.native_dispatch_headers(now, now),
            Err("native_deadline_not_verified_or_stale")
        );
        forecast.native_deadline_capability_valid_until = Some(now + Duration::from_secs(5));
        assert_eq!(
            forecast
                .native_dispatch_headers(now - Duration::from_secs(2), now)
                .unwrap()
                .unwrap()
                .1,
            Some(8000)
        );
        assert_eq!(
            forecast.native_dispatch_headers(now - Duration::from_secs(10), now),
            Err("native_deadline_budget_exhausted_before_dispatch")
        );
        forecast.deadline_dispatch = None;
        assert_eq!(
            forecast.native_dispatch_headers(now, now),
            Err("native_deadline_request_not_tracked")
        );
    }

    #[test]
    fn native_restart_between_capability_and_send_never_confirms_requested_deadline() {
        let (registry, worker) = feature_registry();
        let admission = registry.estimated_wait();
        let native: WorkerLoadResponse = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/native_deadline/sglang_active.json"
        )))
        .unwrap();
        let raw = load(Some(0), 0, 100.0, 0.0);
        let watermark = admission.poll_started(&worker);
        let start = Instant::now();
        admission.publish(&worker, Some(&raw), watermark.clone(), start);
        admission.publish_native_deadline(&worker, Some(&native), watermark, start, Instant::now());
        let mut forecast = deadline_shadow_placement(&registry, &worker, true)
            .unwrap()
            .unwrap();
        forecast.native_hard_ttft_timeout_secs = Some(10.0);
        let request_start = Instant::now();
        assert!(forecast
            .native_dispatch_headers(request_start, request_start)
            .unwrap()
            .unwrap()
            .1
            .is_some());
        let mut observer = StreamTtftObserver::new(&registry, worker.url(), request_start)
            .with_calibration(Some(forecast));
        observer.upstream_send_started();
        let mut restarted = native;
        let telemetry = restarted.loads[0].deadline_recovery.as_mut().unwrap();
        telemetry.engine_epoch = "engine-restarted-before-dispatch".into();
        telemetry.snapshot_seq = 1;
        telemetry.native_population = 0;
        telemetry.active_requests.clear();
        telemetry.hard_ttft_deadline_supported = false;
        telemetry.abort_enforcement_enabled = false;
        let watermark = admission.poll_started(&worker);
        let start = Instant::now();
        admission.publish(&worker, Some(&raw), watermark.clone(), start);
        admission.publish_native_deadline(
            &worker,
            Some(&restarted),
            watermark,
            start,
            Instant::now(),
        );
        observer.downstream_disconnected();
        drop(observer);
        let entries = admission.entries.lock();
        let entry = &entries[worker.url()];
        assert_eq!(entry.deadline_work.lock().unwrap().summary().pending, 1);
        assert_eq!(entry.native_deadline.acknowledged_requests, 0);
        assert!(entry.native_deadline.confirmed_deadlines.is_empty());
        assert!(!entry
            .native_deadline
            .hard_deadlines_present(Instant::now(), 30.0));
    }

    #[test]
    fn deadline_shadow_local_failure_rolls_back_once_without_changing_raw_control() {
        let (registry, worker) = feature_registry();
        publish(registry.estimated_wait(), &worker, 0);
        let forecast = deadline_shadow_placement(&registry, &worker, true)
            .unwrap()
            .unwrap();
        let clone = forecast.clone();
        let ledger =
            Arc::clone(&registry.estimated_wait().entries.lock()[worker.url()].deadline_work);
        assert_eq!(ledger.lock().unwrap().summary().pending, 1);
        drop(forecast); // serialization/dispatch-time veto before any send
        assert_eq!(ledger.lock().unwrap().summary().pending, 1);
        drop(clone);
        assert_eq!(ledger.lock().unwrap().summary().pending, 0);
        let entries = registry.estimated_wait().entries.lock();
        assert_eq!(entries[worker.url()].total_dispatched, 100);
        assert_eq!(entries[worker.url()].total_dispatched_requests, 1);
    }

    #[test]
    fn deadline_shadow_unsupported_observation_path_does_not_claim_tracking() {
        let (registry, worker) = feature_registry();
        publish(registry.estimated_wait(), &worker, 0);
        // gRPC and nonstreaming callers do not provide feature_context_supported.
        let forecast = deadline_shadow_placement(&registry, &worker, false).unwrap();
        assert!(forecast
            .as_ref()
            .is_none_or(|f| f.deadline_dispatch.is_none()));
        let entries = registry.estimated_wait().entries.lock();
        assert_eq!(
            entries[worker.url()]
                .deadline_work
                .lock()
                .unwrap()
                .summary()
                .pending,
            0
        );
        assert_eq!(entries[worker.url()].total_dispatched, 100);
    }

    #[test]
    fn deadline_shadow_stream_timeout_disconnect_and_drop_require_native_release() {
        for terminal in ["timeout", "disconnect", "drop", "first_text"] {
            let (registry, worker) = feature_registry();
            publish(registry.estimated_wait(), &worker, 0);
            let forecast = deadline_shadow_placement(&registry, &worker, true).unwrap();
            let ledger =
                Arc::clone(&registry.estimated_wait().entries.lock()[worker.url()].deadline_work);
            let mut observer = StreamTtftObserver::new(
                &registry,
                worker.url(),
                Instant::now() - Duration::from_millis(1),
            )
            .with_calibration(forecast);
            observer.upstream_send_started();
            assert_eq!(ledger.lock().unwrap().summary().forwarded, 1);
            match terminal {
                "timeout" => observer.upstream_timeout(),
                "disconnect" => observer.downstream_disconnected(),
                "first_text" => {
                    observer.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n");
                }
                _ => {}
            }
            observer.cancel();
            drop(observer);
            let summary = ledger.lock().unwrap().summary();
            assert_eq!(summary.pending, usize::from(terminal != "first_text"));
            assert_eq!(summary.unknown_work, summary.pending);
            assert_eq!(
                registry.estimated_wait().entries.lock()[worker.url()].total_dispatched,
                100
            );
        }
    }

    fn feature_observer(
        registry: &Arc<WorkerRegistry>,
        worker: &Arc<dyn Worker>,
    ) -> StreamTtftObserver {
        let mut guard = registry.estimated_wait().begin().unwrap();
        let mut forecast = guard.capture_calibration(worker);
        guard.attach_feature_context(worker, &mut forecast);
        drop(guard);
        StreamTtftObserver::new(registry, worker.url(), Instant::now()).with_calibration(forecast)
    }
    fn pending_work_report() -> WorkerLoadResponse {
        let mut report = load(None, 4, 100.0, 0.4);
        report.dp_rank_count = 1;
        let rank = &mut report.loads[0];
        rank.num_running_reqs = 1;
        rank.native_request_counts_available = Some(true);
        rank.native_request_counts_profile = Some(7);
        rank.timely_prefill_progress_source = Some("local_compute_tokens".to_owned());
        rank.avg_request_prefill_kv_computed_tokens = Some(100.0);
        rank.prefill_size_sample_count = Some(32);
        rank.prefill_capacity_learning_active = Some(false);
        rank.prefill_capacity_sample_count = Some(8);
        rank.low_load_ttft_seconds = Some(0.2);
        report
    }

    #[test]
    fn explicit_size_basis_provenance_leaves_original_queue_and_capacity_arithmetic_unchanged() {
        let cfg = config();
        let mut report = pending_work_report();
        let original = cfg.prepare(&report).unwrap();
        report.loads[0].prefill_metric_provenance =
            Some(openai_protocol::worker::PrefillMetricProvenance {
                schema_version: 1,
                progress_source: Some("local_compute_tokens".into()),
                queue_size_observed_sample_count: Some(32),
                ..Default::default()
            });
        let enriched = cfg.prepare(&report).unwrap();
        assert!(original.size_basis.is_none());
        assert_eq!(enriched.size_basis.unwrap().kind, basis::Kind::Observed);
        assert_eq!(original.seconds(200), enriched.seconds(200));
        assert_eq!(
            original.queue_excess_seconds(200),
            enriched.queue_excess_seconds(200)
        );
        assert_eq!(original.decision_profile, enriched.decision_profile);
        assert_eq!(
            original.effective_prefill_capacity,
            enriched.effective_prefill_capacity
        );
        assert_eq!(original.capacity_ready, enriched.capacity_ready);
    }

    fn pending_work_observer(
        registry: &Arc<WorkerRegistry>,
        worker: &Arc<dyn Worker>,
        bytes: usize,
    ) -> StreamTtftObserver {
        let mut guard = registry.estimated_wait().begin().unwrap();
        let mut forecast = guard.capture_calibration(worker);
        guard.attach_feature_context(worker, &mut forecast);
        forecast
            .as_mut()
            .unwrap()
            .set_prompt_work(openai_protocol::common::PromptWorkHint::TextBytes(bytes));
        drop(guard);
        StreamTtftObserver::new(registry, worker.url(), Instant::now()).with_calibration(forecast)
    }

    #[test]
    fn pending_work_preserves_native_drain_cap_and_drops_it_after_rebound_or_staleness() {
        let (registry, worker) = feature_registry();
        let admission = registry.estimated_wait();
        let mut report = pending_work_report();
        publish_load(admission, &worker, &report);
        let _observer = pending_work_observer(&registry, &worker, 400_000);
        report.loads[0].num_waiting_reqs = 40;
        publish_load(admission, &worker, &report);
        let mut guard = admission.begin().unwrap();
        let now = Instant::now();
        {
            let entry = guard.entries.get_mut(worker.url()).unwrap();
            entry.recovery.reset("test_native_drain");
            for i in 0..7 {
                report.loads[0].num_waiting_reqs = 100 - i * 10;
                entry.recovery.observe(
                    &report,
                    now - Duration::from_secs(30 - i as u64 * 5),
                    60.0,
                    entry.total_dispatched_requests,
                );
            }
        }
        guard.credit(&worker, Some(&[0; 100]));
        let mut forecast = guard.capture_calibration(&worker);
        guard.attach_feature_context(&worker, &mut forecast);
        let candidate = forecast.unwrap().pending_work_admission.unwrap();
        assert!(candidate.queue.applied);
        assert_eq!(candidate.recovery_cap_seconds, Some(20.5));
        assert!(candidate.debt_seconds > 0.0);
        assert_eq!(candidate.candidate_x, 20.5 + candidate.debt_seconds);
        assert_eq!(candidate.control_x, candidate.candidate_x);

        let entry = guard.entries.get_mut(worker.url()).unwrap();
        // An unsupported population rebound invalidates the existing drain evidence.
        report.loads[0].num_waiting_reqs = 100;
        entry.recovery.observe(
            &report,
            Instant::now(),
            60.0,
            entry.total_dispatched_requests,
        );
        let at = Instant::now();
        let snapshot = entry.decision_snapshot(admission.config(), at).unwrap();
        let original = decision::Admission {
            snapshot,
            predictions: entry
                .decision
                .predictions(snapshot, diagnostic_time(at), true),
        };
        let candidate = entry
            .pending_work_admission(admission.config(), &original, at)
            .unwrap();
        assert_eq!(candidate.recovery_cap_seconds, None);
        assert_eq!(candidate.candidate_x, candidate.control_x);
        let mut stale = original;
        stale.snapshot.fresh = false;
        let candidate = entry
            .pending_work_admission(admission.config(), &stale, at)
            .unwrap();
        assert_eq!(candidate.recovery_cap_seconds, None);
        assert!(!candidate.queue.applied);
        assert_eq!(candidate.candidate_x, stale.snapshot.x);
    }

    #[test]
    fn pending_work_capacity_source_selects_same_poll_control_or_exact_historical() {
        for source in [
            Some("local_compute_tokens"),
            Some("prefill_started_requests"),
            Some("future_unknown_source"),
            None,
        ] {
            let (registry, worker) = feature_registry();
            let admission = registry.estimated_wait();
            let mut report = pending_work_report();
            report.loads[0].timely_prefill_progress_source = source.map(str::to_owned);
            publish_load(admission, &worker, &report);
            let _observer = pending_work_observer(&registry, &worker, 40000);
            publish_load(admission, &worker, &report);
            let mut guard = admission.begin().unwrap();
            guard.credit(&worker, Some(&[0; 100]));
            let mut forecast = guard.capture_calibration(&worker);
            guard.attach_feature_context(&worker, &mut forecast);
            let forecast = forecast.unwrap();
            let original = forecast.decision_admission.unwrap();
            let entry = &guard.entries[worker.url()];
            let candidate = forecast.pending_work_admission.unwrap();
            assert!(candidate.queue.applied);
            assert_eq!(candidate.capacity_progress_source.as_deref(), source);
            assert!(candidate.control_x > candidate.original_x);
            assert!(candidate.debt_seconds > 0.0);
            assert_eq!(
                candidate.selected_pending_applied,
                source == Some("local_compute_tokens")
            );
            if candidate.selected_pending_applied {
                assert_eq!(candidate.source_gate_reason, "local_compute_tokens");
                assert_eq!(candidate.candidate_x, candidate.control_x);
                assert_eq!(
                    serde_json::to_value(&candidate.predictions).unwrap(),
                    serde_json::to_value(&entry.decision.predictions(
                        decision::Snapshot {
                            x: candidate.control_x,
                            ..original.snapshot
                        },
                        original.snapshot.at,
                        candidate.available
                    ))
                    .unwrap()
                );
            } else {
                assert_eq!(
                    candidate.source_gate_reason,
                    if source.is_some() {
                        "capacity_progress_source_unsupported"
                    } else {
                        "capacity_progress_source_missing"
                    }
                );
                assert_eq!(candidate.candidate_x, candidate.original_x);
                assert_eq!(candidate.candidate_raw_total, candidate.original_raw_total);
                assert_eq!(
                    serde_json::to_value(&candidate.predictions).unwrap(),
                    serde_json::to_value(&original.predictions).unwrap()
                );
            }
        }
    }

    #[test]
    fn pending_work_source_change_resets_inventory_before_selecting_new_poll() {
        let (registry, worker) = feature_registry();
        let admission = registry.estimated_wait();
        let mut report = pending_work_report();
        publish_load(admission, &worker, &report);
        let _old = pending_work_observer(&registry, &worker, 40000);
        let watermark = admission.poll_started(&worker);
        report.loads[0].timely_prefill_progress_source =
            Some("prefill_started_requests".to_owned());
        admission.publish(&worker, Some(&report), watermark, Instant::now());
        let mut guard = admission.begin().unwrap();
        let mut forecast = guard.capture_calibration(&worker);
        guard.attach_feature_context(&worker, &mut forecast);
        let candidate = forecast.unwrap().pending_work_admission.unwrap();
        assert_eq!(
            candidate.capacity_progress_source.as_deref(),
            Some("prefill_started_requests")
        );
        assert!(!candidate.queue.applied);
        assert!(!candidate.selected_pending_applied);
        assert_eq!(candidate.source_gate_reason, "pending_queue_fallback");
        assert_eq!(candidate.queue.reason, "inventory_generation_changed");
        assert_eq!(candidate.candidate_x, candidate.original_x);
        assert_eq!(candidate.control_x, candidate.original_x);
    }

    #[test]
    fn pending_work_poll_freezes_inventory_and_preserves_control_and_saved_labels() {
        let (registry, worker) = feature_registry();
        let admission = registry.estimated_wait();
        let report = pending_work_report();
        publish_load(admission, &worker, &report);
        let mut first = pending_work_observer(&registry, &worker, 4000);
        let second = pending_work_observer(&registry, &worker, 12000);
        let started = Instant::now();
        let watermark = admission.poll_started(&worker);
        first.cancel(); // Resolution during I/O affects only the next frozen summary.
        admission.publish(&worker, Some(&report), watermark, started);
        let mut guard = admission.begin().unwrap();
        let before =
            serde_json::to_value(guard.entries[worker.url()].decision.diagnostics()).unwrap();
        let overloaded = guard.entries[worker.url()]
            .sample
            .as_ref()
            .unwrap()
            .overloaded;
        let mut forecast = guard.capture_calibration(&worker);
        guard.attach_feature_context(&worker, &mut forecast);
        let saved = forecast
            .as_ref()
            .unwrap()
            .pending_work_admission
            .as_ref()
            .unwrap();
        assert_eq!(saved.queue.summary.known, 2);
        assert_eq!(saved.queue.coverage, 0.4);
        assert_eq!(saved.queue.candidate_tokens, 3440.0);
        assert_eq!(saved.candidate_x, 34.4);
        assert_eq!(saved.original_x, 4.0);
        assert_eq!(
            forecast
                .as_ref()
                .unwrap()
                .decision_admission
                .as_ref()
                .unwrap()
                .snapshot
                .x,
            4.0
        );
        assert_eq!(forecast.as_ref().unwrap().queue_extra, 4.0);
        let original = forecast
            .as_ref()
            .unwrap()
            .decision_admission
            .as_ref()
            .unwrap();
        for (old, new) in original.predictions.iter().zip(&saved.predictions) {
            assert_eq!(old.coefficients, new.coefficients);
            assert_eq!(old.source_arm, new.source_arm);
            assert!((new.extra / old.extra - 8.6).abs() < 1e-12);
        }
        assert_eq!(
            serde_json::to_value(guard.entries[worker.url()].decision.diagnostics()).unwrap(),
            before
        );
        assert_eq!(
            guard.entries[worker.url()]
                .sample
                .as_ref()
                .unwrap()
                .overloaded,
            overloaded
        );
        drop(guard);
        let mut current = StreamTtftObserver::new(&registry, worker.url(), Instant::now())
            .with_calibration(forecast);
        current.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n");
        current.cancel(); // First text then cancellation must not release another row.
        drop(current);
        drop(second);
        publish_load(admission, &worker, &report);
        assert_eq!(
            admission.begin().unwrap().entries[worker.url()]
                .sample
                .as_ref()
                .unwrap()
                .pending_work_queue
                .unwrap()
                .summary
                .pending,
            0
        );
    }

    #[test]
    fn pending_work_generation_changes_during_poll_reject_frozen_sizes() {
        let (registry, worker) = feature_registry();
        let admission = registry.estimated_wait();
        let report = pending_work_report();
        publish_load(admission, &worker, &report);
        let mut observer = pending_work_observer(&registry, &worker, 40000);
        let started = Instant::now();
        let watermark = admission.poll_started(&worker);
        admission
            .begin()
            .unwrap()
            .entries
            .get_mut(worker.url())
            .unwrap()
            .reset_calibration("test_baseline_changed_during_io");
        observer.cancel(); // Old handle cannot change the new inventory generation.
        admission.publish(&worker, Some(&report), watermark, started);
        let guard = admission.begin().unwrap();
        let q = guard.entries[worker.url()]
            .sample
            .as_ref()
            .unwrap()
            .pending_work_queue
            .unwrap();
        assert_eq!(q.summary.known, 1);
        assert!(!q.applied);
        assert_eq!(q.reason, "inventory_generation_changed");
        assert_eq!(q.original_tokens, q.candidate_tokens);
        drop(guard);
        // A reset after publication is independently fenced at admission capture.
        let _fresh = pending_work_observer(&registry, &worker, 40000);
        publish_load(admission, &worker, &report);
        let mut guard = admission.begin().unwrap();
        let mut forecast = guard.capture_calibration(&worker);
        guard.attach_feature_context(&worker, &mut forecast);
        let original = forecast.unwrap().decision_admission.unwrap();
        assert!(
            guard.entries[worker.url()]
                .pending_work_admission(admission.config(), &original, Instant::now())
                .unwrap()
                .queue
                .applied
        );
        let entry = guard.entries.get_mut(worker.url()).unwrap();
        entry.reset_calibration("test_after_publication");
        let candidate = entry
            .pending_work_admission(admission.config(), &original, Instant::now())
            .unwrap();
        assert!(!candidate.queue.applied);
        assert_eq!(candidate.queue.reason, "inventory_generation_changed");
        assert_eq!(candidate.candidate_x, original.snapshot.x);
    }

    #[test]
    fn pending_work_invalid_counts_stale_and_native_authoritative_are_fallbacks() {
        let (registry, worker) = feature_registry();
        let admission = registry.estimated_wait();
        let report = pending_work_report();
        publish_load(admission, &worker, &report);
        let _observer = pending_work_observer(&registry, &worker, 40000);
        for mode in 0..6 {
            let mut report = report.clone();
            if mode == 0 {
                report.loads[0].num_running_reqs = -1;
            }
            if mode == 1 {
                report.loads[0].native_request_counts_profile = None;
            }
            if mode == 2 {
                report.loads[0].num_waiting_uncached_tokens_available = Some(true);
                report.loads[0].num_waiting_uncached_tokens = 0;
            }
            if mode == 3 {
                report.loads[0].num_waiting_uncached_tokens_available = None;
                report.loads[0].num_waiting_uncached_tokens = 0;
            }
            if mode == 4 {
                report.loads[0].dp_rank = 1;
            }
            publish_load(admission, &worker, &report);
            let mut guard = admission.begin().unwrap();
            let published_reason = guard.entries[worker.url()]
                .sample
                .as_ref()
                .unwrap()
                .pending_work_queue
                .unwrap()
                .reason;
            let mut forecast = guard.capture_calibration(&worker);
            guard.attach_feature_context(&worker, &mut forecast);
            let original = forecast
                .as_ref()
                .unwrap()
                .decision_admission
                .as_ref()
                .unwrap();
            let candidate = forecast
                .as_ref()
                .unwrap()
                .pending_work_admission
                .as_ref()
                .unwrap();
            if mode < 5 {
                assert!(!candidate.queue.applied);
                assert_eq!(candidate.candidate_x, original.snapshot.x);
            }
            if mode == 2 {
                assert_eq!(published_reason, "authoritative_native_tokens");
            }
            if mode == 3 {
                assert_eq!(published_reason, "native_token_availability_unknown");
            }
            if mode == 5 {
                let entry = guard.entries.get_mut(worker.url()).unwrap();
                let mut stale = original.clone();
                stale.snapshot.fresh = false;
                let candidate = entry
                    .pending_work_admission(admission.config(), &stale, Instant::now())
                    .unwrap();
                assert!(!candidate.available);
                assert!(!candidate.queue.applied);
                assert_eq!(candidate.candidate_x, stale.snapshot.x);
                stale.snapshot.fresh = true;
                stale.snapshot.native_ready = false;
                assert!(
                    !entry
                        .pending_work_admission(admission.config(), &stale, Instant::now())
                        .unwrap()
                        .available
                );
            }
        }
    }

    #[test]
    fn native_deadline_sse_is_measured_censor_only_after_done_without_ack() {
        let (registry, worker) = feature_registry();
        publish_load(
            registry.estimated_wait(),
            &worker,
            &load(Some(100), 1, 100.0, 0.0),
        );
        let mut forecast = deadline_shadow_placement(&registry, &worker, true)
            .unwrap()
            .unwrap();
        forecast.native_hard_ttft_timeout_secs = Some(300.0);
        let ledger =
            Arc::clone(&registry.estimated_wait().entries.lock()[worker.url()].deadline_work);
        let mut observer = StreamTtftObserver::new(&registry, worker.url(), Instant::now())
            .with_calibration(Some(forecast));
        observer.upstream_send_started();
        observer.push(b"data: {\"error\":{\"type\":\"native_ttft_dead");
        assert!(!observer.feature_resolved);
        observer.push(b"line\",\"message\":\"expired\"}}\n\n");
        assert!(!observer.feature_resolved);
        observer.push(b"data: [DONE]\n\n");
        assert!(observer.feature_resolved);
        observer.upstream_timeout();
        drop(observer);
        assert_eq!(ledger.lock().unwrap().summary().pending, 1);
        let entries = registry.estimated_wait().entries.lock();
        let entry = &entries[worker.url()];
        assert_eq!(entry.native_deadline.acknowledged_requests, 0);
        let labels = entry.decision.labels_for_test();
        assert_eq!(labels.len(), 1);
        assert!(!labels[0].exact);
        assert!(labels[0].elapsed > 0.0 && labels[0].elapsed < 300.0);
    }

    #[test]
    fn malformed_interrupted_and_after_text_native_deadline_sse_preserve_truth() {
        for mode in [
            "missing_done",
            "wrong_type",
            "missing_message",
            "after_text",
            "not_configured",
        ] {
            let (registry, worker) = feature_registry();
            // Keep a visible native queue so first-text observations qualify for training.
            publish_load(
                registry.estimated_wait(),
                &worker,
                &load(Some(100), 2, 100.0, 0.0),
            );
            let mut forecast = deadline_shadow_placement(&registry, &worker, true)
                .unwrap()
                .unwrap();
            if mode != "not_configured" {
                forecast.native_hard_ttft_timeout_secs = Some(300.0);
            }
            let mut observer = StreamTtftObserver::new(&registry, worker.url(), Instant::now())
                .with_calibration(Some(forecast));
            observer.upstream_send_started();
            if mode == "after_text" {
                observer.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"first\"}}]}\n\n");
            }
            let error = match mode {
                "wrong_type" => b"data: {\"error\":{\"type\":\"native_ttft_deadline_extra\",\"message\":\"expired\"}}\n\n".as_slice(),
                "missing_message" => b"data: {\"error\":{\"type\":\"native_ttft_deadline\"}}\n\n".as_slice(),
                _ => b"data: {\"error\":{\"type\":\"native_ttft_deadline\",\"message\":\"expired\"}}\n\n".as_slice(),
            };
            observer.push(error);
            if mode != "missing_done" {
                observer.push(b"data: [DONE]\n\n");
            }
            drop(observer);
            let entries = registry.estimated_wait().entries.lock();
            let labels = entries[worker.url()].decision.labels_for_test();
            assert_eq!(labels.len(), usize::from(mode == "after_text"), "{mode}");
            assert!(labels.iter().all(|label| label.exact), "{mode}");
        }
    }
    #[test]
    fn feature_deadline_poll_and_real_terminal_are_separate_epoch_safe_observations() {
        let (registry, worker) = feature_registry();
        let mut guard = registry.estimated_wait().begin().unwrap();
        let mut forecast = guard.capture_calibration(&worker);
        guard.attach_feature_context(&worker, &mut forecast);
        drop(guard);
        let start = Instant::now() - Duration::from_secs(301);
        let mut observer =
            StreamTtftObserver::new(&registry, worker.url(), start).with_calibration(forecast);
        let api = registry.estimated_wait();
        let poll = api.poll_started(&worker);
        // A failed poll intentionally invalidates the learning profile. Use a
        // fresh matching poll to exercise lower-bound-to-exact label upgrading.
        let mut report = load(Some(1000), 2, 100.0, 0.4);
        report.dp_rank_count = 1;
        report.loads[0].low_load_ttft_seconds = Some(0.2);
        report.loads[0].prefill_capacity_learning_active = Some(false);
        report.loads[0].prefill_capacity_sample_count = Some(8);
        api.publish(&worker, Some(&report), poll, Instant::now());
        {
            let entries = api.entries.lock();
            let labels = entries[worker.url()].decision.labels_for_test();
            assert_eq!(labels.len(), 1);
            assert!(!labels[0].exact);
            assert_eq!(labels[0].elapsed, 300.0);
        }
        observer.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"late\"}}]}\n\n");
        let entries = api.entries.lock();
        let labels = entries[worker.url()].decision.labels_for_test();
        assert_eq!(labels.len(), 1);
        assert!(labels[0].exact);
        assert!(labels[0].elapsed >= 301.0);
    }

    #[test]
    fn feature_old_outcomes_and_wrong_worker_identity_cannot_mutate_replacement() {
        let (registry, worker) = feature_registry();
        let mut observer = feature_observer(&registry, &worker);
        registry
            .estimated_wait()
            .entries
            .lock()
            .get_mut(worker.url())
            .unwrap()
            .reset_calibration("test");
        observer.downstream_disconnected();
        let mut guard = registry.estimated_wait().begin().unwrap();
        let other = self::worker(worker.url());
        guard.record_feature_decision(&other, false, false);
        let entry = &guard.entries[worker.url()];
        assert!(entry.decision.labels_for_test().is_empty());
        assert!(
            entry
                .decision
                .context(diagnostic_time(Instant::now()), None, None, false)
                .complete
        );
        assert!(AdmissionGuard::feature_scope(std::slice::from_ref(&worker), false).is_none());
        assert!(AdmissionGuard::feature_scope(&[worker.clone(), other], true).is_none());
    }

    #[test]
    fn decision_progress_requires_complete_timely_rank_evidence() {
        let cfg = config();
        let mut report = load(Some(1000), 2, 100.0, 0.4);
        report.dp_rank_count = 1;
        let rank = &mut report.loads[0];
        rank.native_request_counts_available = Some(true);
        rank.native_request_counts_profile = Some(7);
        rank.timely_prefill_progress = Some(0.0);
        rank.timely_prefill_progress_source = Some("local_compute_tokens".to_owned());
        rank.timely_prefill_progress_interval = Some(1.0);
        assert_eq!(cfg.prepare(&report).unwrap().timely_progress, Some(0.0));
        report.loads[0].timely_prefill_progress_interval = Some(1000.0);
        assert_eq!(cfg.prepare(&report).unwrap().timely_progress, None);
        report.loads[0].timely_prefill_progress_interval = Some(1.0);
        report.loads[0].timely_prefill_progress_source = Some("completed_uncached".to_owned());
        assert_eq!(cfg.prepare(&report).unwrap().timely_progress, None);
        report.loads[0].timely_prefill_progress_source = Some("local_compute_tokens".to_owned());
        report.dp_rank_count = 2;
        assert!(cfg.prepare(&report).is_none());
        report.loads.push(report.loads[0].clone());
        assert_eq!(cfg.prepare(&report).unwrap().timely_progress, None);
    }

    #[test]
    fn decision_saved_prediction_is_immutable_and_never_changes_admission() {
        let (registry, worker) = feature_registry();
        let mut guard = registry.estimated_wait().begin().unwrap();
        let before = guard
            .entries
            .get(worker.url())
            .unwrap()
            .sample
            .as_ref()
            .unwrap()
            .overloaded;
        let mut forecast = guard.capture_calibration(&worker);
        guard.attach_feature_context(&worker, &mut forecast);
        let saved = forecast.unwrap();
        let frozen = serde_json::to_value(&saved.decision_admission).unwrap();
        let entry = guard.entries.get_mut(worker.url()).unwrap();
        entry.decision.mark_incomplete();
        assert_eq!(
            serde_json::to_value(&saved.decision_admission).unwrap(),
            frozen
        );
        assert_eq!(entry.sample.as_ref().unwrap().overloaded, before);
        assert!(saved.decision_admission.unwrap().snapshot.context.complete);
        assert!(
            !entry
                .decision
                .context(diagnostic_time(Instant::now()), None, None, false)
                .complete
        );
    }

    fn calibration_registry(native_ready: bool) -> (Arc<WorkerRegistry>, Arc<dyn Worker>) {
        let registry = Arc::new(WorkerRegistry::new());
        registry.estimated_wait().configure(EstimatedWaitConfig {
            estimated_wait_calibration_shadow: true,
            ..config()
        });
        let worker = worker("http://calibration:1");
        registry.register(worker.clone()).unwrap();
        let mut report = load(Some(1_000), 2, 100.0, 0.0);
        report.loads[0].low_load_ttft_seconds = Some(0.2);
        report.loads[0].prefill_capacity_learning_active = Some(false);
        report.loads[0].prefill_capacity_sample_count = Some(if native_ready { 8 } else { 0 });
        if !native_ready {
            report.loads[0].learned_prefill_capacity = None;
        }
        publish_load(registry.estimated_wait(), &worker, &report);
        (registry, worker)
    }

    fn calibration_probe(
        admission: &EstimatedWaitAdmission,
        worker: &Arc<dyn Worker>,
    ) -> TtftProbe {
        let forecast = admission
            .begin()
            .unwrap()
            .capture_calibration(worker)
            .unwrap();
        let mut probe = admission.ttft_probe(worker.url()).unwrap();
        probe.calibration = Some(forecast);
        probe
    }

    #[test]
    fn calibration_rejects_a_saved_zero_queue_even_if_it_grows_before_first_text() {
        let (registry, worker) = calibration_registry(true);
        let admission = registry.estimated_wait();
        let mut report = load(Some(0), 0, 100.0, 0.0);
        report.loads[0].low_load_ttft_seconds = Some(0.2);
        report.loads[0].prefill_capacity_learning_active = Some(false);
        report.loads[0].prefill_capacity_sample_count = Some(8);
        publish_load(admission, &worker, &report);
        admission.begin().unwrap().credit(&worker, Some(&[0; 100]));
        let probes: Vec<_> = (0..4)
            .map(|_| calibration_probe(admission, &worker))
            .collect();
        report.loads[0].num_waiting_reqs = 8;
        report.loads[0].num_waiting_uncached_tokens = 800;
        publish_load(admission, &worker, &report);
        for probe in probes {
            admission.observe_ttft(probe, 20.2, 0.0);
        }
        let entries = admission.entries.lock();
        assert!(
            entries[worker.url()]
                .sample
                .as_ref()
                .unwrap()
                .prepared
                .native_waiting
                > 0.0
        );
    }

    #[test]
    fn calibration_ineligible_healthy_requests_still_learn_the_low_load_baseline() {
        let (registry, worker) = calibration_registry(true);
        let admission = registry.estimated_wait();
        worker.increment_load();
        let mut report = load(Some(0), 0, 100.0, 0.0);
        report.loads[0].low_load_ttft_seconds = None;
        report.loads[0].prefill_capacity_learning_active = Some(false);
        report.loads[0].prefill_capacity_sample_count = Some(8);
        publish_load(admission, &worker, &report);
        for _ in 0..8 {
            let probe = calibration_probe(admission, &worker);
            admission.observe_ttft(probe, 0.25, 0.05);
        }
        let entries = admission.entries.lock();
        let entry = &entries[worker.url()];
        assert_eq!(ttft_median(&entry.low_load_ttfts), Some(0.25));
        assert_eq!(entry.sample.as_ref().unwrap().prepared.baseline, Some(0.25));

        worker.decrement_load();
    }

    fn provisional_calibration_registry(
        provisional: Option<f64>,
    ) -> (Arc<WorkerRegistry>, Arc<dyn Worker>) {
        let registry = Arc::new(WorkerRegistry::new());
        registry.estimated_wait().configure(EstimatedWaitConfig {
            estimated_wait_calibration_shadow: true,
            estimated_wait_provisional_baseline_secs: provisional,
            ..config()
        });
        let worker = worker("http://provisional:1");
        registry.register(worker.clone()).unwrap();
        let mut report = load(Some(1_000), 2, 100.0, 0.0);
        report.loads[0].low_load_ttft_seconds = None;
        report.loads[0].prefill_capacity_learning_active = Some(false);
        report.loads[0].prefill_capacity_sample_count = Some(8);
        publish_load(registry.estimated_wait(), &worker, &report);
        (registry, worker)
    }

    #[test]
    fn provisional_baseline_bootstraps_fit_without_changing_raw_baseline() {
        let (registry, worker) = provisional_calibration_registry(Some(2.0));
        let admission = registry.estimated_wait();
        let mut entries = admission.entries.lock();
        let entry = entries.get_mut(worker.url()).unwrap();
        let now = Instant::now();
        let forecast = entry.calibration_forecast(admission.config(), now).unwrap();
        assert_eq!(forecast.baseline, Some(2.0));
        assert_eq!(forecast.baseline_source, BaselineSource::Provisional);
        assert_eq!(forecast.raw_baseline, None);
        assert_eq!(forecast.seconds(), 12.0);

        assert_eq!(entry.sample.as_ref().unwrap().prepared.baseline, None);
        assert_eq!(entry.sample.as_ref().unwrap().prepared.seconds(0), 10.0);
        assert!(entry.sample.as_ref().unwrap().prepared.capacity_ready);
    }

    #[test]
    fn unknown_baseline_never_becomes_zero_calibration_baseline() {
        let (registry, worker) = provisional_calibration_registry(None);
        let forecast = registry
            .estimated_wait()
            .begin()
            .unwrap()
            .capture_calibration(&worker)
            .unwrap();
        assert_eq!(forecast.baseline_source, BaselineSource::Unknown);
        assert_eq!(forecast.baseline, None);
        assert_eq!(forecast.seconds(), -1.0);
        assert!(forecast.selected_prediction().is_none());
    }

    #[test]
    fn measured_baseline_needs_four_observations_and_material_promotion_isolates_feedback() {
        let (registry, worker) = provisional_calibration_registry(Some(2.0));
        let admission = registry.estimated_wait();
        let old_probe = calibration_probe(admission, &worker);
        let saved = old_probe.calibration.as_ref().unwrap().clone();
        {
            let mut entries = admission.entries.lock();
            let entry = entries.get_mut(worker.url()).unwrap();
            let now = Instant::now();

            for _ in 0..3 {
                push_ttft_sample(&mut entry.low_load_ttfts, 3.0);
            }
            EstimatedWaitAdmission::apply_observations(entry);
            let before = entry.calibration_forecast(admission.config(), now).unwrap();
            assert_eq!(before.baseline_source, BaselineSource::Provisional);
            assert_eq!(before.epoch, saved.epoch);
            push_ttft_sample(&mut entry.low_load_ttfts, 3.0);
            EstimatedWaitAdmission::apply_observations(entry);
            let after = entry.calibration_forecast(admission.config(), now).unwrap();
            assert_eq!(after.baseline, Some(3.0));
            assert_eq!(after.baseline_source, BaselineSource::Measured);
            assert_eq!(after.epoch, saved.epoch + 1);

            assert!(!Arc::ptr_eq(
                &after.calibration_generation,
                &saved.calibration_generation
            ));
            assert_eq!(saved.baseline, Some(2.0));
            // Re-reading the new baseline must not repeatedly reset.
            assert_eq!(
                entry
                    .calibration_forecast(admission.config(), now)
                    .unwrap()
                    .epoch,
                after.epoch
            );
        }
        admission.observe_ttft(old_probe, 22.0, 0.0);
    }

    #[test]
    fn small_baseline_promotion_keeps_history_and_saved_baseline() {
        let (registry, worker) = provisional_calibration_registry(Some(2.0));
        let admission = registry.estimated_wait();
        let mut entries = admission.entries.lock();
        let entry = entries.get_mut(worker.url()).unwrap();
        let now = Instant::now();
        let saved = entry.calibration_forecast(admission.config(), now).unwrap();

        entry.sample.as_mut().unwrap().prepared.baseline = Some(2.25);
        let promoted = entry.calibration_forecast(admission.config(), now).unwrap();
        assert_eq!(promoted.baseline_source, BaselineSource::Measured);
        assert_eq!(promoted.baseline, Some(2.25));
        assert_eq!(promoted.epoch, saved.epoch);

        assert_eq!(saved.baseline, Some(2.0));
    }

    #[test]
    fn provisional_baseline_validation_and_material_change_boundaries() {
        assert!(!materially_different_baseline(5.0, 6.0));
        assert!(materially_different_baseline(5.0, 6.000_001));
        assert!(!materially_different_baseline(5.0, 4.0));
        assert!(materially_different_baseline(5.0, 3.999_999));
        assert!(!materially_different_baseline(0.1, 0.2));
        assert!(materially_different_baseline(0.1, 0.200_001));
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(EstimatedWaitConfig {
                estimated_wait_provisional_baseline_secs: Some(invalid),
                ..config()
            }
            .validate()
            .is_err());
        }
        assert!(EstimatedWaitConfig {
            estimated_wait_provisional_baseline_secs: Some(0.2),
            ..config()
        }
        .validate()
        .is_ok());
        assert_eq!(
            EstimatedWaitConfig::default().estimated_wait_provisional_baseline_secs,
            None
        );
    }

    #[test]
    fn calibrated_verdict_never_uses_floor_promoted_native_readiness() {
        let (registry, worker) = calibration_registry(false);
        let mut entries = registry.estimated_wait().entries.lock();
        let entry = entries.get_mut(worker.url()).unwrap();
        entry.sample.as_mut().unwrap().prepared.capacity_ready = true;
        let forecast = entry
            .calibration_forecast(registry.estimated_wait().config(), Instant::now())
            .unwrap();
        assert!(forecast.raw_ready);
        assert!(!forecast.native_capacity_ready);
        assert!(forecast.selected_prediction().is_none());
    }

    #[test]
    fn excess_calibration_placement_captures_before_dispatch_credit() {
        let (registry, worker) = calibration_registry(true);
        let mut report = load(Some(100), 2, 100.0, 0.0);
        report.loads[0].low_load_ttft_seconds = Some(0.2);
        report.loads[0].prefill_capacity_learning_active = Some(false);
        report.loads[0].prefill_capacity_sample_count = Some(8);
        publish_load(registry.estimated_wait(), &worker, &report);
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let capture = Mutex::new(None);
        let selected = placement::select_from(
            &registry,
            &policies,
            "m",
            std::slice::from_ref(&worker),
            PlacementInputs {
                tokens: Some(&[0; 100]),
                calibration: Some(&capture),
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(selected.url(), worker.url());
        let saved = capture.into_inner().unwrap();
        assert_eq!(saved.raw_extra, 1.0);
        assert!(saved.native_capacity_ready);
        assert!(saved.queue_extra > 0.0);
        assert_eq!(
            registry.estimated_wait().entries.lock()[worker.url()].total_dispatched,
            100
        );
        assert_eq!(
            registry.estimated_wait().entries.lock()[worker.url()]
                .sample
                .as_ref()
                .unwrap()
                .prepared
                .excess_seconds(100),
            2.0
        );
        let mut observer = StreamTtftObserver::new(
            &registry,
            worker.url(),
            Instant::now() - Duration::from_secs(20),
        )
        .with_calibration(Some(saved));
        observer.push(b"data: {\"choices\":[{\"text\":\"first text\"}]}\n\n");
    }

    #[test]
    fn calibration_drain_requires_two_fresh_empty_polls_and_invalidates_old_forecasts() {
        let (registry, worker) = calibration_registry(true);
        let admission = registry.estimated_wait();
        for _ in 0..4 {
            let probe = calibration_probe(admission, &worker);
            admission.observe_ttft(probe, 20.2, 0.0);
        }
        let old = admission
            .begin()
            .unwrap()
            .capture_calibration(&worker)
            .unwrap();
        let mut empty = load(Some(0), 0, 100.0, 0.0);
        // Match native capacity profile; continuing decode does not prevent queue drain.
        empty.loads[0].prefill_capacity_sample_count = Some(8);
        empty.loads[0].num_running_reqs = 10;
        let pending = admission.ttft_probe(worker.url()).unwrap();
        publish_load(admission, &worker, &empty);
        publish_load(admission, &worker, &empty);
        assert_eq!(
            admission.entries.lock()[worker.url()].calibration_epoch,
            old.epoch
        );
        admission.finish_ttft_probe(&pending);
        publish_load(admission, &worker, &empty);
        let published = admission.entries.lock()[worker.url()]
            .last_published
            .unwrap();
        let duplicate = admission.poll_started(&worker);
        admission.publish(&worker, Some(&empty), duplicate, published);
        assert_eq!(
            admission.entries.lock()[worker.url()].calibration_epoch,
            old.epoch
        );
        // A failed report breaks confirmation rather than proving drain.
        let stamp = admission.poll_started(&worker);
        admission.publish(&worker, None, stamp, Instant::now());
        publish_load(admission, &worker, &empty);
        assert_eq!(
            admission.entries.lock()[worker.url()].calibration_epoch,
            old.epoch
        );
        publish_load(admission, &worker, &empty);
        let reset = admission
            .begin()
            .unwrap()
            .capture_calibration(&worker)
            .unwrap();
        assert_eq!(reset.epoch, old.epoch + 1);

        publish_load(admission, &worker, &empty);
        assert_eq!(
            admission.entries.lock()[worker.url()].calibration_epoch,
            reset.epoch
        );
        // Selection can race observer registration; a saved pre-drain forecast
        // must not train the new epoch even if its observer is registered later.
        let mut late = admission.ttft_probe(worker.url()).unwrap();
        late.calibration = Some(old);
        admission.observe_ttft(late, 40.2, 0.0);
    }

    #[test]
    fn calibration_drain_does_not_use_stale_polls_or_hide_dispatch_during_poll() {
        let (registry, worker) = calibration_registry(true);
        let admission = registry.estimated_wait();
        let mut empty = load(Some(0), 0, 100.0, 0.0);
        empty.loads[0].prefill_capacity_sample_count = Some(8);
        publish_load(admission, &worker, &empty);
        let stamp = admission.poll_started(&worker);
        admission.begin().unwrap().credit(&worker, None);
        admission.publish(&worker, Some(&empty), stamp, Instant::now());
        assert_eq!(
            admission.entries.lock()[worker.url()].calibration_empty_polls,
            0
        );
        {
            let mut entries = admission.entries.lock();
            entries.get_mut(worker.url()).unwrap().last_published = None;
        }
        let stamp = admission.poll_started(&worker);
        admission.publish(
            &worker,
            Some(&empty),
            stamp,
            Instant::now() - Duration::from_secs(60),
        );
        assert_eq!(
            admission.entries.lock()[worker.url()].calibration_empty_polls,
            0
        );
    }

    #[test]
    fn estimated_wait_late_old_poll_preserves_replacement_sample() {
        let registry = WorkerRegistry::new();
        let admission = registry.estimated_wait();
        admission.configure(config());
        let old = worker("http://a:1");
        let id = registry.register(old.clone()).unwrap();
        publish(admission, &old, 0);
        let replacement = worker(old.url());
        assert!(registry.replace(&id, replacement.clone()));
        for reset in [false, true] {
            if reset {
                admission.clear();
                assert!(admission.poll_started(&old).is_none());
            }
            publish(admission, &replacement, 200);
            assert!(admission.poll_started(&old).is_none());
            admission.begin().unwrap().credit(&old, Some(&[0; 100]));
            assert!(admission
                .begin()
                .unwrap()
                .check(std::slice::from_ref(&replacement), "m")
                .is_err());
            let entries = admission.entries.lock();
            assert_eq!(entries[replacement.url()].total_dispatched, 0);
        }
    }

    #[test]
    fn estimated_wait_missing_http_tokens_uses_configured_proxy() {
        let config = EstimatedWaitConfig {
            estimated_wait_queue_tokens_per_request: 100,
            ..config()
        };
        let report = WorkerMonitor::decode_native_loads(
            serde_json::json!({"loads":[{"num_waiting_reqs":10,"token_usage":0.0,"gen_throughput":100.0}]}),
        ).unwrap();
        assert_eq!(config.score(&report, 0), Some((10.0, true, true)));
        let no_proxy = EstimatedWaitConfig {
            estimated_wait_queue_tokens_per_request: 0,
            ..config.clone()
        };
        assert!(no_proxy.score(&report, 0).is_none());
        for availability in [None, Some(true), Some(false)] {
            let mut value = serde_json::json!({"loads":[{
                "num_waiting_reqs":10,"token_usage":0.0,"gen_throughput":100.0,
                "num_waiting_uncached_tokens":0
            }]});
            if let Some(available) = availability {
                value["loads"][0]["num_waiting_uncached_tokens_available"] = available.into();
            }
            let report = WorkerMonitor::decode_native_loads(value).unwrap();
            let expected = if availability == Some(false) {
                (10.0, true, true)
            } else {
                (0.0, false, true)
            };
            assert_eq!(config.score(&report, 0), Some(expected));
        }
    }

    #[test]
    fn qualified_prefill_rate_replaces_output_generation_rate() {
        let config = EstimatedWaitConfig {
            estimated_wait_queue_tokens_per_request: 100,
            estimated_wait_fallback_prefill_throughput: 10.0,
            estimated_wait_prompt_size_prior_samples: 0,
            ..config()
        };
        let mut report = load(None, 2, 50.0, 0.0);
        report.loads[0].prefill_throughput = Some(200.0);
        report.loads[0].learned_prefill_capacity = None;
        report.loads[0].prefill_capacity_learning_active = Some(true);
        report.loads[0].avg_request_prefill_kv_computed_tokens = Some(1000.0);
        report.loads[0].prefill_size_sample_count = Some(10);
        assert_eq!(config.score(&report, 0), Some((10.0, true, false)));
    }

    #[test]
    fn generation_rate_does_not_override_learned_prefill_capacity() {
        let config = EstimatedWaitConfig {
            estimated_wait_fallback_prefill_throughput: 2000.0,
            ..config()
        };
        for vllm in [false, true] {
            for generation in [0.088, 1999.0, 2000.0, 4000.0] {
                let mut report = load(Some(4000), 0, generation, 0.0);
                report.loads[0].prefill_throughput = Some(20_000.0);
                report.loads[0].learned_prefill_capacity = Some(10_000.0);
                if vllm {
                    report.loads[0].prefill_capacity_learning_active = Some(false);
                    report.loads[0].learned_prefill_capacity = Some(10_000.0);
                }
                assert_eq!(config.score(&report, 0), Some((0.4, false, false)));
            }
        }
    }

    #[test]
    fn prefill_floor_applies_to_aggregate_not_each_rank() {
        let config = EstimatedWaitConfig {
            estimated_wait_fallback_prefill_throughput: 2000.0,
            ..config()
        };
        let mut report = load(Some(4000), 0, 1500.0, 0.0);
        report.loads.push(SchedulerLoadSnapshot {
            gen_throughput: 2500.0,
            learned_prefill_capacity: Some(2500.0),
            ..Default::default()
        });
        assert_eq!(config.score(&report, 0), Some((1.0, false, false)));
        report.loads[1].learned_prefill_capacity = None;
        assert_eq!(config.score(&report, 0), Some((2.0, false, true)));
    }

    #[test]
    fn sparse_vllm_rate_cannot_collapse_calibrated_capacity() {
        let config = EstimatedWaitConfig {
            estimated_wait_fallback_prefill_throughput: 2000.0,
            ..config()
        };
        let mut report = load(None, 0, 2.5, 0.0);
        report.loads[0].prefill_throughput = Some(10.0);
        report.loads[0].avg_request_prefill_kv_computed_tokens = Some(20.0);
        report.loads[0].prefill_capacity_learning_active = Some(false);

        assert_eq!(config.score(&report, 1024), Some((0.512, true, true)));
    }

    #[test]
    fn cold_vllm_histogram_uses_configured_prompt_prior() {
        let mut report = load(None, 1, 100.0, 0.0);
        report.loads[0].avg_request_prefill_kv_computed_tokens = Some(20.0);
        report.loads[0].prefill_capacity_learning_active = Some(false);

        assert_eq!(config().score(&report, 0), Some((1.0, true, false)));
    }

    #[test]
    fn tiny_vllm_observation_is_blended_with_prompt_prior() {
        let mut report = load(None, 1, 100.0, 0.0);
        report.loads[0].median_request_prefill_kv_computed_tokens = Some(20.0);
        report.loads[0].prefill_size_sample_count = Some(1);
        let score = config().score(&report, 0).unwrap().0;
        let expected_tokens = (100.0 * 32.0 + 20.0) / 33.0;
        assert!((score - expected_tokens / 100.0).abs() < 1e-10);
    }

    #[test]
    fn representative_vllm_observations_converge_toward_median() {
        let mut report = load(None, 1, 100.0, 0.0);
        report.loads[0].median_request_prefill_kv_computed_tokens = Some(80.0);
        report.loads[0].prefill_size_sample_count = Some(320);
        let score = config().score(&report, 0).unwrap().0;
        let expected_tokens = (100.0 * 32.0 + 80.0 * 320.0) / 352.0;
        assert!((score - expected_tokens / 100.0).abs() < 1e-10);
        assert!((expected_tokens - 80.0).abs() < 2.0);
    }

    #[test]
    fn dispatch_factor_changes_only_dispatch_work() {
        let state = load(Some(300), 0, 100.0, 0.5);
        let legacy = config().score(&state, 100).unwrap().0;
        assert!((legacy - 4.0).abs() < f64::EPSILON);

        let calibrated = EstimatedWaitConfig {
            estimated_wait_dispatch_blocking_factor: 0.25,
            ..config()
        };
        // (300 + 0.25 * 100) / 100
        assert!((calibrated.score(&state, 100).unwrap().0 - 3.25).abs() < 1e-10);
    }

    #[test]
    fn low_load_baseline_is_added_once_but_not_required_for_native_extra_delay() {
        let mut report = load(Some(300), 0, 100.0, 0.0);
        report.loads[0].low_load_ttft_seconds = Some(0.25);
        report.loads[0].prefill_capacity_learning_active = Some(false);
        let prepared = config().prepare(&report).unwrap();
        assert_eq!(prepared.seconds(100), 4.25);
        assert!(prepared.capacity_ready);
        report.loads[0].low_load_ttft_seconds = None;
        let uncalibrated = config().prepare(&report).unwrap();
        assert_eq!(uncalibrated.seconds(100), 4.0);
        assert!(uncalibrated.capacity_ready);
        assert_eq!(
            prepared.excess_seconds(100),
            uncalibrated.excess_seconds(100)
        );
    }

    #[test]
    fn measured_active_prefill_is_not_discarded_during_capacity_warmup() {
        let mut report = load(Some(3_000), 3, 1_000.0, 0.0);
        report.loads[0].low_load_ttft_seconds = Some(0.25);
        report.loads[0].learned_prefill_capacity = None;
        report.loads[0].prefill_capacity_learning_active = Some(true);
        report.loads[0].prefill_throughput = Some(1_000.0);
        let mut prepared = config().prepare(&report).unwrap();
        prepared.observed_delay_available = true;
        prepared.observed_delay_floor = 1.0;
        assert!(!prepared.capacity_learned_ready);
        assert!(!prepared.capacity_ready);
        assert!(!prepared.measured_delay_only());
        // Preserve the measured 3,000 / 1,000 queue estimate. The readiness
        // gate still prevents this warm-up estimate from rejecting traffic.
        assert_eq!(prepared.seconds(0), 3.25);

        report.loads[0].prefill_throughput = None;
        let mut fallback = config().prepare(&report).unwrap();
        fallback.observed_delay_available = true;
        fallback.observed_delay_floor = 1.0;
        assert!(fallback.measured_delay_only());
        // An unmeasured default must not inflate the observation to 30 s.
        assert_eq!(fallback.seconds(0), 1.25);
        assert!(!fallback.capacity_ready);
    }

    #[test]
    fn total_ttft_budget_includes_intrinsic_low_load_latency() {
        let admission = EstimatedWaitAdmission::default();
        admission.configure(EstimatedWaitConfig {
            max_estimated_wait_secs: Some(5.0),
            ..config()
        });
        let worker = worker("http://slow-baseline:1");
        let mut report = load(Some(0), 0, 100.0, 0.0);
        report.loads[0].low_load_ttft_seconds = Some(6.0);
        report.loads[0].prefill_capacity_learning_active = Some(false);
        publish_load(&admission, &worker, &report);
        assert!(admission
            .begin()
            .unwrap()
            .check(std::slice::from_ref(&worker), "m")
            .is_err());
        report.loads[0].num_waiting_uncached_tokens = 600;
        report.loads[0].num_waiting_reqs = 6;
        publish_load(&admission, &worker, &report);
        assert!(admission
            .begin()
            .unwrap()
            .check(std::slice::from_ref(&worker), "m")
            .is_err());
    }

    #[test]
    fn observed_delay_floor_shrinks_with_backlog_and_cannot_penalize_a_drained_worker() {
        let admission = EstimatedWaitAdmission::default();
        admission.configure(config());
        let worker = worker("http://queue-feedback:1");
        worker.increment_load();
        publish(&admission, &worker, 0);
        for _ in 0..TTFT_MIN_SAMPLES {
            let probe = admission.ttft_probe(worker.url()).unwrap();
            admission.observe_ttft(probe, 0.25, 0.05);
        }
        let mut report = load(Some(400), 4, 1_000.0, 0.0);
        publish_load(&admission, &worker, &report);
        for _ in 0..TTFT_MIN_SAMPLES {
            let probe = admission.ttft_probe(worker.url()).unwrap();
            admission.observe_ttft(probe, 10.25, 8.05);
        }
        let old = admission.ttft_probe(worker.url()).unwrap();
        let late: Vec<_> = (0..TTFT_MIN_SAMPLES)
            .map(|_| admission.ttft_probe(worker.url()).unwrap())
            .collect();
        {
            let entries = admission.entries.lock();
            let sample = entries[worker.url()].sample.as_ref().unwrap();
            assert_eq!(sample.prepared.effective_prefill_capacity, 1_000.0);
            assert_eq!(sample.prepared.observed_delay_floor, 10.0);
            assert!((sample.prepared.seconds(0) - 10.25).abs() < 1e-9);
        }
        report.loads[0].num_waiting_reqs = 40;
        report.loads[0].num_waiting_uncached_tokens = 4_000;
        publish_load(&admission, &worker, &report);
        assert_eq!(
            admission.entries.lock()[worker.url()]
                .sample
                .as_ref()
                .unwrap()
                .prepared
                .observed_delay_floor,
            10.0
        );
        report.loads[0].num_waiting_reqs = 1;
        report.loads[0].num_waiting_uncached_tokens = 100;
        publish_load(&admission, &worker, &report);
        assert_eq!(
            admission.entries.lock()[worker.url()]
                .sample
                .as_ref()
                .unwrap()
                .prepared
                .observed_delay_floor,
            2.5
        );
        // Late responses must retain their larger dispatch-time queue as the
        // reference; they cannot reinflate a now smaller queue's correction.
        for probe in late {
            admission.observe_ttft(probe, 10.25, 8.05);
        }
        assert_eq!(
            admission.entries.lock()[worker.url()]
                .sample
                .as_ref()
                .unwrap()
                .prepared
                .observed_delay_floor,
            2.5
        );
        // The prior stream is no longer awaiting text. Decode can remain
        // active after drain; its old delays must not reject new hits.
        admission.finish_ttft_probe(&old);
        report.loads[0].num_waiting_reqs = 0;
        report.loads[0].num_waiting_uncached_tokens = 0;
        report.loads[0].num_running_reqs = 20;
        publish_load(&admission, &worker, &report);
        {
            let entries = admission.entries.lock();
            let sample = entries[worker.url()].sample.as_ref().unwrap();
            assert_eq!(sample.prepared.excess_seconds(0), 0.0);
            assert_eq!(sample.prepared.effective_prefill_capacity, 1_000.0);
        }
        report.loads[0].num_running_reqs = 0;
        publish_load(&admission, &worker, &report);
        admission.observe_ttft(old, 99.0, 0.0);
        let entries = admission.entries.lock();
        assert!(entries[worker.url()].observed_delays.is_empty());
        assert!(entries[worker.url()].busy_visibility.is_empty());
        worker.decrement_load();
    }

    #[test]
    fn pending_first_text_telemetry_does_not_invent_remaining_prefill_work() {
        let admission = EstimatedWaitAdmission::default();
        admission.configure(EstimatedWaitConfig {
            estimated_wait_mean_prefill_tokens: 1024,
            ..config()
        });
        let worker = worker("http://active-prefill:1");
        worker.increment_load();
        publish(&admission, &worker, 0);
        for _ in 0..TTFT_MIN_SAMPLES {
            let probe = admission.ttft_probe(worker.url()).unwrap();
            admission.observe_ttft(probe, 0.25, 0.05);
        }
        let mut report = load(Some(0), 0, 1_000.0, 0.0);
        report.loads[0].num_running_reqs = 20;
        publish_load(&admission, &worker, &report);
        let probes: Vec<_> = (0..8)
            .map(|_| admission.ttft_probe(worker.url()).unwrap())
            .collect();
        {
            let entries = admission.entries.lock();
            let p = entries[worker.url()].sample.as_ref().unwrap().prepared;
            assert_eq!(p.pending_engine, 8.0);
            assert_eq!(p.excess_seconds(0), 0.0);
        }
        for probe in probes {
            admission.observe_ttft(probe, 0.25, 0.0);
        }
        {
            let entries = admission.entries.lock();
            let p = entries[worker.url()].sample.as_ref().unwrap().prepared;
            assert_eq!(p.pending_engine, 0.0);
            assert_eq!(p.excess_seconds(0), 0.0);
        }
        let probes: Vec<_> = (0..8)
            .map(|_| admission.ttft_probe(worker.url()).unwrap())
            .collect();
        report.loads[0].num_running_reqs = 0;
        publish_load(&admission, &worker, &report);
        {
            let entries = admission.entries.lock();
            let p = entries[worker.url()].sample.as_ref().unwrap().prepared;
            assert_eq!(p.pending_engine, 0.0);
            assert_eq!(p.excess_seconds(0), 0.0);
        }
        for probe in probes {
            admission.finish_ttft_probe(&probe);
        }
        assert!(admission.entries.lock()[worker.url()]
            .pending_first_text
            .is_empty());
        worker.decrement_load();
    }

    #[test]
    fn cancelled_or_dropped_stream_observers_release_pending_prefill() {
        let registry = Arc::new(WorkerRegistry::new());
        let admission = registry.estimated_wait();
        admission.configure(config());
        let worker = worker("http://observer-lifecycle:1");
        registry.register(Arc::clone(&worker));
        publish(admission, &worker, 0);
        let mut observer = StreamTtftObserver::new(&registry, worker.url(), Instant::now());
        assert_eq!(
            admission.entries.lock()[worker.url()]
                .pending_first_text
                .len(),
            1
        );
        observer.cancel();
        assert!(admission.entries.lock()[worker.url()]
            .pending_first_text
            .is_empty());
        let observer = StreamTtftObserver::new(&registry, worker.url(), Instant::now());
        assert_eq!(
            admission.entries.lock()[worker.url()]
                .pending_first_text
                .len(),
            1
        );
        drop(observer);
        assert!(admission.entries.lock()[worker.url()]
            .pending_first_text
            .is_empty());
    }

    #[test]
    fn native_queue_delay_without_baseline_does_not_prove_total_ttft() {
        let admission = EstimatedWaitAdmission::default();
        admission.configure(EstimatedWaitConfig {
            max_estimated_wait_secs: Some(5.0),
            ..config()
        });
        let worker = worker("http://native-queue-delay:1");
        let mut report = load(Some(60_000), 6, 100.0, 0.0);
        report.loads[0].low_load_ttft_seconds = None;
        report.loads[0].learned_prefill_capacity = None;
        report.loads[0].prefill_capacity_learning_active = Some(false);
        report.loads[0].recent_queue_delay_seconds = Some(0.1);
        report.loads[0].recent_queue_delay_sample_count = Some(4);
        for _ in 0..TTFT_MIN_SAMPLES {
            publish_load(&admission, &worker, &report);
        }
        assert!(admission
            .begin()
            .unwrap()
            .check(std::slice::from_ref(&worker), "m")
            .is_ok());
        assert_eq!(
            admission.entries.lock()[worker.url()]
                .sample
                .as_ref()
                .unwrap()
                .prepared
                .seconds(0),
            0.1
        );
        report.loads[0].recent_queue_delay_seconds = Some(6.0);
        for _ in 0..TTFT_WINDOW_SAMPLES {
            publish_load(&admission, &worker, &report);
        }
        assert!(admission
            .begin()
            .unwrap()
            .check(std::slice::from_ref(&worker), "m")
            .is_ok());
        assert_eq!(
            admission.entries.lock()[worker.url()]
                .sample
                .as_ref()
                .unwrap()
                .prepared
                .seconds(0),
            6.0
        );
        report.loads[0].num_waiting_reqs = 0;
        report.loads[0].num_waiting_uncached_tokens = 0;
        publish_load(&admission, &worker, &report);
        assert!(admission
            .begin()
            .unwrap()
            .check(std::slice::from_ref(&worker), "m")
            .is_ok());
    }

    #[test]
    fn delayed_streams_cannot_reenter_after_native_idle_precedes_pipeline_drain() {
        let admission = EstimatedWaitAdmission::default();
        admission.configure(config());
        let worker = worker("http://delayed-drain:1");
        publish_load(&admission, &worker, &load(Some(400), 4, 1_000.0, 0.0));
        let mut probes: Vec<_> = (0..4)
            .map(|_| admission.ttft_probe(worker.url()).unwrap())
            .collect();
        publish(&admission, &worker, 0);
        while probes.len() > 1 {
            admission.finish_ttft_probe(&probes.pop().unwrap());
        }
        publish(&admission, &worker, 0);
        publish_load(&admission, &worker, &load(Some(400), 4, 1_000.0, 0.0));
        admission.observe_ttft(probes.pop().unwrap(), 20.0, 0.0);
        assert!(admission.entries.lock()[worker.url()]
            .observed_delays
            .is_empty());
    }

    #[test]
    fn sse_observer_handles_fragmented_reasoning_and_ignores_empty_role() {
        let started = Instant::now();
        let mut observer = StreamTtftObserver {
            registry: Weak::new(),
            probe: None,
            started,
            first_event: None,
            pending: Vec::new(),
            done: false,
            worker_url: "http://observer-test".to_owned(),
            request_id: None,
            feature_identity: None,
            feature_resolved: false,
            feature_deadline: None,
            native_deadline_pending: false,
            metric_observation: None,
        };
        assert_eq!(
            observer.scan(
                b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
                started + Duration::from_secs(1)
            ),
            None
        );
        assert_eq!(
            observer.scan(
                b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hel",
                started + Duration::from_secs(2)
            ),
            None
        );
        assert_eq!(
            observer.scan(b"lo\"}}]}\n\n", started + Duration::from_secs(3)),
            Some((3.0, 2.0))
        );
        assert_eq!(
            observer.scan(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"later\"}}]}\n\n",
                started + Duration::from_secs(4)
            ),
            None
        );
    }

    #[test]
    fn sse_observer_does_not_train_after_backpressure_or_oversized_events() {
        let started = Instant::now();
        let mut observer = StreamTtftObserver {
            registry: Weak::new(),
            probe: None,
            started,
            first_event: None,
            pending: Vec::new(),
            done: false,
            worker_url: "http://observer-test".to_owned(),
            request_id: None,
            feature_identity: None,
            feature_resolved: false,
            feature_deadline: None,
            native_deadline_pending: false,
            metric_observation: None,
        };
        observer.cancel();
        assert_eq!(
            observer.scan(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"text\"}}]}\n",
                started
            ),
            None
        );
        observer.done = false;
        assert_eq!(
            observer.scan(&vec![b'x'; MAX_TTFT_EVENT_BYTES + 1], started),
            None
        );
        assert!(observer.done && observer.pending.is_empty());
    }

    #[tokio::test]
    async fn http_relay_observes_first_text_before_hook_output_or_stream_finish() {
        use bytes::Bytes;

        use crate::{
            routers::http::{
                stream_failure::RelaySource,
                stream_relay::{relay_streaming_body_observed, ChunkMode, ChunkStep, Relay},
            },
            worker::WorkerLoadGuard,
        };
        let (registry, worker) = calibration_registry(true);
        let guard = WorkerLoadGuard::new(worker.clone(), None);
        let mut report = load(Some(0), 0, 100.0, 0.0);
        report.loads[0].low_load_ttft_seconds = None;
        publish_load(registry.estimated_wait(), &worker, &report);
        let observer = StreamTtftObserver::new(&registry, worker.url(), Instant::now());
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, reqwest::Error>>(4);
        let response = relay_streaming_body_observed(
            tokio_stream::wrappers::ReceiverStream::new(rx),
            Relay {
                status: StatusCode::OK,
                upstream_sse: true,
                source: RelaySource::new("m", "/v1/chat/completions", worker.clone()),
                guard,
                headers: http::HeaderMap::new(),
                // Hold all downstream output: observation must precede relay transforms.
                mode: ChunkMode::Hook(Box::new(|_| ChunkStep {
                    bytes: Bytes::new(),
                    last: false,
                })),
            },
            Some(observer),
        );
        tx.send(Ok(Bytes::from_static(
            b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
        )))
        .await
        .unwrap();
        tokio::task::yield_now().await;
        assert!(registry.estimated_wait().entries.lock()[worker.url()]
            .low_load_ttfts
            .is_empty());
        tx.send(Ok(Bytes::from_static(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"first\"}}]}\n\n",
        )))
        .await
        .unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if registry.estimated_wait().entries.lock()[worker.url()]
                    .low_load_ttfts
                    .len()
                    == 1
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("first text is observed while the upstream stream remains open");
        assert!(!tx.is_closed());
        drop(response);
        drop(tx);
    }

    #[test]
    fn router_baseline_and_visibility_use_shared_rules_and_reset_with_worker() {
        let admission = EstimatedWaitAdmission::default();
        admission.configure(config());
        let worker = worker("http://measured:1");
        admission.worker_added(&worker);
        worker.increment_load();
        publish(&admission, &worker, 0);
        for _ in 0..TTFT_MIN_SAMPLES {
            let probe = admission.ttft_probe(worker.url()).unwrap();
            admission.observe_ttft(probe, 0.25, 0.05);
        }
        publish(&admission, &worker, 100);
        for _ in 0..TTFT_MIN_SAMPLES {
            let probe = admission.ttft_probe(worker.url()).unwrap();
            admission.observe_ttft(probe, 4.0, 3.0);
        }
        publish(&admission, &worker, 100);
        let entries = admission.entries.lock();
        let sample = entries[worker.url()].sample.as_ref().unwrap();
        assert!((sample.prepared.seconds(0) - 4.0).abs() < 1e-9);
        assert!((sample.prepared.visibility_delay - 2.95).abs() < 1e-9);
        drop(entries);
        let old = admission.ttft_probe(worker.url()).unwrap();
        admission.evict(&worker);
        publish(&admission, &worker, 0);
        admission.observe_ttft(old, 99.0, 0.0);
        assert!(admission.entries.lock()[worker.url()]
            .low_load_ttfts
            .is_empty());
        worker.decrement_load();
    }

    #[test]
    fn overlapping_dispatch_is_not_a_low_load_baseline() {
        let admission = EstimatedWaitAdmission::default();
        admission.configure(config());
        let worker = worker("http://overlap:1");
        admission.worker_added(&worker);
        worker.increment_load();
        publish(&admission, &worker, 0);
        let probe = admission.ttft_probe(worker.url()).unwrap();
        admission.begin().unwrap().credit(&worker, Some(&[1; 10]));
        admission.observe_ttft(probe, 12.0, 0.0);
        assert!(admission.entries.lock()[worker.url()]
            .low_load_ttfts
            .is_empty());
        worker.decrement_load();
    }

    #[test]
    fn uncached_prompt_mean_is_preferred_to_histogram_bucket_median() {
        let mut report = load(None, 2, 100.0, 0.0);
        report.loads[0].avg_request_prefill_kv_computed_tokens = Some(50.0);
        report.loads[0].median_request_prefill_kv_computed_tokens = Some(200.0);
        report.loads[0].prefill_size_sample_count = Some(32);
        // The 100-token prior and 50-token observed mean have equal weight.
        assert_eq!(config().score(&report, 0), Some((1.5, true, false)));
    }

    #[test]
    fn learned_prefill_capacity_is_used_after_learning() {
        let mut report = load(None, 0, 10.0, 0.0);
        report.loads[0].avg_request_prefill_kv_computed_tokens = Some(100.0);
        report.loads[0].prefill_capacity_learning_active = Some(false);
        assert_eq!(config().score(&report, 100), Some((1.0, true, true)));

        report.loads[0].learned_prefill_capacity = Some(40.0);
        report.loads[0].prefill_capacity_sample_count = Some(8);
        assert_eq!(config().score(&report, 100), Some((1.0, true, true)));

        report.loads[0].learned_prefill_capacity = Some(200.0);
        assert_eq!(config().score(&report, 100), Some((0.5, true, false)));
    }

    #[test]
    fn unlearned_prefill_rate_requires_backlog_to_override_fallback() {
        let mut report = load(None, 2, 10.0, 0.0);
        report.loads[0].avg_request_prefill_kv_computed_tokens = Some(100.0);
        report.loads[0].prefill_throughput = Some(200.0);
        report.loads[0].prefill_capacity_learning_active = Some(true);
        report.loads[0].learned_prefill_capacity = None;

        assert_eq!(config().score(&report, 0), Some((1.0, true, false)));

        // Sparse demand does not train or replace a capacity estimate.
        report.loads[0].prefill_capacity_learning_active = Some(false);
        assert_eq!(config().score(&report, 0), Some((2.0, true, true)));
    }

    #[test]
    fn formula_uses_exact_tokens_and_live_throughput_without_rank_kv() {
        let config = config();
        let mut state = load(Some(100), 80, 50.0, 0.25);
        state.loads.push(SchedulerLoadSnapshot {
            num_waiting_uncached_tokens: 200,
            gen_throughput: 50.0,
            learned_prefill_capacity: Some(50.0),
            token_usage: 0.75,
            ..Default::default()
        });
        assert_eq!(config.score(&state, 100), Some((4.0, false, false)));
    }

    #[test]
    fn admission_discounts_dispatched_work_independently_of_kv_usage() {
        let config = EstimatedWaitConfig {
            estimated_wait_dispatch_blocking_factor: 0.05,
            ..config()
        };

        // The formula never reads KV, even when that diagnostic is invalid.
        for usage in [0.0, 0.5, 0.999, 1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                config.score(&load(Some(0), 0, 100.0, usage), 1_000),
                Some((0.5, false, false))
            );
        }
    }

    #[test]
    fn exact_zero_is_preserved_and_proxy_is_explicit_per_rank() {
        let mut config = config();
        assert_eq!(
            config.score(&load(Some(0), 10, 0.0, 0.0), 0),
            Some((0.0, false, true))
        );
        assert!(config.score(&load(None, 0, 100.0, 0.0), 0).is_none());
        config.estimated_wait_queue_tokens_per_request = 100;
        let mut state = load(Some(0), 10, 50.0, 0.0);
        state.loads.push(SchedulerLoadSnapshot {
            num_waiting_uncached_tokens: 0,
            num_waiting_uncached_tokens_available: Some(false),
            num_waiting_reqs: 2,
            gen_throughput: 50.0,
            ..Default::default()
        });
        assert_eq!(config.score(&state, 0), Some((2.0, true, true)));
    }

    #[test]
    fn invalid_data_fails_open_and_invalid_throughput_uses_fallback() {
        for rate in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                config().score(&load(Some(200), 0, rate, 0.0), 0),
                Some((2.0, false, true))
            );
        }
        assert!(config().score(&WorkerLoadResponse::default(), 0).is_none());
        let mut partial = load(Some(0), 0, 100.0, 0.0);
        partial.dp_rank_count = 2;
        assert!(config().score(&partial, 0).is_none());
        assert!(config().score(&load(Some(-1), 0, 100.0, 0.0), 0).is_none());
        assert!(config().score(&load(None, -1, 100.0, 0.0), 0).is_none());
        assert_eq!(
            config().score(&load(Some(0), 0, 100.0, f64::NAN), 0),
            Some((0.0, false, false))
        );
    }

    #[test]
    fn registering_disabled_worker_does_not_freeze_default_configuration() {
        let registry = Arc::new(WorkerRegistry::new());
        let ordinary = worker("http://configured-later:8000");
        registry.register(ordinary.clone()).unwrap();
        assert!(registry.estimated_wait().config.get().is_none());
        registry.estimated_wait().configure(config());
        assert!(registry.estimated_wait().needs_load(&ordinary));
        assert!(registry.estimated_wait().poll_started(&ordinary).is_some());
    }

    #[test]
    fn unconfigured_workers_have_no_estimator_state_or_stream_observer() {
        let registry = Arc::new(WorkerRegistry::new());
        registry
            .estimated_wait()
            .configure(EstimatedWaitConfig::default());
        let ordinary = worker("http://ordinary:8000");
        registry.register(ordinary.clone()).unwrap();
        let admission = registry.estimated_wait();
        assert!(admission.entries.lock().is_empty());
        assert!(admission.poll_started(&ordinary).is_none());
        assert!(
            StreamTtftObserver::for_worker(&registry, ordinary.as_ref(), Instant::now()).is_none()
        );

        let configured: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://configured:8000")
                .status(WorkerStatus::Ready)
                .overload(openai_protocol::worker::OverloadUpdate {
                    max_estimated_wait_secs: Some(300.0),
                    ..Default::default()
                })
                .build(),
        );
        registry.register(configured.clone()).unwrap();
        assert!(admission.enabled());
        assert!(admission.native_deadline_enabled());
        assert_eq!(admission.entries.lock().len(), 1);
        assert!(admission.poll_started(&configured).is_some());
        assert!(
            StreamTtftObserver::for_worker(&registry, configured.as_ref(), Instant::now())
                .is_some()
        );
        // An override on a different worker must not start work for this one.
        assert!(admission.poll_started(&ordinary).is_none());
        assert!(
            StreamTtftObserver::for_worker(&registry, ordinary.as_ref(), Instant::now()).is_none()
        );
        assert_eq!(admission.entries.lock().len(), 1);
    }

    #[test]
    fn configuration_rejects_invalid_numbers_and_defaults_to_disabled() {
        let admission = EstimatedWaitAdmission::default();
        admission.configure(EstimatedWaitConfig::default());
        assert!(admission.begin().is_none());
        assert!(!admission.native_deadline_enabled());
        assert_eq!(
            EstimatedWaitConfig::default().estimated_wait_dispatch_blocking_factor,
            0.05
        );
        assert_eq!(
            EstimatedWaitConfig::default().estimated_wait_kv_pressure_weight,
            0.0
        );
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(EstimatedWaitConfig {
                max_estimated_wait_secs: Some(bad),
                ..config()
            }
            .validate()
            .is_err());
            assert!(EstimatedWaitConfig {
                estimated_wait_fallback_prefill_throughput: bad,
                ..config()
            }
            .validate()
            .is_err());
            assert!(EstimatedWaitConfig {
                estimated_wait_max_snapshot_age_secs: bad,
                ..config()
            }
            .validate()
            .is_err());
        }
        for bad in [-1.0, 1.01, f64::NAN, f64::INFINITY] {
            assert!(EstimatedWaitConfig {
                estimated_wait_dispatch_blocking_factor: bad,
                ..config()
            }
            .validate()
            .is_err());
        }
        assert!(EstimatedWaitConfig {
            estimated_wait_mean_prefill_tokens: 0,
            ..config()
        }
        .validate()
        .is_err());
        for weight in [-1.0, 0.15, f64::NAN, f64::INFINITY] {
            assert!(EstimatedWaitConfig {
                estimated_wait_kv_pressure_weight: weight,
                ..config()
            }
            .validate()
            .is_err());
        }
        assert!(config().validate().is_ok());
        let decoded: EstimatedWaitConfig = serde_json::from_str("{}").unwrap();
        assert!(decoded.max_estimated_wait_secs.is_none());
        assert!(decoded.estimated_wait_shadow);
        assert!(decoded.estimated_wait_feature_shadow && decoded.estimated_wait_calibration_shadow);
        let shadow: EstimatedWaitConfig =
            serde_json::from_str(r#"{"estimated_wait_shadow":true}"#).unwrap();
        assert!(shadow.estimated_wait_shadow);
        assert!(shadow.max_estimated_wait_secs.is_none());
        let renamed: EstimatedWaitConfig =
            serde_json::from_str(r#"{"estimated_wait_fallback_prefill_throughput":321.0}"#)
                .unwrap();
        assert_eq!(renamed.estimated_wait_fallback_prefill_throughput, 321.0);
        let interim: EstimatedWaitConfig =
            serde_json::from_str(r#"{"estimated_wait_min_prefill_throughput":222.0}"#).unwrap();
        assert_eq!(interim.estimated_wait_fallback_prefill_throughput, 222.0);
        let legacy: EstimatedWaitConfig =
            serde_json::from_str(r#"{"estimated_wait_default_throughput":123.0}"#).unwrap();
        assert_eq!(legacy.estimated_wait_fallback_prefill_throughput, 123.0);
    }

    #[test]
    fn admission_uses_minimum_and_unknown_workers_prevent_false_rejection() {
        let admission = EstimatedWaitAdmission::default();
        admission.configure(config());
        let a = worker("http://a:1");
        let b = worker("http://b:1");
        let pool = [a.clone(), b.clone()];
        publish(&admission, &a, 200);
        assert!(admission.begin().unwrap().check(&pool, "m").is_ok());
        publish(&admission, &b, 199);
        assert!(admission.begin().unwrap().check(&pool, "m").is_ok());
        publish(&admission, &b, 200);
        let shed = admission.begin().unwrap().check(&pool, "m").unwrap_err();
        assert_eq!(shed.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            shed.headers().get(error::HEADER_X_SMG_ERROR_CODE).unwrap(),
            "worker_overload_protection_shed"
        );
        assert!(shed.headers().contains_key(RETRY_AFTER));
        assert!(!is_retryable_response(&shed));
        publish(&admission, &b, 0);
        for _ in 0..5 {
            b.record_outcome(503);
        }
        assert!(!b.circuit_breaker_can_execute());
        assert!(admission.begin().unwrap().check(&pool, "m").is_err());
        b.set_status(WorkerStatus::NotReady);
        publish(&admission, &b, 0);
        assert!(admission.begin().unwrap().check(&pool, "m").is_err());
        a.set_status(WorkerStatus::NotReady);
        assert!(admission.begin().unwrap().check(&pool, "m").is_ok());
        assert!(admission.begin().unwrap().check(&[], "m").is_ok());
    }

    #[test]
    fn vllm_admission_waits_for_learned_capacity_without_hiding_estimate() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let admission = EstimatedWaitAdmission::default();
            admission.configure(config());
            let worker = worker("http://vllm:1");

            let mut report = load(Some(200), 0, 100.0, 0.0);
            report.loads[0].prefill_capacity_learning_active = Some(false);
            report.loads[0].learned_prefill_capacity = None;
            publish_load(&admission, &worker, &report);

            // The cold estimate is still published and usable for calibration,
            // but it cannot reject until the conservative capacity is learned.
            assert_eq!(config().score(&report, 0), Some((2.0, false, true)));
            assert!(admission
                .begin()
                .unwrap()
                .check(std::slice::from_ref(&worker), "m")
                .is_ok());
            let cold = handle.render();
            assert!(
                cold.contains("smg_estimated_wait_capacity_ready{worker=\"http://vllm:1\"} 0"),
                "{cold}"
            );
            assert!(
                cold.contains("smg_estimated_wait_data_usable{worker=\"http://vllm:1\"} 1"),
                "{cold}"
            );
            assert!(
                cold.contains(
                    "smg_estimated_wait_unknown_total{model=\"m\",reason=\"capacity_unready\"} 1"
                ),
                "{cold}"
            );

            report.loads[0].learned_prefill_capacity = Some(100.0);
            report.loads[0].prefill_capacity_sample_count = Some(8);
            publish_load(&admission, &worker, &report);
            assert_eq!(config().score(&report, 0), Some((2.0, false, false)));
            assert!(admission
                .begin()
                .unwrap()
                .check(std::slice::from_ref(&worker), "m")
                .is_err());
            let ready = handle.render();
            assert!(
                ready.contains("smg_estimated_wait_capacity_ready{worker=\"http://vllm:1\"} 1"),
                "{ready}"
            );
        });
    }

    #[test]
    fn shadow_does_not_count_uncalibrated_vllm_estimate_as_would_reject() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let admission = EstimatedWaitAdmission::default();
            admission.configure(EstimatedWaitConfig {
                estimated_wait_shadow: true,
                ..config()
            });
            let worker = worker("http://vllm:1");
            let mut report = load(Some(200), 0, 100.0, 0.0);
            report.loads[0].prefill_capacity_learning_active = Some(false);
            report.loads[0].learned_prefill_capacity = None;
            publish_load(&admission, &worker, &report);

            assert!(admission
                .begin()
                .unwrap()
                .check(std::slice::from_ref(&worker), "m")
                .is_ok());
            assert!(!handle
                .render()
                .contains("smg_estimated_wait_shadow_rejections_total"));
        });
    }

    #[test]
    fn shadow_counts_only_would_reject_decisions_and_keeps_credit() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let registry = WorkerRegistry::new();
            registry.estimated_wait().configure(EstimatedWaitConfig {
                estimated_wait_shadow: true,
                ..config()
            });
            let a = worker("http://a:1");
            let b = worker("http://b:1");
            let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
            let pool = [a.clone(), b.clone()];
            publish(registry.estimated_wait(), &a, 200);
            // Missing, below-budget, stale and empty pools are not would-rejects.
            assert!(registry
                .estimated_wait()
                .begin()
                .unwrap()
                .check(&pool, "m")
                .is_ok());
            publish(registry.estimated_wait(), &b, 199);
            assert!(registry
                .estimated_wait()
                .begin()
                .unwrap()
                .check(&pool, "m")
                .is_ok());
            publish(registry.estimated_wait(), &b, 200);
            registry
                .estimated_wait()
                .entries
                .lock()
                .get_mut(b.url())
                .unwrap()
                .sample
                .as_mut()
                .unwrap()
                .started = Instant::now() - Duration::from_secs(31);
            assert!(registry
                .estimated_wait()
                .begin()
                .unwrap()
                .check(&pool, "m")
                .is_ok());
            assert!(registry
                .estimated_wait()
                .begin()
                .unwrap()
                .check(&[], "m")
                .is_ok());
            assert!(!handle
                .render()
                .contains("smg_estimated_wait_shadow_rejections_total"));

            // All workers at budget still reach policy selection and reserve credit.
            publish(registry.estimated_wait(), &b, 200);
            for _ in 0..4 {
                assert!(placement::select_from(
                    &registry,
                    &policies,
                    "m",
                    &pool,
                    PlacementInputs::default()
                )
                .unwrap()
                .is_some());
            }
            let entries = registry.estimated_wait().entries.lock();
            assert_eq!(
                entries.values().map(|e| e.total_dispatched).sum::<u64>(),
                400
            );
            drop(entries);
            let rendered = handle.render();
            assert!(
                rendered.contains("smg_estimated_wait_shadow_rejections_total{model=\"m\"} 4"),
                "{rendered}"
            );
            assert!(!rendered.contains("smg_estimated_wait_rejections_total"));

            // Shadow mode does not remove explicitly enforced static shedding.
            // Current upstream otherwise steers an all-vetoed pool by default.
            registry.set_overload_shed(true);
            a.set_overloaded(true);
            b.set_overloaded(true);
            assert!(placement::select_from(
                &registry,
                &policies,
                "m",
                &pool,
                PlacementInputs::default()
            )
            .unwrap()
            .is_none());
            assert!(matches!(
                placement::failure_from(&pool, "m", true),
                PlacementFailure::AllOverloaded(_)
            ));
        });
    }

    #[test]
    fn shadow_requires_a_budget_and_observes_worker_only_overrides() {
        use openai_protocol::worker::OverloadUpdate;

        let registry = WorkerRegistry::new();
        registry.estimated_wait().configure(EstimatedWaitConfig {
            estimated_wait_shadow: true,
            ..Default::default()
        });
        assert!(registry.estimated_wait().begin().is_none());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://override:1")
                .status(WorkerStatus::Ready)
                .overload(OverloadUpdate {
                    max_estimated_wait_secs: Some(2.0),
                    ..Default::default()
                })
                .build(),
        );
        registry.register(worker.clone()).unwrap();
        assert!(registry.estimated_wait().needs_load(&worker));
        publish(registry.estimated_wait(), &worker, 4000);
        assert!(registry
            .estimated_wait()
            .begin()
            .unwrap()
            .check(std::slice::from_ref(&worker), "m")
            .is_ok());
        assert!(registry
            .estimated_wait()
            .entries
            .lock()
            .get(worker.url())
            .unwrap()
            .sample
            .as_ref()
            .unwrap()
            .overloaded
            .unwrap());
    }

    #[test]
    fn concurrent_placements_credit_once_and_recover_on_fresh_poll() {
        let registry = Arc::new(WorkerRegistry::new());
        registry.estimated_wait().configure(config());
        let worker = worker("http://a:1");
        publish(registry.estimated_wait(), &worker, 0);
        let barrier = Arc::new(Barrier::new(16));
        let policies = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
        let tasks: Vec<_> = (0..16)
            .map(|_| {
                let registry = registry.clone();
                let worker = worker.clone();
                let policies = policies.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    placement::select_from(
                        &registry,
                        &policies,
                        "m",
                        &[worker],
                        PlacementInputs::default(),
                    )
                    .is_ok()
                })
            })
            .collect();
        let admitted = tasks
            .into_iter()
            .map(|t| t.join().unwrap())
            .filter(|ok| *ok)
            .count();
        assert_eq!(admitted, 2);
        publish(registry.estimated_wait(), &worker, 0);
        assert!(placement::select_from(
            &registry,
            &policies,
            "m",
            &[worker],
            PlacementInputs::default()
        )
        .unwrap()
        .is_some());
    }

    #[test]
    fn in_flight_poll_retains_new_credit_and_failure_invalidates_snapshot() {
        let admission = EstimatedWaitAdmission::default();
        admission.configure(config());
        let worker = worker("http://a:1");
        publish(&admission, &worker, 0);
        admission.begin().unwrap().credit(&worker, None);
        let stamp = admission.poll_started(&worker);
        admission.begin().unwrap().credit(&worker, Some(&[0; 200]));
        admission.publish(
            &worker,
            Some(&load(Some(0), 0, 100.0, 0.0)),
            stamp,
            Instant::now(),
        );
        assert!(admission
            .begin()
            .unwrap()
            .check(std::slice::from_ref(&worker), "m")
            .is_err());
        publish(&admission, &worker, 0);
        assert!(admission
            .begin()
            .unwrap()
            .check(std::slice::from_ref(&worker), "m")
            .is_ok());
        publish(&admission, &worker, 200);
        let stamp = admission.poll_started(&worker);
        admission.publish(&worker, None, stamp, Instant::now());
        assert!(admission
            .begin()
            .unwrap()
            .check(std::slice::from_ref(&worker), "m")
            .is_ok());
    }

    #[test]
    fn stale_and_evicted_samples_cannot_be_resurrected_by_late_polls() {
        let admission = EstimatedWaitAdmission::default();
        admission.configure(config());
        let old = worker("http://a:1");
        let stamp = admission.poll_started(&old);
        admission.publish(
            &old,
            Some(&load(Some(200), 0, 100.0, 0.0)),
            stamp,
            Instant::now() - Duration::from_secs(31),
        );
        assert!(admission
            .begin()
            .unwrap()
            .check(std::slice::from_ref(&old), "m")
            .is_ok());
        let late = admission.poll_started(&old);
        admission.evict(&old);
        publish(&admission, &old, 200);
        admission.publish(
            &old,
            Some(&load(Some(0), 0, 100.0, 0.0)),
            late,
            Instant::now(),
        );
        assert!(admission
            .begin()
            .unwrap()
            .check(std::slice::from_ref(&old), "m")
            .is_err());
        let late = admission.poll_started(&old);
        let replacement = worker(old.url());
        admission.worker_added(&replacement);
        publish(&admission, &replacement, 200);
        admission.publish(
            &old,
            Some(&load(Some(0), 0, 100.0, 0.0)),
            late,
            Instant::now(),
        );
        admission.evict(&old);
        assert!(admission
            .begin()
            .unwrap()
            .check(&[replacement], "m")
            .is_err());
    }

    #[test]
    fn retired_worker_diagnostics_are_invalidated_without_clearing_replacement() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let admission = EstimatedWaitAdmission::default();
            admission.configure(config());
            let old = worker("http://a:1");
            admission.worker_added(&old);
            publish(&admission, &old, 200);

            let replacement = worker(old.url());
            admission.worker_added(&replacement);
            publish(&admission, &replacement, 100);
            admission.worker_removed(&old);

            let rendered = handle.render();
            assert!(
                rendered.contains("smg_estimated_wait_data_usable{worker=\"http://a:1\"} 1"),
                "{rendered}"
            );

            let assert_invalidated = |rendered: &str| {
                for metric in [
                    "smg_estimated_wait_snapshot_age_seconds",
                    "smg_estimated_wait_capacity_ready",
                ] {
                    assert!(
                        rendered.contains(&format!("{metric}{{worker=\"http://a:1\"}} -1")),
                        "{rendered}"
                    );
                }
                assert!(
                    rendered.contains("smg_estimated_wait_data_usable{worker=\"http://a:1\"} 0"),
                    "{rendered}"
                );
            };

            admission.evict(&replacement);
            assert_invalidated(&handle.render());
            publish(&admission, &replacement, 100);
            admission.clear();
            assert_invalidated(&handle.render());
            publish(&admission, &replacement, 100);
            admission.worker_removed(&replacement);
            assert_invalidated(&handle.render());
        });
    }

    #[test]
    fn per_worker_budget_enables_admission_and_survives_monitor_reset() {
        use openai_protocol::worker::OverloadUpdate;

        let registry = WorkerRegistry::new();
        registry.estimated_wait().configure(EstimatedWaitConfig {
            estimated_wait_shadow: false,
            ..Default::default()
        });
        assert!(registry.estimated_wait().begin().is_none());
        let make_worker = |budget| -> Arc<dyn Worker> {
            Arc::new(
                BasicWorkerBuilder::new("http://override:1")
                    .status(WorkerStatus::Ready)
                    .overload(OverloadUpdate {
                        max_estimated_wait_secs: Some(budget),
                        ..Default::default()
                    })
                    .build(),
            )
        };
        let worker = make_worker(2.0);
        let id = registry.register(worker.clone()).unwrap();
        assert!(registry.estimated_wait().needs_load(&worker));
        publish(registry.estimated_wait(), &worker, 4000);
        assert!(registry
            .estimated_wait()
            .begin()
            .unwrap()
            .check(std::slice::from_ref(&worker), "m")
            .is_err());
        assert!(
            !worker.is_overloaded(),
            "estimated admission must not alter the static veto"
        );
        registry.estimated_wait().clear();
        assert!(
            registry.estimated_wait().enabled(),
            "monitor reset must retain worker configuration"
        );
        assert!(registry
            .estimated_wait()
            .begin()
            .unwrap()
            .check(std::slice::from_ref(&worker), "m")
            .is_ok());
        publish(registry.estimated_wait(), &worker, 200);
        let late = registry.estimated_wait().poll_started(&worker);
        let replacement = make_worker(4.0);
        assert!(registry.replace(&id, replacement.clone()));
        publish(registry.estimated_wait(), &replacement, 200);
        registry.estimated_wait().publish(
            &worker,
            Some(&load(Some(1000), 0, 100.0, 0.0)),
            late,
            Instant::now(),
        );
        assert!(registry
            .estimated_wait()
            .begin()
            .unwrap()
            .check(std::slice::from_ref(&replacement), "m")
            .is_ok());
        registry.remove(&id).unwrap();
        assert!(registry.estimated_wait().begin().is_none());
    }

    #[test]
    fn worker_budget_overrides_gateway_and_unprotected_peer_fails_open() {
        use openai_protocol::worker::OverloadUpdate;

        let registry = WorkerRegistry::new();
        registry.estimated_wait().configure(config());
        let protected: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://override:2")
                .status(WorkerStatus::Ready)
                .overload(OverloadUpdate {
                    max_estimated_wait_secs: Some(4.0),
                    ..Default::default()
                })
                .build(),
        );
        registry.register(protected.clone()).unwrap();
        publish(registry.estimated_wait(), &protected, 300);
        assert!(registry
            .estimated_wait()
            .begin()
            .unwrap()
            .check(std::slice::from_ref(&protected), "m")
            .is_ok());
        publish(registry.estimated_wait(), &protected, 400);
        assert!(registry
            .estimated_wait()
            .begin()
            .unwrap()
            .check(std::slice::from_ref(&protected), "m")
            .is_err());

        let standalone = WorkerRegistry::new();
        standalone.register(protected.clone()).unwrap();
        publish(standalone.estimated_wait(), &protected, 400);
        let unprotected = worker("http://unprotected:1");
        assert!(!standalone.estimated_wait().needs_load(&unprotected));
        assert!(standalone
            .estimated_wait()
            .begin()
            .unwrap()
            .check(&[protected, unprotected], "m")
            .is_ok());
    }

    #[test]
    fn later_failed_poll_fences_an_older_successful_poll() {
        let admission = EstimatedWaitAdmission::default();
        admission.configure(config());
        let worker = worker("http://ordering:1");
        let started = Instant::now() - Duration::from_secs(1);
        let old = admission.poll_started(&worker);
        let new = admission.poll_started(&worker);
        admission.publish(&worker, None, new, Instant::now());
        admission.publish(&worker, Some(&load(Some(200), 0, 100.0, 0.0)), old, started);
        assert!(admission.begin().unwrap().check(&[worker], "m").is_ok());
    }

    #[test]
    fn native_json_preserves_legacy_required_tokens() {
        let absent: SchedulerLoadSnapshot = serde_json::from_str("{}").unwrap();
        let empty: SchedulerLoadSnapshot =
            serde_json::from_str(r#"{"num_waiting_uncached_tokens":0}"#).unwrap();
        assert_eq!(absent.num_waiting_uncached_tokens, 0);
        assert_eq!(absent.num_waiting_uncached_tokens_available, None);
        assert_eq!(empty.num_waiting_uncached_tokens, 0);
        assert_eq!(
            serde_json::to_value(&absent).unwrap()["num_waiting_uncached_tokens"],
            0
        );
    }
}
