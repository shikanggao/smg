//! Calibrated worker-backlog admission, independent of the routing policy.
//!
//! The ledger lock spans admission and selection/credit. Polls capture a
//! watermark before I/O so admissions during a poll survive its publication.
//!
//! `--estimated-wait-calibration-shadow` adds diagnostic excess-TTFT learning:
//! T* = B + g * (T_raw - B), including the raw observed-delay floor in the
//! scaled term. A per-worker weighted median fits g from the last 32 usable
//! first-text observations (minimum 4, otherwise g=1), weighted by saved raw
//! extra delay. Native-capacity unready/ready states have separate histories.
//! The applied scale approaches that median target at at most 0.1 absolute
//! scale units per second. New targets start at the old trajectory's current
//! value, so callback bursts and prior idle time cannot cause scale jumps.
//! Regular HTTP OpenAI-compatible SSE carries the admission-time snapshot to
//! its first nonempty reasoning/content text callback. Metadata, cancellation,
//! completion, and nonstreaming responses never supply training observations.
//! No calibrated field feeds admission or promotes raw readiness. Worker
//! invalidation and detectable native-capacity profile resets clear learning;
//! a model/configuration change without either signal is not detectable here.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, OnceLock, Weak,
    },
    time::Instant,
};

use axum::response::Response;
use openai_protocol::worker::WorkerLoadResponse;
use parking_lot::{Mutex, MutexGuard};
use serde::{Deserialize, Serialize};

use super::{
    expected_wait::{
        ExpectedWait, DEFAULT_DISPATCH_BLOCKING_FACTOR, DEFAULT_MEAN_PREFILL_TOKENS,
        DEFAULT_THROUGHPUT,
    },
    Worker, WorkerRegistry,
};
use crate::{
    config::{ConfigError, ConfigResult},
    routers::common::overload,
};

/// All defaults are starting points for calibration, not capacity guarantees.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct EstimatedWaitConfig {
    pub max_estimated_wait_secs: Option<f64>,
    /// Deprecated compatibility field; the admission formula has no KV term.
    /// Only zero is accepted so old disabled-KV configurations still load.
    pub estimated_wait_kv_pressure_weight: f64,
    pub estimated_wait_base_overhead_secs: f64,
    pub estimated_wait_queue_work_correction: f64,
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
    /// Learn excess-TTFT correction for diagnostics only; never changes admission.
    pub estimated_wait_calibration_shadow: bool,
}

impl Default for EstimatedWaitConfig {
    fn default() -> Self {
        Self {
            max_estimated_wait_secs: None,
            estimated_wait_kv_pressure_weight: 0.0,
            estimated_wait_base_overhead_secs: 0.0,
            estimated_wait_queue_work_correction: 1.0,
            estimated_wait_dispatch_blocking_factor: DEFAULT_DISPATCH_BLOCKING_FACTOR,
            estimated_wait_mean_prefill_tokens: DEFAULT_MEAN_PREFILL_TOKENS,
            estimated_wait_fallback_prefill_throughput: DEFAULT_THROUGHPUT,
            estimated_wait_prompt_size_prior_samples: 32,
            estimated_wait_queue_tokens_per_request: 0,
            estimated_wait_max_snapshot_age_secs: 30.0,
            estimated_wait_shadow: false,
            estimated_wait_calibration_shadow: false,
        }
    }
}

impl EstimatedWaitConfig {
    pub fn validate(&self) -> ConfigResult<()> {
        if self.estimated_wait_kv_pressure_weight != 0.0 {
            return Err(ConfigError::InvalidValue {
                field: "estimated_wait_kv_pressure_weight".to_string(),
                value: self.estimated_wait_kv_pressure_weight.to_string(),
                reason: "KV pressure was removed from estimated-wait admission; use zero"
                    .to_string(),
            });
        }
        for (field, value, allow_zero) in [
            (
                "max_estimated_wait_secs",
                self.max_estimated_wait_secs.unwrap_or(1.0),
                false,
            ),
            (
                "estimated_wait_base_overhead_secs",
                self.estimated_wait_base_overhead_secs,
                true,
            ),
            (
                "estimated_wait_queue_work_correction",
                self.estimated_wait_queue_work_correction,
                true,
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
                    proxy = true;
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
            baseline.unwrap_or(0.0) + self.estimated_wait_base_overhead_secs,
            self.estimated_wait_queue_work_correction,
            self.estimated_wait_dispatch_blocking_factor,
        );
        estimate.seconds(0).is_finite().then_some(PreparedWait {
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
            pending_engine: 0.0,
            proxy,
            fallback,
            capacity_ready,
            blended_prompt_size,
            effective_prefill_capacity: throughput,
            native_prefill_capacity: throughput,
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
    estimate: ExpectedWait,
    native_estimate: ExpectedWait,
    native_waiting: f64,
    native_running: f64,
    pending_engine: f64,
    proxy: bool,
    fallback: bool,
    capacity_ready: bool,
    blended_prompt_size: Option<f64>,
    effective_prefill_capacity: f64,
    native_prefill_capacity: f64,
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
    metrics::gauge!("smg_estimated_wait_seconds", "worker" => worker.clone()).set(-1.0);
    metrics::gauge!("smg_estimated_wait_excess_seconds", "worker" => worker.clone()).set(-1.0);
    metrics::gauge!("smg_estimated_wait_native_queue_delay_seconds", "worker" => worker.clone())
        .set(-1.0);
    metrics::gauge!("smg_estimated_wait_pending_engine_requests", "worker" => worker.clone())
        .set(-1.0);
    metrics::gauge!("smg_estimated_wait_pending_first_text_requests", "worker" => worker.clone())
        .set(-1.0);
    metrics::gauge!("smg_estimated_wait_observed_delay_floor_seconds", "worker" => worker.clone())
        .set(-1.0);
    metrics::gauge!("smg_estimated_wait_observed_delay_sample_count", "worker" => worker.clone())
        .set(-1.0);
    metrics::gauge!("smg_estimated_wait_threshold_seconds", "worker" => worker.clone()).set(-1.0);
    metrics::gauge!("smg_estimated_wait_queue_proxy", "worker" => worker.clone()).set(-1.0);
    metrics::gauge!("smg_estimated_wait_throughput_fallback", "worker" => worker.clone()).set(-1.0);
    metrics::gauge!("smg_estimated_wait_snapshot_age_seconds", "worker" => worker.clone())
        .set(-1.0);
    metrics::gauge!("smg_estimated_wait_blended_prompt_tokens", "worker" => worker.clone())
        .set(-1.0);
    metrics::gauge!("smg_estimated_wait_effective_prefill_capacity", "worker" => worker.clone())
        .set(-1.0);
    metrics::gauge!("smg_estimated_wait_capacity_ready", "worker" => worker.clone()).set(-1.0);
    metrics::gauge!("smg_estimated_wait_baseline_seconds", "worker" => worker.clone()).set(-1.0);
    metrics::gauge!("smg_estimated_wait_visibility_delay_seconds", "worker" => worker.clone())
        .set(-1.0);
    metrics::gauge!("smg_estimated_wait_baseline_sample_count", "worker" => worker.clone())
        .set(-1.0);
    for name in [
        "smg_estimated_wait_calibrated_seconds",
        "smg_estimated_wait_calibrated_excess_seconds",
        "smg_estimated_wait_calibration_scale",
        "smg_estimated_wait_calibration_target_scale",
        "smg_estimated_wait_calibration_sample_count",
        "smg_estimated_wait_calibration_ready",
        "smg_estimated_wait_calibration_native_capacity_ready",
        "smg_estimated_wait_calibrated_would_reject",
    ] {
        metrics::gauge!(name, "worker" => worker.clone()).set(-1.0);
    }
    metrics::gauge!("smg_estimated_wait_data_usable", "worker" => worker).set(0.0);
}

const TTFT_WINDOW_SAMPLES: usize = 32;
const TTFT_MIN_SAMPLES: usize = 4;
const MAX_TTFT_EVENT_BYTES: usize = 64 * 1024;
const CALIBRATION_MAX_SCALE_CHANGE_PER_SECOND: f64 = 0.1;

/// Admission-time inputs travel with the request, not with the mutable load sample.
#[derive(Clone, Debug)]
pub(crate) struct CalibrationForecast {
    generation: Arc<()>,
    calibration_generation: Arc<()>,
    epoch: u64,
    baseline: Option<f64>,
    raw_extra: f64,
    native_capacity_ready: bool,
    raw_ready: bool,
    fresh: bool,
    scale: f64,
    target_scale: f64,
    time_seconds: f64,
    sample_count: usize,
    observation_sequence: u64,
    threshold: f64,
}

impl CalibrationForecast {
    fn state(&self) -> &'static str {
        if self.native_capacity_ready {
            "capacity_ready"
        } else {
            "capacity_unready"
        }
    }

    fn extra(&self) -> f64 {
        self.scale * self.raw_extra
    }

    fn seconds(&self) -> f64 {
        self.baseline.unwrap_or(0.0) + self.extra()
    }

    fn would_reject(&self) -> bool {
        self.raw_ready && self.extra().is_finite() && self.extra() >= self.threshold
    }
}

#[derive(Debug)]
struct ScaleTrajectory {
    anchor_scale: f64,
    target_scale: f64,
    anchor_time: Instant,
}

impl ScaleTrajectory {
    fn new(now: Instant) -> Self {
        Self {
            anchor_scale: 1.0,
            target_scale: 1.0,
            anchor_time: now,
        }
    }

    fn applied(&self, now: Instant) -> f64 {
        let limit = CALIBRATION_MAX_SCALE_CHANGE_PER_SECOND
            * now
                .saturating_duration_since(self.anchor_time)
                .as_secs_f64();
        if self.target_scale >= self.anchor_scale {
            self.anchor_scale + (self.target_scale - self.anchor_scale).min(limit)
        } else {
            self.anchor_scale - (self.anchor_scale - self.target_scale).min(limit)
        }
    }

    fn retarget(&mut self, target: f64, now: Instant) {
        self.anchor_scale = self.applied(now);
        self.anchor_time = now;
        self.target_scale = target;
    }
}

#[derive(Debug)]
struct ExcessCalibration {
    // Separate histories across native capacity warm-up, even when F promotes
    // the independent raw admission-readiness gate.
    samples: [VecDeque<(f64, f64)>; 2],
    accepted: [u64; 2],
    trajectories: [ScaleTrajectory; 2],
    started: Instant,
}

impl Default for ExcessCalibration {
    fn default() -> Self {
        Self::new(Instant::now())
    }
}

impl ExcessCalibration {
    fn new(now: Instant) -> Self {
        Self {
            samples: Default::default(),
            accepted: [0; 2],
            trajectories: [ScaleTrajectory::new(now), ScaleTrajectory::new(now)],
            started: now,
        }
    }

    fn time_seconds(&self, now: Instant) -> f64 {
        now.saturating_duration_since(self.started).as_secs_f64()
    }

    fn stats(&self, native_ready: bool, now: Instant) -> (f64, usize) {
        let index = usize::from(native_ready);
        (
            self.trajectories[index].applied(now),
            self.samples[index].len(),
        )
    }

    fn target(&self, native_ready: bool) -> f64 {
        self.trajectories[usize::from(native_ready)].target_scale
    }

    fn median_target(samples: &VecDeque<(f64, f64)>) -> f64 {
        if samples.len() < TTFT_MIN_SAMPLES {
            return 1.0;
        }
        let mut ratios: Vec<_> = samples.iter().copied().collect();
        ratios.sort_by(|left, right| left.0.total_cmp(&right.0));
        // Normalize weights before summing so finite extreme inputs cannot
        // overflow the weighted median's total.
        let largest = ratios.iter().map(|(_, weight)| *weight).fold(0.0, f64::max);
        let half = ratios
            .iter()
            .map(|(_, weight)| weight / largest)
            .sum::<f64>()
            / 2.0;
        let mut cumulative = 0.0;
        for (ratio, weight) in ratios {
            cumulative += weight / largest;
            if cumulative >= half {
                return ratio;
            }
        }
        1.0
    }

    fn observe(&mut self, forecast: &CalibrationForecast, ttft: f64, now: Instant) -> &'static str {
        if !forecast.fresh {
            return "stale_at_admission";
        }
        let Some(baseline) = forecast.baseline else {
            return "baseline_unavailable";
        };
        if !baseline.is_finite() || baseline < 0.0 || !ttft.is_finite() || ttft <= 0.0 {
            return "invalid_observation";
        }
        if !forecast.raw_extra.is_finite() || forecast.raw_extra <= 0.0 {
            return "nonpositive_extra";
        }
        let ratio = (ttft - baseline).max(0.0) / forecast.raw_extra;
        if !ratio.is_finite() {
            return "invalid_ratio";
        }
        let samples = &mut self.samples[usize::from(forecast.native_capacity_ready)];
        samples.push_back((ratio, forecast.raw_extra));
        self.accepted[usize::from(forecast.native_capacity_ready)] += 1;
        while samples.len() > TTFT_WINDOW_SAMPLES {
            samples.pop_front();
        }
        let target = Self::median_target(samples);
        self.trajectories[usize::from(forecast.native_capacity_ready)].retarget(target, now);
        "accepted"
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
    request_id: Option<String>,
}

impl StreamTtftObserver {
    pub(crate) fn cancel(&mut self) {
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
            request_id: None,
        }
    }

    pub(crate) fn with_request_id(mut self, request_id: Option<&str>) -> Self {
        self.request_id = request_id.map(|id| id.chars().take(256).collect());
        if let Some(probe) = &mut self.probe {
            probe.request_id.clone_from(&self.request_id);
        }
        self
    }

    pub(crate) fn with_calibration(mut self, forecast: Option<CalibrationForecast>) -> Self {
        if let Some(probe) = &mut self.probe {
            probe.calibration = forecast;
        }
        self
    }

    pub(crate) fn push(&mut self, bytes: &[u8]) {
        if self.probe.is_none() {
            return;
        }
        if let Some((ttft, visibility)) = self.scan(bytes, Instant::now()) {
            tracing::debug!(
                target: "smg::estimated_wait::observation",
                event = "first_text",
                request_id = self.request_id.as_deref().unwrap_or(""),
                ttft_seconds = ttft,
                visibility_seconds = visibility,
            );
            if let (Some(registry), Some(probe)) = (self.registry.upgrade(), self.probe.take()) {
                registry
                    .estimated_wait()
                    .observe_ttft(probe, ttft, visibility);
            }
        }
    }

    fn scan(&mut self, bytes: &[u8], now: Instant) -> Option<(f64, f64)> {
        if self.done {
            return None;
        }
        if self.pending.len().saturating_add(bytes.len()) > MAX_TTFT_EVENT_BYTES {
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
        }
        None
    }
}

impl Drop for StreamTtftObserver {
    fn drop(&mut self) {
        self.finish_probe();
    }
}

#[derive(Debug)]
struct Sample {
    prepared: PreparedWait,
    started: Instant,
    watermark: u64,
    threshold: f64,
    // Latched on publication and dispatch, just like the static overload bit.
    // The request path only reads it and checks freshness.
    overloaded: Option<bool>,
}

#[derive(Debug)]
struct Entry {
    source: Weak<dyn Worker>,
    generation: Arc<()>,
    total_dispatched: u64,
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
    calibration: ExcessCalibration,
    calibration_generation: Arc<()>,
    calibration_epoch: u64,
    calibration_profile: Option<(bool, Option<u64>)>,
}

impl Entry {
    fn new(worker: &Arc<dyn Worker>) -> Self {
        Self {
            source: Arc::downgrade(worker),
            generation: Arc::new(()),
            total_dispatched: 0,
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
            calibration: ExcessCalibration::default(),
            calibration_generation: Arc::new(()),
            calibration_epoch: 0,
            calibration_profile: None,
        }
    }

    fn invalidate(&mut self) {
        self.generation = Arc::new(());
        self.total_dispatched = 0;
        self.last_published = None;
        self.sample = None;
        self.low_load_ttfts.clear();
        self.low_load_visibility.clear();
        self.busy_visibility.clear();
        self.observed_delays.clear();
        self.pending_first_text.clear();
        self.backlog_peak_size = 0.0;
        self.pipeline_idle = true;
        self.calibration = ExcessCalibration::default();
        self.calibration_generation = Arc::new(());
        self.calibration_epoch = self.calibration_epoch.wrapping_add(1);
        self.calibration_profile = None;
        self.backlog_generation = self.backlog_generation.wrapping_add(1);
    }

    fn calibration_forecast(
        &self,
        config: &EstimatedWaitConfig,
        now: Instant,
    ) -> Option<CalibrationForecast> {
        if !config.estimated_wait_calibration_shadow {
            return None;
        }
        let sample = self.sample.as_ref()?;
        let fresh = now.saturating_duration_since(sample.started).as_secs_f64()
            <= config.estimated_wait_max_snapshot_age_secs;
        let (scale, sample_count) = self
            .calibration
            .stats(sample.prepared.capacity_learned_ready, now);
        Some(CalibrationForecast {
            generation: Arc::clone(&self.generation),
            calibration_generation: Arc::clone(&self.calibration_generation),
            epoch: self.calibration_epoch,
            baseline: sample.prepared.baseline,
            raw_extra: sample
                .prepared
                .excess_seconds(self.total_dispatched.saturating_sub(sample.watermark)),
            native_capacity_ready: sample.prepared.capacity_learned_ready,
            raw_ready: fresh && sample.prepared.capacity_ready,
            fresh,
            scale,
            target_scale: self
                .calibration
                .target(sample.prepared.capacity_learned_ready),
            time_seconds: self.calibration.time_seconds(now),
            sample_count,
            observation_sequence: self.calibration.accepted
                [usize::from(sample.prepared.capacity_learned_ready)],
            threshold: sample.threshold,
        })
    }

    fn refresh(&mut self, worker: &Arc<dyn Worker>, config: &EstimatedWaitConfig) {
        let now = Instant::now();
        if let Some(sample) = &mut self.sample {
            let seconds = sample
                .prepared
                .seconds(self.total_dispatched.saturating_sub(sample.watermark));
            let excess = sample
                .prepared
                .excess_seconds(self.total_dispatched.saturating_sub(sample.watermark));
            sample.overloaded = excess.is_finite().then_some(excess >= sample.threshold);
            metrics::gauge!("smg_estimated_wait_pending_engine_requests", "worker" => worker.url().to_owned()).set(sample.prepared.pending_engine);
            metrics::gauge!("smg_estimated_wait_pending_first_text_requests", "worker" => worker.url().to_owned()).set(self.pending_first_text.len() as f64);
            metrics::gauge!("smg_estimated_wait_native_queue_delay_seconds", "worker" => worker.url().to_owned()).set(sample.prepared.native_queue_delay.unwrap_or(-1.0));
            metrics::gauge!("smg_estimated_wait_seconds", "worker" => worker.url().to_owned())
                .set(seconds);
            metrics::gauge!("smg_estimated_wait_excess_seconds", "worker" => worker.url().to_owned())
                .set(excess);
            metrics::gauge!("smg_estimated_wait_snapshot_age_seconds", "worker" => worker.url().to_owned()).set(now.saturating_duration_since(sample.started).as_secs_f64());
            metrics::gauge!("smg_estimated_wait_capacity_ready", "worker" => worker.url().to_owned())
                .set(u8::from(sample.prepared.capacity_ready));
            metrics::gauge!("smg_estimated_wait_effective_prefill_capacity", "worker" => worker.url().to_owned())
                .set(sample.prepared.effective_prefill_capacity);
            metrics::gauge!("smg_estimated_wait_baseline_seconds", "worker" => worker.url().to_owned())
                .set(sample.prepared.baseline.unwrap_or(-1.0));
            metrics::gauge!("smg_estimated_wait_visibility_delay_seconds", "worker" => worker.url().to_owned())
                .set(sample.prepared.visibility_delay);
            metrics::gauge!("smg_estimated_wait_baseline_sample_count", "worker" => worker.url().to_owned())
                .set(self.low_load_ttfts.len() as f64);
            metrics::gauge!("smg_estimated_wait_observed_delay_floor_seconds", "worker" => worker.url().to_owned())
                .set(sample.prepared.observed_delay_floor);
            metrics::gauge!("smg_estimated_wait_observed_delay_sample_count", "worker" => worker.url().to_owned())
                .set(self.observed_delays.len() as f64);
            if config.estimated_wait_calibration_shadow {
                let (scale, count) = self
                    .calibration
                    .stats(sample.prepared.capacity_learned_ready, now);
                let calibrated_extra = scale * excess;
                let fresh = now.saturating_duration_since(sample.started).as_secs_f64()
                    <= config.estimated_wait_max_snapshot_age_secs;
                metrics::gauge!("smg_estimated_wait_calibrated_seconds", "worker" => worker.url().to_owned()).set(sample.prepared.baseline.unwrap_or(0.0) + calibrated_extra);
                metrics::gauge!("smg_estimated_wait_calibrated_excess_seconds", "worker" => worker.url().to_owned()).set(calibrated_extra);
                metrics::gauge!("smg_estimated_wait_calibration_scale", "worker" => worker.url().to_owned()).set(scale);
                metrics::gauge!("smg_estimated_wait_calibration_target_scale", "worker" => worker.url().to_owned()).set(self.calibration.target(sample.prepared.capacity_learned_ready));
                metrics::gauge!("smg_estimated_wait_calibration_sample_count", "worker" => worker.url().to_owned()).set(count as f64);
                metrics::gauge!("smg_estimated_wait_calibration_ready", "worker" => worker.url().to_owned()).set(u8::from(count >= TTFT_MIN_SAMPLES));
                metrics::gauge!("smg_estimated_wait_calibration_native_capacity_ready", "worker" => worker.url().to_owned()).set(u8::from(sample.prepared.capacity_learned_ready));
                metrics::gauge!("smg_estimated_wait_calibrated_would_reject", "worker" => worker.url().to_owned()).set(u8::from(fresh && sample.prepared.capacity_ready && calibrated_extra.is_finite() && calibrated_extra >= sample.threshold));
            }
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
            backlog_size: sample.prepared.backlog_size,
            calibration: None,
            request_id: None,
        };
        entry.pending_first_text.insert(probe.observation_id);
        Self::apply_observations(entry, self.config());
        entry.refresh(&worker, self.config());
        Some(probe)
    }

    fn finish_ttft_probe(&self, probe: &TtftProbe) {
        let mut entries = self.entries.lock();
        let Some(entry) = entries.get_mut(&probe.worker_url) else {
            return;
        };
        if !Arc::ptr_eq(&entry.generation, &probe.generation) {
            return;
        }
        entry.pending_first_text.remove(&probe.observation_id);
        Self::apply_observations(entry, self.config());
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
        if let Some(forecast) = probe
            .calibration
            .as_ref()
            .filter(|_| self.config().estimated_wait_calibration_shadow)
        {
            let now = Instant::now();
            let before = entry.calibration.stats(forecast.native_capacity_ready, now);
            let target_before = entry.calibration.target(forecast.native_capacity_ready);
            let sequence_before =
                entry.calibration.accepted[usize::from(forecast.native_capacity_ready)];
            let reason = if Arc::ptr_eq(&entry.generation, &forecast.generation)
                && Arc::ptr_eq(
                    &entry.calibration_generation,
                    &forecast.calibration_generation,
                ) {
                entry.calibration.observe(forecast, ttft, now)
            } else {
                "calibration_generation_changed"
            };
            let after = entry.calibration.stats(forecast.native_capacity_ready, now);
            let sequence_after =
                entry.calibration.accepted[usize::from(forecast.native_capacity_ready)];
            tracing::debug!(
                target: "smg::estimated_wait::observation", event = "calibration_feedback",
                request_id = probe.request_id.as_deref().unwrap_or(""), worker = probe.worker_url,
                baseline_seconds = forecast.baseline.unwrap_or(0.0), baseline_available = forecast.baseline.is_some(),
                raw_excess_seconds = forecast.raw_extra, calibration_state = forecast.state(),
                ttft_seconds = ttft, reason, admission_scale = forecast.scale,
                admission_sample_count = forecast.sample_count,
                admission_observation_sequence = forecast.observation_sequence, sequence_before, sequence_after,
                calibration_epoch = forecast.epoch, current_calibration_epoch = entry.calibration_epoch,
                scale_before = before.0, sample_count_before = before.1,
                scale_after = after.0, sample_count_after = after.1,
                target_scale_before = target_before, target_scale_after = entry.calibration.target(forecast.native_capacity_ready),
                calibration_time_seconds = entry.calibration.time_seconds(now), admission_calibration_time_seconds = forecast.time_seconds,
            );
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
        Self::apply_observations(entry, self.config());
        entry.refresh(&worker, self.config());
    }

    fn apply_observations(entry: &mut Entry, config: &EstimatedWaitConfig) {
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
            prepared.estimate = prepared
                .estimate
                .with_base_overhead(baseline + config.estimated_wait_base_overhead_secs);
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

    pub(crate) fn enabled(&self) -> bool {
        self.config
            .get()
            .is_some_and(|c| c.max_estimated_wait_secs.is_some())
            || self.worker_overrides.load(Ordering::Acquire) > 0
    }

    pub(crate) fn needs_load(&self, worker: &Arc<dyn Worker>) -> bool {
        self.config().threshold(worker).is_some()
    }

    /// Called under the registry's per-worker mutation lock, before publication.
    pub(crate) fn worker_added(&self, worker: &Arc<dyn Worker>) {
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
        let mut entries = self.entries.lock();
        let entry = Self::entry(&mut entries, worker)?;
        Some(PollWatermark {
            generation: Arc::clone(&entry.generation),
            dispatched: entry.total_dispatched,
        })
    }

    pub(crate) fn publish(
        &self,
        worker: &Arc<dyn Worker>,
        load: Option<&WorkerLoadResponse>,
        watermark: Option<PollWatermark>,
        started: Instant,
    ) {
        let Some(watermark) = watermark else {
            return;
        };
        // Rank aggregation, validation and the queue/KV formula are report work,
        // outside the admission transaction lock.
        let config = self.config();
        let prepared = load.and_then(|load| config.prepare(load));
        let threshold = config.threshold(worker);
        let mut entries = self.entries.lock();
        let Some(entry) = entries.get_mut(worker.url()) else {
            return;
        };
        if !Arc::ptr_eq(&entry.generation, &watermark.generation)
            || entry.last_published.is_some_and(|last| last > started)
        {
            return;
        }
        entry.last_published = Some(started);
        // Native capacity samples only shrink when the monitor resets its
        // counter/profile history. Do not let pre-reset responses train the
        // new profile, and leave all raw estimator state untouched.
        if let Some(new) = prepared.filter(|_| config.estimated_wait_calibration_shadow) {
            if entry
                .calibration_profile
                .is_some_and(|(old_ready, old_count)| {
                    (old_ready && !new.capacity_learned_ready)
                        || old_count
                            .zip(new.capacity_profile_sample_count)
                            .is_some_and(|(old, new)| new < old)
                })
            {
                entry.calibration = ExcessCalibration::default();
                entry.calibration_generation = Arc::new(());
                entry.calibration_epoch = entry.calibration_epoch.wrapping_add(1);
            }
            // A failed scrape invalidates the raw report, not this reset
            // watermark. Otherwise a counter reset across one failed poll
            // could silently retain the preceding calibration history.
            entry.calibration_profile = Some((
                new.capacity_learned_ready,
                new.capacity_profile_sample_count,
            ));
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
        entry.sample = prepared.zip(threshold).map(|(prepared, threshold)| Sample {
            prepared,
            started,
            watermark: watermark.dispatched,
            threshold,
            overloaded: None,
        });
        Self::apply_observations(entry, config);
        if let Some(sample) = &entry.sample {
            metrics::gauge!("smg_estimated_wait_threshold_seconds", "worker" => worker.url().to_owned()).set(sample.threshold);
            metrics::gauge!("smg_estimated_wait_queue_proxy", "worker" => worker.url().to_owned())
                .set(u8::from(sample.prepared.proxy));
            metrics::gauge!("smg_estimated_wait_throughput_fallback", "worker" => worker.url().to_owned()).set(u8::from(sample.prepared.fallback));
            metrics::gauge!("smg_estimated_wait_blended_prompt_tokens", "worker" => worker.url().to_owned())
                .set(sample.prepared.blended_prompt_size.unwrap_or(-1.0));
            metrics::gauge!("smg_estimated_wait_effective_prefill_capacity", "worker" => worker.url().to_owned())
                .set(sample.prepared.effective_prefill_capacity);
            metrics::gauge!("smg_estimated_wait_baseline_seconds", "worker" => worker.url().to_owned())
                .set(sample.prepared.baseline.unwrap_or(-1.0));
            metrics::gauge!("smg_estimated_wait_visibility_delay_seconds", "worker" => worker.url().to_owned())
                .set(sample.prepared.visibility_delay);
            metrics::gauge!("smg_estimated_wait_baseline_sample_count", "worker" => worker.url().to_owned())
                .set(entry.low_load_ttfts.len() as f64);
            metrics::gauge!("smg_estimated_wait_observed_delay_floor_seconds", "worker" => worker.url().to_owned())
                .set(sample.prepared.observed_delay_floor);
            metrics::gauge!("smg_estimated_wait_observed_delay_sample_count", "worker" => worker.url().to_owned())
                .set(entry.observed_delays.len() as f64);
        }
        entry.refresh(worker, config);
    }

    pub(crate) fn evict(&self, worker: &Arc<dyn Worker>) {
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

pub(crate) struct PollWatermark {
    generation: Arc<()>,
    dispatched: u64,
}

pub(crate) struct AdmissionGuard<'a> {
    config: &'a EstimatedWaitConfig,
    entries: MutexGuard<'a, HashMap<String, Entry>>,
}

impl AdmissionGuard<'_> {
    pub(crate) fn capture_calibration(
        &self,
        worker: &Arc<dyn Worker>,
        request_id: Option<&str>,
    ) -> Option<CalibrationForecast> {
        let forecast = self
            .entries
            .get(worker.url())?
            .calibration_forecast(self.config, Instant::now())?;
        tracing::debug!(
            target: "smg::estimated_wait::observation", event = "calibration_forecast",
            request_id = request_id.unwrap_or(""), worker = worker.url(),
            calibration_enabled = true, calibration_state = forecast.state(),
            calibration_epoch = forecast.epoch,
            calibration_observation_sequence = forecast.observation_sequence,
            calibration_scale = forecast.scale, calibration_sample_count = forecast.sample_count,
            calibration_target_scale = forecast.target_scale, calibration_time_seconds = forecast.time_seconds,
            calibration_ready = forecast.sample_count >= TTFT_MIN_SAMPLES,
            baseline_seconds = forecast.baseline.unwrap_or(0.0), baseline_available = forecast.baseline.is_some(),
            predicted_ttft_seconds = forecast.baseline.unwrap_or(0.0) + forecast.raw_extra,
            predicted_excess_seconds = forecast.raw_extra,
            calibrated_ttft_seconds = forecast.seconds(), calibrated_excess_seconds = forecast.extra(),
            ready = forecast.raw_ready, native_ready = forecast.native_capacity_ready,
            would_reject = forecast.raw_ready && forecast.raw_extra >= forecast.threshold,
            threshold_seconds = forecast.threshold, calibrated_would_reject = forecast.would_reject(),
        );
        Some(forecast)
    }

    pub(crate) fn record_forecast(&self, candidates: &[Arc<dyn Worker>], request_id: Option<&str>) {
        let Some(request_id) = request_id else {
            return;
        };
        if !tracing::enabled!(target: "smg::estimated_wait::observation", tracing::Level::DEBUG) {
            return;
        }
        for worker in candidates.iter().filter(|worker| worker.is_available()) {
            let now = Instant::now();
            let Some(entry) = self.entries.get(worker.url()) else {
                continue;
            };
            let Some(sample) = &entry.sample else {
                continue;
            };
            let dispatched = entry.total_dispatched.saturating_sub(sample.watermark);
            let fresh = now.saturating_duration_since(sample.started).as_secs_f64()
                <= self.config.estimated_wait_max_snapshot_age_secs;
            let ready = fresh && sample.prepared.capacity_ready;
            let legacy_visibility = if sample.prepared.low_load {
                0.0
            } else {
                ttft_median(&entry.busy_visibility)
                    .zip(ttft_median(&entry.low_load_visibility))
                    .map(|(busy, low)| (busy - low).max(0.0))
                    .unwrap_or(0.0)
            };
            // Counterfactual arithmetic on the same shadow trace. It does not
            // simulate a separately enforced policy or its resulting queue.
            let native_excess = self.config.estimated_wait_base_overhead_secs
                + legacy_visibility
                + sample.prepared.native_estimate.token_work(dispatched)
                    / sample.prepared.native_prefill_capacity;
            let calibration = entry.calibration_forecast(self.config, now);
            tracing::debug!(
                target: "smg::estimated_wait::observation",
                event = "forecast",
                request_id,
                worker = worker.url(),
                predicted_ttft_seconds = sample.prepared.seconds(dispatched),
                predicted_excess_seconds = sample.prepared.excess_seconds(dispatched),
                baseline_seconds = sample.prepared.baseline.unwrap_or(0.0),
                threshold_seconds = sample.threshold,
                ready,
                would_reject = ready && sample.overloaded == Some(true),
                native_excess_seconds = native_excess,
                native_ready = fresh && sample.prepared.capacity_learned_ready,
                legacy_baseline_gate_ready = fresh && sample.prepared.capacity_learned_ready && sample.prepared.baseline.is_some(),
                token_work = sample.prepared.estimate.token_work(dispatched),
                queued_tokens = sample.prepared.estimate.queued_work(),
                dispatched_tokens_since_poll = dispatched,
                observed_backlog = sample.prepared.backlog_size,
                pending_first_text_requests = entry.pending_first_text.len(),
                observed_delay_available = sample.prepared.observed_delay_available,
                measured_delay_only = sample.prepared.measured_delay_only(),
                capacity_learned_ready = sample.prepared.capacity_learned_ready,
                snapshot_age_seconds = now.saturating_duration_since(sample.started).as_secs_f64(),
                native_capacity = sample.prepared.native_prefill_capacity,
                effective_capacity = sample.prepared.effective_prefill_capacity,
                visibility_seconds = sample.prepared.visibility_delay,
                observed_samples = entry.observed_delays.len(),
                observed_delay_floor = sample.prepared.observed_delay_floor,
                pending_engine = sample.prepared.pending_engine,
                native_waiting = sample.prepared.native_waiting,
                native_queue_delay = sample.prepared.native_queue_delay.unwrap_or(-1.0),
                baseline_available = sample.prepared.baseline.is_some(),
                calibration_enabled = calibration.is_some(),
                calibrated_ttft_seconds = calibration.as_ref().map_or(-1.0, CalibrationForecast::seconds),
                calibrated_excess_seconds = calibration.as_ref().map_or(-1.0, CalibrationForecast::extra),
                calibration_scale = calibration.as_ref().map_or(-1.0, |forecast| forecast.scale),
                calibration_target_scale = calibration.as_ref().map_or(-1.0, |forecast| forecast.target_scale),
                calibration_time_seconds = calibration.as_ref().map_or(-1.0, |forecast| forecast.time_seconds),
                calibration_sample_count = calibration.as_ref().map_or(0, |forecast| forecast.sample_count),
                calibration_state = calibration.as_ref().map_or("disabled", CalibrationForecast::state),
                calibration_epoch = calibration.as_ref().map_or(0, |forecast| forecast.epoch),
                calibration_observation_sequence = calibration.as_ref().map_or(0, |forecast| forecast.observation_sequence),
                calibration_ready = calibration.as_ref().is_some_and(|forecast| forecast.sample_count >= TTFT_MIN_SAMPLES),
                calibrated_would_reject = calibration.as_ref().is_some_and(CalibrationForecast::would_reject),
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
        for worker in candidates.iter().filter(|w| w.is_available()) {
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
            let verdict = match sample {
                Some(sample)
                    if sample.started.elapsed().as_secs_f64()
                        > self.config.estimated_wait_max_snapshot_age_secs =>
                {
                    Err("stale")
                }
                Some(sample) if !sample.prepared.capacity_ready => Err("capacity_unready"),
                Some(sample) => sample.overloaded.ok_or("unusable"),
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

    fn config() -> EstimatedWaitConfig {
        EstimatedWaitConfig {
            max_estimated_wait_secs: Some(2.0),
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
            .capture_calibration(worker, Some("test"))
            .unwrap();
        let mut probe = admission.ttft_probe(worker.url()).unwrap();
        probe.calibration = Some(forecast);
        probe
    }

    #[test]
    fn calibration_slew_rises_falls_clamps_and_recovers_from_zero() {
        let start = Instant::now();
        let mut trajectory = ScaleTrajectory::new(start);
        trajectory.retarget(3.0, start);
        assert_eq!(trajectory.applied(start), 1.0);
        assert_eq!(trajectory.applied(start + Duration::from_secs(5)), 1.5);
        assert_eq!(trajectory.applied(start + Duration::from_secs(30)), 3.0);
        trajectory.retarget(0.0, start + Duration::from_secs(30));
        assert_eq!(trajectory.applied(start + Duration::from_secs(30)), 3.0);
        assert_eq!(trajectory.applied(start + Duration::from_secs(40)), 2.0);
        assert_eq!(trajectory.applied(start + Duration::from_secs(60)), 0.0);
        // Seventy idle seconds at zero cannot be spent on a newly seen target.
        trajectory.retarget(2.0, start + Duration::from_secs(100));
        assert_eq!(trajectory.applied(start + Duration::from_secs(100)), 0.0);
        assert_eq!(trajectory.applied(start + Duration::from_secs(105)), 0.5);
        assert_eq!(trajectory.applied(start + Duration::from_secs(130)), 2.0);
    }

    #[test]
    fn calibration_fourth_sample_has_no_jump_or_prior_idle_credit() {
        let (registry, worker) = calibration_registry(true);
        let forecast = registry
            .estimated_wait()
            .begin()
            .unwrap()
            .capture_calibration(&worker, None)
            .unwrap();
        let start = Instant::now();
        let update = start + Duration::from_secs(3_600);
        let mut calibration = ExcessCalibration::new(start);
        for _ in 0..3 {
            calibration.observe(&forecast, 20.2, update);
        }
        assert_eq!(calibration.target(true), 1.0);
        assert_eq!(calibration.stats(true, update), (1.0, 3));
        calibration.observe(&forecast, 20.2, update);
        assert_eq!(calibration.target(true), 2.0);
        assert_eq!(calibration.stats(true, update), (1.0, 4));
        for _ in 0..64 {
            calibration.observe(&forecast, 20.2, update);
        }
        assert_eq!(calibration.stats(true, update), (1.0, 32));
        assert_eq!(
            calibration.stats(true, update + Duration::from_secs(1)).0,
            1.1
        );
        assert_eq!(
            calibration.stats(false, update + Duration::from_secs(100)),
            (1.0, 0)
        );
    }

    #[test]
    fn calibration_slew_is_independent_of_callback_frequency() {
        let (registry, worker) = calibration_registry(true);
        let forecast = registry
            .estimated_wait()
            .begin()
            .unwrap()
            .capture_calibration(&worker, None)
            .unwrap();
        let start = Instant::now();
        let mut sparse = ExcessCalibration::new(start);
        let mut dense = ExcessCalibration::new(start);
        for _ in 0..4 {
            sparse.observe(&forecast, 50.2, start);
            dense.observe(&forecast, 50.2, start);
        }
        for step in 1..=1_000 {
            dense.observe(&forecast, 50.2, start + Duration::from_millis(step * 10));
        }
        let later = start + Duration::from_secs(15);
        assert_eq!(sparse.stats(true, later).0, 2.5);
        assert!((sparse.stats(true, later).0 - dense.stats(true, later).0).abs() < 1e-12);
        assert_eq!(dense.target(true), 5.0);
        assert_eq!(sparse.target(true), 5.0);
    }

    #[test]
    fn calibration_new_median_reverses_old_trajectory_without_a_jump() {
        let (registry, worker) = calibration_registry(true);
        let forecast = registry
            .estimated_wait()
            .begin()
            .unwrap()
            .capture_calibration(&worker, None)
            .unwrap();
        let start = Instant::now();
        let mut calibration = ExcessCalibration::new(start);
        for _ in 0..4 {
            calibration.observe(&forecast, 50.2, start);
        }
        let reversal = start + Duration::from_secs(5);
        assert_eq!(calibration.stats(true, reversal).0, 1.5);
        for _ in 0..32 {
            calibration.observe(&forecast, 0.1, reversal);
        }
        assert_eq!(calibration.target(true), 0.0);
        assert_eq!(calibration.stats(true, reversal).0, 1.5);
        assert_eq!(
            calibration.stats(true, reversal + Duration::from_secs(5)).0,
            1.0
        );
        assert_eq!(
            calibration
                .stats(true, reversal + Duration::from_secs(20))
                .0,
            0.0
        );
    }

    #[test]
    fn calibration_snapshot_advances_purely_with_time_and_records_one_clock() {
        let (registry, worker) = calibration_registry(true);
        let admission = registry.estimated_wait();
        let mut entries = admission.entries.lock();
        let entry = entries.get_mut(worker.url()).unwrap();
        let start = Instant::now();
        entry.calibration = ExcessCalibration::new(start);
        let original = entry
            .calibration_forecast(admission.config(), start)
            .unwrap();
        for _ in 0..4 {
            entry.calibration.observe(&original, 20.2, start);
        }
        let at_start = entry
            .calibration_forecast(admission.config(), start)
            .unwrap();
        assert_eq!(
            (at_start.scale, at_start.target_scale, at_start.time_seconds),
            (1.0, 2.0, 0.0)
        );
        let later = start + Duration::from_secs(5);
        let mut unready = original.clone();
        unready.native_capacity_ready = false;
        for _ in 0..4 {
            entry.calibration.observe(&unready, 0.1, start);
        }
        assert_eq!(entry.calibration.stats(false, later), (0.5, 4));
        let first = entry
            .calibration_forecast(admission.config(), later)
            .unwrap();
        let repeated = entry
            .calibration_forecast(admission.config(), later)
            .unwrap();
        assert_eq!(
            (first.scale, first.target_scale, first.time_seconds),
            (1.5, 2.0, 5.0)
        );
        assert_eq!(first.scale, repeated.scale);
        assert_eq!((first.observation_sequence, first.sample_count), (4, 4));
        assert_eq!(original.scale, 1.0);
        assert_eq!(entry.calibration.trajectories[1].anchor_time, start);
        assert_eq!(entry.calibration.trajectories[1].anchor_scale, 1.0);
        // Resetting a history also discards its elapsed movement and old target.
        entry.calibration = ExcessCalibration::new(later);
        let reset = entry
            .calibration_forecast(admission.config(), later)
            .unwrap();
        assert_eq!(
            (reset.scale, reset.target_scale, reset.time_seconds),
            (1.0, 1.0, 0.0)
        );
        assert_eq!((reset.observation_sequence, reset.sample_count), (0, 0));
    }

    #[test]
    fn excess_calibration_weighted_median_window_and_invalid_inputs() {
        let (registry, worker) = calibration_registry(true);
        let admission = registry.estimated_wait();
        let mut forecast = admission
            .begin()
            .unwrap()
            .capture_calibration(&worker, None)
            .unwrap();
        let now = Instant::now();
        let mut learned = ExcessCalibration::new(now);
        assert_eq!(learned.stats(true, now), (1.0, 0));
        for _ in 0..3 {
            assert_eq!(learned.observe(&forecast, 20.2, now), "accepted");
        }
        assert_eq!(learned.stats(true, now), (1.0, 3));
        assert_eq!(learned.observe(&forecast, 20.2, now), "accepted");
        assert_eq!(learned.target(true), 2.0);
        assert_eq!(learned.stats(true, now), (1.0, 4));
        // A huge ratio from tiny extra delay gets tiny weight, not a huge influence.
        forecast.raw_extra = 0.000_001;
        learned.observe(&forecast, 10.2, now);
        assert_eq!(learned.target(true), 2.0);
        forecast.raw_extra = 10.0;
        for _ in 0..TTFT_WINDOW_SAMPLES {
            learned.observe(&forecast, 5.2, now);
        }
        assert_eq!(learned.target(true), 0.5);
        assert_eq!(learned.stats(true, now), (1.0, 32));
        assert_eq!(learned.stats(false, now), (1.0, 0));
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            forecast.raw_extra = bad;
            assert_eq!(learned.observe(&forecast, 5.0, now), "nonpositive_extra");
        }
        forecast.raw_extra = 10.0;
        forecast.baseline = None;
        assert_eq!(learned.observe(&forecast, 5.0, now), "baseline_unavailable");
        forecast.baseline = Some(0.2);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(learned.observe(&forecast, bad, now), "invalid_observation");
        }
        forecast.fresh = false;
        assert_eq!(learned.observe(&forecast, 5.0, now), "stale_at_admission");
        assert_eq!(learned.target(true), 0.5);
        assert_eq!(learned.stats(true, now), (1.0, 32));
    }

    #[test]
    fn excess_calibration_preserves_baseline_and_scales_observed_floor() {
        let (registry, worker) = calibration_registry(true);
        let admission = registry.estimated_wait();
        let mut entries = admission.entries.lock();
        let entry = entries.get_mut(worker.url()).unwrap();
        let sample = entry.sample.as_mut().unwrap();
        sample.prepared.observed_delay_floor = 30.0;
        sample.prepared.observed_delay_available = true;
        let mut forecast = entry
            .calibration_forecast(admission.config(), Instant::now())
            .unwrap();
        assert_eq!(forecast.raw_extra, 30.0);
        forecast.scale = 0.5;
        assert_eq!(forecast.seconds(), 15.2);
        forecast.raw_extra = 0.0;
        forecast.scale = 50.0;
        assert_eq!(forecast.seconds(), 0.2);
        drop(entries);
    }

    #[test]
    fn excess_calibration_first_text_callback_is_causal_and_exactly_once() {
        let (registry, worker) = calibration_registry(true);
        let admission = registry.estimated_wait();
        let forecast = admission
            .begin()
            .unwrap()
            .capture_calibration(&worker, Some("causal"))
            .unwrap();
        // Later ledger changes must not replace the raw excess saved at admission.
        admission.begin().unwrap().credit(&worker, Some(&[0; 100]));
        let mut observer = StreamTtftObserver::new(
            &registry,
            worker.url(),
            Instant::now() - Duration::from_secs(20),
        )
        .with_calibration(Some(forecast))
        .with_request_id(Some("causal"));
        observer.push(
            b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
        );
        assert_eq!(
            admission.entries.lock()[worker.url()]
                .calibration
                .stats(true, Instant::now()),
            (1.0, 0)
        );
        observer.push(b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thi");
        assert_eq!(
            admission.entries.lock()[worker.url()]
                .calibration
                .stats(true, Instant::now())
                .1,
            0
        );
        observer.push(b"nking\"}}]}\n\n");
        {
            let entries = admission.entries.lock();
            let samples = &entries[worker.url()].calibration.samples[1];
            assert_eq!(samples.len(), 1);
            assert_eq!(samples[0].1, 10.0);
            assert!((samples[0].0 - 1.98).abs() < 0.01);
        }
        observer.push(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\ndata: [DONE]\n\n",
        );
        drop(observer);
        assert_eq!(
            admission.entries.lock()[worker.url()]
                .calibration
                .stats(true, Instant::now())
                .1,
            1
        );
        let forecast = admission
            .begin()
            .unwrap()
            .capture_calibration(&worker, None);
        let mut cancelled = StreamTtftObserver::new(&registry, worker.url(), Instant::now())
            .with_calibration(forecast);
        cancelled.cancel();
        cancelled.push(b"data: {\"choices\":[{\"text\":\"ignored\"}]}\n\n");
        assert_eq!(
            admission.entries.lock()[worker.url()]
                .calibration
                .stats(true, Instant::now())
                .1,
            1
        );
    }

    #[test]
    fn excess_calibration_placement_captures_before_dispatch_credit() {
        let (registry, worker) = calibration_registry(false);
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
        assert_eq!(saved.raw_extra, 10.0);
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
            11.0
        );
        let mut observer = StreamTtftObserver::new(
            &registry,
            worker.url(),
            Instant::now() - Duration::from_secs(20),
        )
        .with_calibration(Some(saved));
        observer.push(b"data: {\"choices\":[{\"text\":\"first text\"}]}\n\n");
        assert_eq!(
            registry.estimated_wait().entries.lock()[worker.url()]
                .calibration
                .samples[0][0]
                .1,
            10.0
        );
    }

    #[test]
    fn excess_calibration_uses_saved_baseline_and_native_state() {
        let (registry, worker) = calibration_registry(false);
        let admission = registry.estimated_wait();
        let probes: Vec<_> = (0..4)
            .map(|_| calibration_probe(admission, &worker))
            .collect();
        let mut report = load(Some(1_000), 2, 100.0, 0.0);
        report.loads[0].low_load_ttft_seconds = Some(9.0);
        report.loads[0].prefill_capacity_learning_active = Some(false);
        report.loads[0].prefill_capacity_sample_count = Some(8);
        publish_load(admission, &worker, &report);
        for probe in probes {
            admission.observe_ttft(probe, 20.2, 0.0);
        }
        let entries = admission.entries.lock();
        assert_eq!(entries[worker.url()].calibration.target(false), 2.0);
        assert_eq!(
            entries[worker.url()]
                .calibration
                .stats(false, Instant::now())
                .1,
            4
        );
        assert_eq!(
            entries[worker.url()]
                .calibration
                .stats(true, Instant::now()),
            (1.0, 0)
        );
    }

    #[test]
    fn excess_calibration_resets_with_lifecycle_and_native_profile() {
        let (registry, worker) = calibration_registry(true);
        let admission = registry.estimated_wait();
        let probes: Vec<_> = (0..4)
            .map(|_| calibration_probe(admission, &worker))
            .collect();
        for probe in probes {
            admission.observe_ttft(probe, 20.2, 0.0);
        }
        assert_eq!(
            admission.entries.lock()[worker.url()]
                .calibration
                .target(true),
            2.0
        );
        assert_eq!(
            admission.entries.lock()[worker.url()]
                .calibration
                .stats(true, Instant::now())
                .1,
            4
        );
        let late = calibration_probe(admission, &worker);
        let old_epoch = late.calibration.as_ref().unwrap().epoch;
        let watermark = admission.poll_started(&worker);
        admission.publish(&worker, None, watermark, Instant::now());
        let mut report = load(Some(1_000), 2, 100.0, 0.0);
        report.loads[0].prefill_capacity_sample_count = Some(1);
        publish_load(admission, &worker, &report);
        admission.observe_ttft(late, 40.0, 0.0);
        let reset = admission
            .begin()
            .unwrap()
            .capture_calibration(&worker, None)
            .unwrap();
        assert_eq!(reset.epoch, old_epoch + 1);
        assert_eq!((reset.scale, reset.target_scale), (1.0, 1.0));
        assert_eq!(reset.observation_sequence, 0);
        assert_eq!(
            admission.entries.lock()[worker.url()]
                .calibration
                .stats(true, Instant::now()),
            (1.0, 0)
        );
        for evict in [true, false] {
            let probe = calibration_probe(admission, &worker);
            if evict {
                admission.evict(&worker);
            } else {
                admission.clear();
            }
            admission.observe_ttft(probe, 20.2, 0.0);
            assert_eq!(
                admission.entries.lock()[worker.url()]
                    .calibration
                    .stats(true, Instant::now()),
                (1.0, 0)
            );
            publish_load(admission, &worker, &report);
        }
    }

    #[test]
    fn excess_calibration_never_changes_raw_admission_or_promotes_readiness() {
        for shadow in [false, true] {
            let (registry, worker) = calibration_registry(false);
            let admission = registry.estimated_wait();
            let mut entries = admission.entries.lock();
            let entry = entries.get_mut(worker.url()).unwrap();
            let now = Instant::now();
            let forecast = entry.calibration_forecast(admission.config(), now).unwrap();
            for _ in 0..4 {
                entry.calibration.observe(&forecast, 100.2, now);
            }
            let calibrated = entry
                .calibration_forecast(admission.config(), now + Duration::from_secs(15))
                .unwrap();
            assert_eq!(calibrated.target_scale, 10.0);
            assert_eq!(calibrated.scale, 2.5);
            assert!(!calibrated.raw_ready);
            assert!(!calibrated.would_reject());
            assert!(!entry.sample.as_ref().unwrap().prepared.capacity_ready);
            drop(entries);
            assert!(admission
                .begin()
                .unwrap()
                .check(std::slice::from_ref(&worker), "m")
                .is_ok());
            // Native-ready raw estimate is 10 s > 2 s; calibration predicts 0,
            // but hard mode must still reject exactly as the raw policy does.
            let mut cfg = config();
            cfg.estimated_wait_shadow = shadow;
            let unchanged = EstimatedWaitAdmission::default();
            unchanged.configure(EstimatedWaitConfig {
                estimated_wait_calibration_shadow: true,
                ..cfg
            });
            publish(&unchanged, &worker, 1_000);
            let mut entries = unchanged.entries.lock();
            let entry = entries.get_mut(worker.url()).unwrap();
            let now = Instant::now();
            let forecast = entry.calibration_forecast(unchanged.config(), now).unwrap();
            for _ in 0..4 {
                entry.calibration.observe(&forecast, 0.01, now);
            }
            assert!(
                entry
                    .calibration_forecast(unchanged.config(), now + Duration::from_secs(10))
                    .unwrap()
                    .extra()
                    < 2.0
            );
            drop(entries);
            assert_eq!(
                unchanged
                    .begin()
                    .unwrap()
                    .check(std::slice::from_ref(&worker), "m")
                    .is_err(),
                !shadow
            );
        }
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
    fn calibrated_coefficients_match_formula_and_defaults_match_legacy() {
        let state = load(Some(300), 0, 100.0, 0.5);
        let legacy = config().score(&state, 100).unwrap().0;
        assert!((legacy - 4.0).abs() < f64::EPSILON);

        let calibrated = EstimatedWaitConfig {
            estimated_wait_base_overhead_secs: 0.25,
            estimated_wait_queue_work_correction: 0.5,
            estimated_wait_dispatch_blocking_factor: 0.25,
            ..config()
        };
        // 0.25 + 0.5 * (300 + 0.25 * 100) / 100
        assert!((calibrated.score(&state, 100).unwrap().0 - 1.875).abs() < 1e-10);
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
    fn admission_budget_excludes_intrinsic_low_load_latency() {
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
            .is_ok());
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
    fn native_queue_delay_establishes_overload_without_baseline_or_capacity_guess() {
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
            .is_err());
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
            request_id: None,
        };
        assert_eq!(
            observer.scan(
                b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
                started + std::time::Duration::from_secs(1)
            ),
            None
        );
        assert_eq!(
            observer.scan(
                b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hel",
                started + std::time::Duration::from_secs(2)
            ),
            None
        );
        assert_eq!(
            observer.scan(b"lo\"}}]}\n\n", started + std::time::Duration::from_secs(3)),
            Some((3.0, 2.0))
        );
        assert_eq!(
            observer.scan(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"later\"}}]}\n\n",
                started + std::time::Duration::from_secs(4)
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
            request_id: None,
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
    fn configuration_rejects_invalid_numbers_and_defaults_to_disabled() {
        let admission = EstimatedWaitAdmission::default();
        admission.configure(EstimatedWaitConfig::default());
        assert!(admission.begin().is_none());
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
        assert!(!decoded.estimated_wait_shadow);
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

            // Shadow mode does not remove the independent static veto.
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
                placement::failure_from(&pool, "m"),
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
                rendered.contains("smg_estimated_wait_seconds{worker=\"http://a:1\"} 1"),
                "{rendered}"
            );
            assert!(
                rendered.contains("smg_estimated_wait_data_usable{worker=\"http://a:1\"} 1"),
                "{rendered}"
            );

            let assert_invalidated = |rendered: &str| {
                for metric in [
                    "smg_estimated_wait_seconds",
                    "smg_estimated_wait_threshold_seconds",
                    "smg_estimated_wait_queue_proxy",
                    "smg_estimated_wait_throughput_fallback",
                    "smg_estimated_wait_snapshot_age_seconds",
                    "smg_estimated_wait_blended_prompt_tokens",
                    "smg_estimated_wait_effective_prefill_capacity",
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
        registry
            .estimated_wait()
            .configure(EstimatedWaitConfig::default());
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
