//! Calibrated worker-backlog admission, independent of the routing policy.
//!
//! The ledger lock spans admission and selection/credit. Polls capture a
//! watermark before I/O so admissions during a poll survive its publication.

use std::{
    collections::HashMap,
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
        ExpectedWait, DEFAULT_KV_PRESSURE_WEIGHT, DEFAULT_MEAN_PREFILL_TOKENS, DEFAULT_THROUGHPUT,
    },
    Worker,
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
    pub estimated_wait_kv_pressure_weight: f64,
    pub estimated_wait_kv_pressure_threshold: f64,
    pub estimated_wait_base_overhead_secs: f64,
    pub estimated_wait_queue_work_correction: f64,
    pub estimated_wait_dispatch_blocking_factor: f64,
    pub estimated_wait_mean_prefill_tokens: u32,
    #[serde(
        alias = "estimated_wait_default_throughput",
        alias = "estimated_wait_min_prefill_throughput"
    )]
    pub estimated_wait_fallback_prefill_throughput: f64,
    pub estimated_wait_prompt_size_prior_samples: u32,
    /// Zero disables the waiting-request compatibility proxy.
    pub estimated_wait_queue_tokens_per_request: u32,
    pub estimated_wait_max_snapshot_age_secs: f64,
    /// Record would-reject decisions without enforcing estimated-wait budgets.
    pub estimated_wait_shadow: bool,
}

impl Default for EstimatedWaitConfig {
    fn default() -> Self {
        Self {
            max_estimated_wait_secs: None,
            estimated_wait_kv_pressure_weight: DEFAULT_KV_PRESSURE_WEIGHT,
            estimated_wait_kv_pressure_threshold: 0.0,
            estimated_wait_base_overhead_secs: 0.0,
            estimated_wait_queue_work_correction: 1.0,
            estimated_wait_dispatch_blocking_factor: 1.0,
            estimated_wait_mean_prefill_tokens: DEFAULT_MEAN_PREFILL_TOKENS,
            estimated_wait_fallback_prefill_throughput: DEFAULT_THROUGHPUT,
            estimated_wait_prompt_size_prior_samples: 32,
            estimated_wait_queue_tokens_per_request: 0,
            estimated_wait_max_snapshot_age_secs: 30.0,
            estimated_wait_shadow: false,
        }
    }
}

impl EstimatedWaitConfig {
    pub fn validate(&self) -> ConfigResult<()> {
        for (field, value, allow_zero) in [
            (
                "max_estimated_wait_secs",
                self.max_estimated_wait_secs.unwrap_or(1.0),
                false,
            ),
            (
                "estimated_wait_kv_pressure_weight",
                self.estimated_wait_kv_pressure_weight,
                true,
            ),
            (
                "estimated_wait_kv_pressure_threshold",
                self.estimated_wait_kv_pressure_threshold,
                true,
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
        if self.estimated_wait_kv_pressure_threshold > 1.0 {
            return Err(ConfigError::InvalidValue {
                field: "estimated_wait_kv_pressure_threshold".to_owned(),
                value: self.estimated_wait_kv_pressure_threshold.to_string(),
                reason: "Must be finite and between 0 and 1".to_owned(),
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
            || load.loads.iter().any(|rank| {
                rank.num_waiting_reqs < 0
                    || rank.num_waiting_uncached_tokens < 0
                    || !rank.token_usage.is_finite()
            })
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
                        .median_request_prefill_kv_computed_tokens
                        .filter(|value| value.is_finite() && *value > 0.0)
                        .zip(rank.prefill_size_sample_count.filter(|count| *count > 0));
                    let configured_prior = if self.estimated_wait_queue_tokens_per_request > 0 {
                        f64::from(self.estimated_wait_queue_tokens_per_request)
                    } else {
                        f64::from(self.estimated_wait_mean_prefill_tokens)
                    };
                    let calibrated_vllm = rank.prefill_capacity_learning_active.is_some();
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
                            if calibrated_vllm {
                                None
                            } else {
                                rank.avg_request_prefill_kv_computed_tokens
                                    .filter(|value| value.is_finite() && *value > 0.0)
                            }
                        })
                        .unwrap_or_else(|| {
                            if calibrated_vllm {
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
        let prefill = load
            .total_prefill_throughput()
            .filter(|value| value.is_finite() && *value > 0.0);
        let generation = load.total_gen_throughput();
        let generation = (generation.is_finite()
            && generation > 0.0
            && load.loads.iter().all(|rank| rank.gen_throughput >= 0.0))
        .then_some(generation);
        let live = prefill.or(generation);
        let capacity_learning_backend = load
            .loads
            .iter()
            .any(|rank| rank.prefill_capacity_learning_active.is_some());
        let learned = load.loads.iter().try_fold(0.0, |sum, rank| {
            rank.learned_prefill_capacity.map(|value| sum + value)
        });
        // vLLM counter rates are observed demand, not capacity. Only saturated
        // intervals train its capacity estimate; cold start and sparse traffic
        // stay on the configured fallback. Native load endpoints retain their
        // existing live-throughput behavior.
        let candidate = if capacity_learning_backend {
            learned
        } else {
            live
        };
        let (throughput, fallback) = match candidate {
            Some(value) if value.is_finite() && value > 0.0 => (value, false),
            _ => (self.estimated_wait_fallback_prefill_throughput, true),
        };
        let estimate = ExpectedWait::calibrated(
            queued,
            throughput,
            load.effective_token_usage(),
            self.estimated_wait_base_overhead_secs,
            self.estimated_wait_queue_work_correction,
            self.estimated_wait_dispatch_blocking_factor,
            self.estimated_wait_kv_pressure_threshold,
            self.estimated_wait_kv_pressure_weight,
        );
        estimate.seconds(0).is_finite().then_some(PreparedWait {
            estimate,
            proxy,
            fallback,
            blended_prompt_size,
            effective_prefill_capacity: throughput,
        })
    }

    #[cfg(test)]
    fn score(&self, load: &WorkerLoadResponse, dispatched: u64) -> Option<(f64, bool, bool)> {
        let prepared = self.prepare(load)?;
        let seconds = prepared.estimate.seconds(dispatched);
        seconds
            .is_finite()
            .then_some((seconds, prepared.proxy, prepared.fallback))
    }
}

#[derive(Clone, Copy, Debug)]
struct PreparedWait {
    estimate: ExpectedWait,
    proxy: bool,
    fallback: bool,
    blended_prompt_size: Option<f64>,
    effective_prefill_capacity: f64,
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
}

impl Entry {
    fn new(worker: &Arc<dyn Worker>) -> Self {
        Self {
            source: Arc::downgrade(worker),
            generation: Arc::new(()),
            total_dispatched: 0,
            last_published: None,
            sample: None,
        }
    }

    fn invalidate(&mut self) {
        self.generation = Arc::new(());
        self.total_dispatched = 0;
        self.last_published = None;
        self.sample = None;
    }

    fn refresh(&mut self, worker: &Arc<dyn Worker>, max_age_secs: f64) {
        if let Some(sample) = &mut self.sample {
            let seconds = sample
                .prepared
                .estimate
                .seconds(self.total_dispatched.saturating_sub(sample.watermark));
            sample.overloaded = seconds.is_finite().then_some(seconds >= sample.threshold);
            metrics::gauge!("smg_estimated_wait_seconds", "worker" => worker.url().to_owned())
                .set(seconds);
            metrics::gauge!("smg_estimated_wait_snapshot_age_seconds", "worker" => worker.url().to_owned()).set(sample.started.elapsed().as_secs_f64());
        }
        metrics::gauge!("smg_estimated_wait_data_usable", "worker" => worker.url().to_owned()).set(
            u8::from(self.sample.as_ref().is_some_and(|s| {
                s.overloaded.is_some() && s.started.elapsed().as_secs_f64() <= max_age_secs
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
            metrics::gauge!("smg_estimated_wait_data_usable", "worker" => worker.url().to_owned())
                .set(0.0);
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
            metrics::gauge!("smg_estimated_wait_data_usable", "worker" => worker.url().to_owned())
                .set(0.0);
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
        entry.sample = prepared.zip(threshold).map(|(prepared, threshold)| Sample {
            prepared,
            started,
            watermark: watermark.dispatched,
            threshold,
            overloaded: None,
        });
        if let Some(sample) = &entry.sample {
            metrics::gauge!("smg_estimated_wait_threshold_seconds", "worker" => worker.url().to_owned()).set(sample.threshold);
            metrics::gauge!("smg_estimated_wait_queue_proxy", "worker" => worker.url().to_owned())
                .set(u8::from(sample.prepared.proxy));
            metrics::gauge!("smg_estimated_wait_throughput_fallback", "worker" => worker.url().to_owned()).set(u8::from(sample.prepared.fallback));
            metrics::gauge!("smg_estimated_wait_blended_prompt_tokens", "worker" => worker.url().to_owned())
                .set(sample.prepared.blended_prompt_size.unwrap_or(-1.0));
            metrics::gauge!("smg_estimated_wait_effective_prefill_capacity", "worker" => worker.url().to_owned())
                .set(sample.prepared.effective_prefill_capacity);
        }
        entry.refresh(worker, config.estimated_wait_max_snapshot_age_secs);
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
            metrics::gauge!("smg_estimated_wait_data_usable", "worker" => worker.url().to_owned())
                .set(0.0);
        }
    }

    pub(crate) fn clear(&self) {
        let mut entries = self.entries.lock();
        for (url, entry) in entries.iter_mut() {
            entry.invalidate();
            metrics::gauge!("smg_estimated_wait_data_usable", "worker" => url.clone()).set(0.0);
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
                    metrics::gauge!("smg_estimated_wait_data_usable", "worker" => worker.url().to_owned()).set(0.0);
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
        entry.refresh(worker, self.config.estimated_wait_max_snapshot_age_secs);
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
                token_usage: kv,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn publish(admission: &EstimatedWaitAdmission, worker: &Arc<dyn Worker>, tokens: i32) {
        let stamp = admission.poll_started(worker);
        admission.publish(
            worker,
            Some(&load(Some(tokens), 0, 100.0, 0.0)),
            stamp,
            Instant::now(),
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
        assert_eq!(config.score(&report, 0), Some((10.0, true, false)));
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
                (10.0, true, false)
            } else {
                (0.0, false, false)
            };
            assert_eq!(config.score(&report, 0), Some(expected));
        }
    }

    #[test]
    fn vllm_dynamic_prefill_metrics_override_static_fallbacks() {
        let config = EstimatedWaitConfig {
            estimated_wait_queue_tokens_per_request: 100,
            estimated_wait_fallback_prefill_throughput: 10.0,
            ..config()
        };
        let mut report = load(None, 2, 50.0, 0.0);
        report.loads[0].prefill_throughput = Some(200.0);
        report.loads[0].avg_request_prefill_kv_computed_tokens = Some(1000.0);
        assert_eq!(config.score(&report, 0), Some((10.0, true, false)));
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

        assert_eq!(config().score(&report, 0), Some((1.0, true, true)));
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
            estimated_wait_kv_pressure_threshold: 0.4,
            estimated_wait_kv_pressure_weight: 2.0,
            ..config()
        };
        // 0.25 + 0.5 * (300 + 0.25 * 100) / 100 + 2 * (0.5 - 0.4) / 0.5
        assert!((calibrated.score(&state, 100).unwrap().0 - 2.275).abs() < 1e-10);
    }

    #[test]
    fn qualified_learned_capacity_replaces_fallback_even_when_lower() {
        let mut report = load(None, 0, 10.0, 0.0);
        report.loads[0].avg_request_prefill_kv_computed_tokens = Some(100.0);
        report.loads[0].prefill_capacity_learning_active = Some(false);
        assert_eq!(config().score(&report, 100), Some((1.0, true, true)));

        report.loads[0].learned_prefill_capacity = Some(40.0);
        report.loads[0].prefill_capacity_sample_count = Some(8);
        assert_eq!(config().score(&report, 100), Some((2.5, true, false)));
    }

    #[test]
    fn formula_uses_exact_tokens_live_throughput_and_mean_rank_kv() {
        let config = EstimatedWaitConfig {
            estimated_wait_kv_pressure_weight: 0.5,
            ..config()
        };
        let mut state = load(Some(100), 80, 50.0, 0.25);
        state.loads.push(SchedulerLoadSnapshot {
            num_waiting_uncached_tokens: 200,
            gen_throughput: 50.0,
            token_usage: 0.75,
            ..Default::default()
        });
        assert_eq!(config.score(&state, 100), Some((4.5, false, false)));
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
        assert_eq!(config.score(&state, 0), Some((2.0, true, false)));
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
        assert!(config()
            .score(&load(Some(0), 0, 100.0, f64::NAN), 0)
            .is_none());
        let config = EstimatedWaitConfig {
            estimated_wait_kv_pressure_weight: 0.5,
            ..config()
        };
        let score = config.score(&load(Some(0), 0, 100.0, 1.0), 0).unwrap().0;
        assert!((score - 499.5).abs() < 1e-8);
    }

    #[test]
    fn configuration_rejects_invalid_numbers_and_defaults_to_disabled() {
        let admission = EstimatedWaitAdmission::default();
        admission.configure(EstimatedWaitConfig::default());
        assert!(admission.begin().is_none());
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
        assert!(EstimatedWaitConfig {
            estimated_wait_mean_prefill_tokens: 0,
            ..config()
        }
        .validate()
        .is_err());
        assert!(EstimatedWaitConfig {
            estimated_wait_kv_pressure_weight: -1.0,
            ..config()
        }
        .validate()
        .is_err());
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
