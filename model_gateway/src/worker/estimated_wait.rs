//! Calibrated worker-backlog admission, independent of the routing policy.
//!
//! The ledger lock spans admission and selection/credit. Polls capture a
//! watermark before I/O so admissions during a poll survive its publication.

use std::{
    collections::HashMap,
    sync::{Arc, OnceLock, Weak},
    time::Instant,
};

use axum::response::Response;
use openai_protocol::worker::WorkerLoadResponse;
use parking_lot::{Mutex, MutexGuard};
use serde::{Deserialize, Serialize};

use super::Worker;
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
    pub estimated_wait_mean_prefill_tokens: u32,
    pub estimated_wait_default_throughput: f64,
    /// Zero disables the waiting-request compatibility proxy.
    pub estimated_wait_queue_tokens_per_request: u32,
    pub estimated_wait_max_snapshot_age_secs: f64,
}

impl Default for EstimatedWaitConfig {
    fn default() -> Self {
        Self {
            max_estimated_wait_secs: None,
            estimated_wait_kv_pressure_weight: 0.15,
            estimated_wait_mean_prefill_tokens: 1024,
            estimated_wait_default_throughput: 2000.0,
            estimated_wait_queue_tokens_per_request: 0,
            estimated_wait_max_snapshot_age_secs: 30.0,
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
                "estimated_wait_default_throughput",
                self.estimated_wait_default_throughput,
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
        Ok(())
    }

    fn score(&self, load: &WorkerLoadResponse, dispatched: u64) -> Option<(f64, bool, bool)> {
        if load.loads.is_empty()
            || (load.dp_rank_count > 0 && load.dp_rank_count as usize != load.loads.len())
            || load.loads.iter().any(|rank| {
                rank.num_waiting_reqs < 0
                    || rank.num_waiting_uncached_tokens.is_some_and(|n| n < 0)
                    || !rank.token_usage.is_finite()
            })
        {
            return None;
        }
        let mut proxy = false;
        let mut queued = 0.0;
        for rank in &load.loads {
            queued += match rank.num_waiting_uncached_tokens {
                Some(tokens) => f64::from(tokens),
                None => {
                    if self.estimated_wait_queue_tokens_per_request == 0 {
                        return None;
                    }
                    proxy = true;
                    f64::from(rank.num_waiting_reqs)
                        * f64::from(self.estimated_wait_queue_tokens_per_request)
                }
            };
        }
        let live = load.total_gen_throughput();
        let fallback = !live.is_finite()
            || live <= 0.0
            || load.loads.iter().any(|rank| rank.gen_throughput < 0.0);
        let throughput = if fallback {
            self.estimated_wait_default_throughput
        } else {
            live
        };
        let k = load.effective_token_usage().clamp(0.0, 0.999);
        let wait = (queued + dispatched as f64) / throughput
            + self.estimated_wait_kv_pressure_weight * k / (1.0 - k);
        wait.is_finite().then_some((wait, proxy, fallback))
    }
}

#[derive(Debug)]
struct Entry {
    source: Weak<dyn Worker>,
    generation: Arc<()>,
    total_dispatched: u64,
    snapshot: Option<(WorkerLoadResponse, Instant, u64)>,
}

#[derive(Debug, Default)]
pub(crate) struct EstimatedWaitAdmission {
    config: OnceLock<EstimatedWaitConfig>,
    entries: Mutex<HashMap<String, Entry>>,
}

impl EstimatedWaitAdmission {
    pub(crate) fn configure(&self, config: EstimatedWaitConfig) {
        if config.max_estimated_wait_secs.is_some() {
            let _ = self.config.set(config);
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.config.get().is_some()
    }

    fn entry<'a>(
        entries: &'a mut HashMap<String, Entry>,
        worker: &Arc<dyn Worker>,
    ) -> &'a mut Entry {
        let entry = entries
            .entry(worker.url().to_owned())
            .or_insert_with(|| Entry {
                source: Arc::downgrade(worker),
                generation: Arc::new(()),
                total_dispatched: 0,
                snapshot: None,
            });
        if !entry.source.ptr_eq(&Arc::downgrade(worker)) {
            *entry = Entry {
                source: Arc::downgrade(worker),
                generation: Arc::new(()),
                total_dispatched: 0,
                snapshot: None,
            };
        }
        entry
    }

    pub(crate) fn poll_started(&self, worker: &Arc<dyn Worker>) -> Option<PollWatermark> {
        if !self.enabled() {
            return None;
        }
        let mut entries = self.entries.lock();
        let entry = Self::entry(&mut entries, worker);
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
        let mut entries = self.entries.lock();
        let Some(entry) = entries.get_mut(worker.url()) else {
            return;
        };
        // A late poll must not recreate evicted state, overwrite a replacement,
        // or cross an unready -> ready transition of the same worker Arc.
        if !Arc::ptr_eq(&entry.generation, &watermark.generation)
            || entry
                .snapshot
                .as_ref()
                .is_some_and(|(_, sampled, _)| *sampled > started)
        {
            return;
        }
        entry.snapshot = load.map(|load| (load.clone(), started, watermark.dispatched));
    }

    pub(crate) fn evict(&self, worker: &Arc<dyn Worker>) {
        if !self.enabled() {
            return;
        }
        let mut entries = self.entries.lock();
        if entries
            .get(worker.url())
            .is_some_and(|entry| entry.source.ptr_eq(&Arc::downgrade(worker)))
        {
            entries.remove(worker.url());
            metrics::gauge!("smg_estimated_wait_data_usable", "worker" => worker.url().to_owned())
                .set(0.0);
        }
    }

    pub(crate) fn clear(&self) {
        let mut entries = self.entries.lock();
        for url in entries.keys() {
            metrics::gauge!("smg_estimated_wait_data_usable", "worker" => url.clone()).set(0.0);
        }
        entries.clear();
    }

    /// No lock or map scan when disabled. Hold the returned guard until every
    /// required pool has passed, selection has completed, and credit is added.
    pub(crate) fn begin(&self) -> Option<AdmissionGuard<'_>> {
        Some(AdmissionGuard {
            config: self.config.get()?,
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
        let threshold = self.config.max_estimated_wait_secs.unwrap_or(f64::INFINITY);
        metrics::gauge!("smg_estimated_wait_threshold_seconds", "model" => model.to_owned())
            .set(threshold);
        let mut minimum = f64::INFINITY;
        let mut unknown = false;
        let mut eligible = false;
        for worker in candidates.iter().filter(|w| w.is_available()) {
            eligible = true;
            let entry = self
                .entries
                .get(worker.url())
                .filter(|e| e.source.ptr_eq(&Arc::downgrade(worker)));
            let snapshot = entry.and_then(|e| e.snapshot.as_ref());
            let mut reason = "missing";
            let score = snapshot.and_then(|(load, sampled, watermark)| {
                let age = sampled.elapsed().as_secs_f64();
                metrics::gauge!("smg_estimated_wait_snapshot_age_seconds", "worker" => worker.url().to_owned()).set(age);
                if age > self.config.estimated_wait_max_snapshot_age_secs { reason = "stale"; return None; }
                reason = "unusable";
                let dispatched = entry.map_or(0, |e| e.total_dispatched.saturating_sub(*watermark));
                self.config.score(load, dispatched)
            });
            metrics::gauge!("smg_estimated_wait_data_usable", "worker" => worker.url().to_owned())
                .set(u8::from(score.is_some()));
            if let Some((wait, proxy, fallback)) = score {
                metrics::gauge!("smg_estimated_wait_seconds", "worker" => worker.url().to_owned())
                    .set(wait);
                metrics::gauge!("smg_estimated_wait_queue_proxy", "worker" => worker.url().to_owned()).set(u8::from(proxy));
                metrics::gauge!("smg_estimated_wait_throughput_fallback", "worker" => worker.url().to_owned()).set(u8::from(fallback));
                minimum = minimum.min(wait);
            } else {
                unknown = true;
                metrics::counter!("smg_estimated_wait_unknown_total", "model" => model.to_owned(), "reason" => reason)
                    .increment(1);
            }
        }
        // A partial pool cannot prove that every eligible worker is saturated.
        if !eligible || unknown || minimum < threshold {
            return Ok(());
        }
        metrics::counter!("smg_estimated_wait_rejections_total", "model" => model.to_owned())
            .increment(1);
        Err(overload::shed_estimated_wait(model, threshold))
    }

    pub(crate) fn credit(&mut self, worker: &Arc<dyn Worker>, tokens: Option<&[u32]>) {
        let count = tokens.map_or_else(
            || u64::from(self.config.estimated_wait_mean_prefill_tokens),
            |t| t.len() as u64,
        );
        let entry = EstimatedWaitAdmission::entry(&mut self.entries, worker);
        entry.total_dispatched = entry.total_dispatched.saturating_add(count);
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
        routers::common::{
            placement::{self, PairCandidates, PlacementFailure, PlacementInputs},
            retry::is_retryable_response,
            error,
        },
        worker::{BasicWorkerBuilder, ConnectionMode, WorkerRegistry, WorkerType},
    };

    fn config() -> EstimatedWaitConfig {
        EstimatedWaitConfig {
            max_estimated_wait_secs: Some(2.0),
            estimated_wait_kv_pressure_weight: 0.0,
            estimated_wait_mean_prefill_tokens: 100,
            estimated_wait_default_throughput: 100.0,
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
                num_waiting_uncached_tokens: tokens,
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
    fn formula_uses_exact_tokens_live_throughput_and_mean_rank_kv() {
        let config = EstimatedWaitConfig {
            estimated_wait_kv_pressure_weight: 0.5,
            ..config()
        };
        let mut state = load(Some(100), 80, 50.0, 0.25);
        state.loads.push(SchedulerLoadSnapshot {
            num_waiting_uncached_tokens: Some(200),
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
            num_waiting_uncached_tokens: None,
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
                estimated_wait_default_throughput: bad,
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
    fn concurrent_placements_credit_once_and_recover_on_fresh_poll() {
        let registry = Arc::new(WorkerRegistry::new());
        registry.estimated_wait.configure(config());
        let worker = worker("http://a:1");
        publish(&registry.estimated_wait, &worker, 0);
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
        publish(&registry.estimated_wait, &worker, 0);
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
    fn pd_sheds_the_saturated_leg_without_crediting_the_other_leg() {
        let registry = WorkerRegistry::new();
        registry.estimated_wait.configure(config());
        let policies = PolicyRegistry::new(PolicyConfig::RoundRobin);
        let prefill = worker("http://prefill:1");
        let decode = worker("http://decode:1");
        publish(&registry.estimated_wait, &prefill, 0);
        publish(&registry.estimated_wait, &decode, 200);
        let failure = placement::select_pair(
            &registry,
            &policies,
            "m",
            PairCandidates {
                prefill: std::slice::from_ref(&prefill),
                decode: std::slice::from_ref(&decode),
            },
            None,
            false,
            PlacementInputs::default(),
        )
        .err()
        .unwrap();
        match failure.verdict {
            PlacementFailure::AllOverloaded(shed) => {
                assert_eq!(shed.status(), StatusCode::SERVICE_UNAVAILABLE);
            }
            _ => panic!("expected an admission shed"),
        }
        let guard = registry.estimated_wait.begin().unwrap();
        assert_eq!(
            guard.entries.get(prefill.url()).unwrap().total_dispatched,
            0
        );
    }

    #[test]
    fn native_json_preserves_absent_and_exact_zero_tokens() {
        let absent: SchedulerLoadSnapshot = serde_json::from_str("{}").unwrap();
        let empty: SchedulerLoadSnapshot =
            serde_json::from_str(r#"{"num_waiting_uncached_tokens":0}"#).unwrap();
        assert_eq!(absent.num_waiting_uncached_tokens, None);
        assert_eq!(empty.num_waiting_uncached_tokens, Some(0));
    }
}
