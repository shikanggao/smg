//! Candidate-only raw/native publication. Legacy raw calibration stays live.
//! A stored value is a source context, never a prediction or a frozen ledger.
use super::*;

#[derive(Clone, Debug)]
pub(crate) struct RawContext {
    sample: Sample,
    snapshot: decision::Snapshot,
    pending: pending_work::Admission,
    native_raw: survival::CandidateRawState,
    sg_proof: Option<SgRawEvidence>,
    calibration_epoch: u64,
    hard_generation: u64,
    raw_expiry: f64,
    native_stamp: Option<(u64, f64, f64, f64)>,
    native_waiting: Option<f64>,
    native_population: Option<f64>,
    pair_available: Option<f64>,
}

#[derive(Debug)]
pub(super) struct State {
    hard_generation: u64,
    issued: Option<RawContext>,
    published: Option<RawContext>,
    last_stamp: Option<(f64, f64, (u64, f64, f64, f64))>,
    reason: &'static str,
    last_safety: Option<RawSafety>,
}
impl Default for State {
    fn default() -> Self {
        Self {
            hard_generation: 0,
            issued: None,
            published: None,
            last_stamp: None,
            reason: "coherent_pair_cold_missing",
            last_safety: None,
        }
    }
}
impl State {
    pub fn revoke(&mut self, reason: &'static str) {
        self.hard_generation = self.hard_generation.saturating_add(1);
        self.issued = None;
        self.published = None;
        self.reason = reason;
    }
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct Diagnostic {
    pub method: &'static str,
    pub selected: bool,
    pub reason: &'static str,
    pub total_ttft_seconds: Option<f64>,
    pub raw_poll_started: Option<f64>,
    pub original_raw_expiry: Option<f64>,
    pub native_snapshot_seq: Option<u64>,
    pub native_raw_poll_started: Option<f64>,
    pub original_native_receipt: Option<f64>,
    pub original_native_expiry: Option<f64>,
    pub pair_available_at_gateway_seconds: Option<f64>,
    pub latest_raw_poll_started: Option<f64>,
    pub hard_generation: u64,
    pub calibration_epoch: u64,
    pub guarded_cutoff: f64,
    pub baseline_seconds: Option<f64>,
    pub capacity_basis: Option<survival::FrozenRawWork>,
    pub projection: Option<survival::Projection>,
    pub sg_raw_basis_deadline: Option<survival::SgProjection>,
    pub training_snapshot: Option<decision::Snapshot>,
    pub normalization: Option<survival::Normalization>,
    pub original_projection: Option<survival::Projection>,
    pub raw_safety: Option<RawSafety>,
}
#[derive(Clone, Debug, Serialize)]
pub(super) struct RawSafety {
    paired_capacity: f64,
    latest_capacity: f64,
    paired_waiting_basis: f64,
    latest_waiting: f64,
    live_local_unseen_rows: usize,
    paired_population_basis: f64,
    latest_population: f64,
}

fn diagnostic(entry: &Entry, at: f64) -> Diagnostic {
    let state = &entry.coherent_pair;
    let stamp = state
        .published
        .as_ref()
        .and_then(|c| {
            c.native_stamp
                .map(|n| (diagnostic_time(c.sample.started), c.raw_expiry, n))
        })
        .or(state.last_stamp);
    Diagnostic {
        method: "raw_native_coherent_candidate_v1",
        selected: false,
        reason: state.reason,
        total_ttft_seconds: None,
        raw_poll_started: stamp.map(|s| s.0),
        original_raw_expiry: stamp.map(|s| s.1),
        native_snapshot_seq: stamp.map(|s| s.2 .0),
        native_raw_poll_started: stamp.map(|s| s.2 .1),
        original_native_receipt: stamp.map(|s| s.2 .2),
        original_native_expiry: stamp.map(|s| s.2 .3),
        pair_available_at_gateway_seconds: state.published.as_ref().and_then(|c| c.pair_available),
        latest_raw_poll_started: entry.sample.as_ref().map(|s| diagnostic_time(s.started)),
        hard_generation: state.hard_generation,
        calibration_epoch: entry.calibration_epoch,
        guarded_cutoff: at,
        baseline_seconds: None,
        capacity_basis: None,
        projection: None,
        sg_raw_basis_deadline: None,
        training_snapshot: None,
        normalization: None,
        original_projection: None,
        raw_safety: state.last_safety.clone(),
    }
}

/// Called after the unchanged raw publication and its SG proof construction,
/// while the admission lock is still held. A bad publication is a sticky fence.
pub(super) fn raw_publication(
    entry: &mut Entry,
    config: &EstimatedWaitConfig,
    load: Option<&WorkerLoadResponse>,
    _watermark: &PollWatermark,
    started: Instant,
    success: bool,
    proof: Option<SgRawEvidence>,
) {
    if entry.coherent_pair.hard_generation == u64::MAX {
        entry.coherent_pair.revoke("coherent_generation_exhausted");
        return;
    }
    let sample = entry.sample.as_ref();
    let ready = success
        && entry.survival.native.candidate_raw_eligible(
            load,
            diagnostic_time(started),
            config.estimated_wait_max_snapshot_age_secs,
        )
        && sample.is_some_and(|s| {
            s.started == started
                && s.prepared.capacity_learned_ready
                && s.prepared.native_prefill_capacity.is_finite()
                && s.prepared.native_prefill_capacity > 0.0
                && s.prepared.size_basis.is_some()
        });
    if !ready {
        entry
            .coherent_pair
            .revoke("coherent_raw_negative_or_capacity_missing");
        return;
    }
    let sample = sample.unwrap();
    let old = entry
        .coherent_pair
        .published
        .as_ref()
        .or(entry.coherent_pair.issued.as_ref());
    let unseen = entry
        .survival
        .native
        .candidate_unseen_local_count(diagnostic_time(Instant::now()));
    entry.coherent_pair.last_safety = old.map(|old| RawSafety {
        paired_capacity: old.sample.prepared.native_prefill_capacity,
        latest_capacity: sample.prepared.native_prefill_capacity,
        paired_waiting_basis: old
            .sample
            .prepared
            .native_waiting
            .max(old.native_waiting.unwrap_or(0.0)),
        latest_waiting: sample.prepared.native_waiting,
        live_local_unseen_rows: unseen,
        paired_population_basis: (old.sample.prepared.native_waiting
            + old.sample.prepared.native_running)
            .max(old.native_population.unwrap_or(0.0)),
        latest_population: sample.prepared.native_waiting + sample.prepared.native_running,
    });
    let revoke = old.and_then(|old| {
        if old.calibration_epoch != entry.calibration_epoch
            || old.snapshot.baseline != entry.calibration_baseline.0.unwrap_or(f64::NAN)
            || old.sample.prepared.decision_profile != sample.prepared.decision_profile
            || old
                .sample
                .prepared
                .size_basis
                .map(|p| (p.dp_rank, p.counter_epoch, p.progress_source))
                != sample
                    .prepared
                    .size_basis
                    .map(|p| (p.dp_rank, p.counter_epoch, p.progress_source))
        {
            Some("coherent_raw_source_epoch_or_baseline_changed")
        } else if sample.prepared.native_prefill_capacity + 1e-9
            < old.sample.prepared.native_prefill_capacity
        {
            Some("coherent_new_raw_capacity_drop")
        } else {
            // New local dispatches are included by the live Native ledger.
            // Queue growth beyond that conservative allowance is new pressure.
            let waiting = old
                .sample
                .prepared
                .native_waiting
                .max(old.native_waiting.unwrap_or(0.0));
            let population = (old.sample.prepared.native_waiting
                + old.sample.prepared.native_running)
                .max(old.native_population.unwrap_or(0.0));
            if sample.prepared.native_waiting > waiting + unseen as f64 + 1e-9 {
                Some("coherent_unexplained_native_queue_growth")
            } else {
                (sample.prepared.native_waiting + sample.prepared.native_running
                    > population + unseen as f64 + 1e-9)
                    .then_some("coherent_unexplained_native_population_growth")
            }
        }
    });
    if let Some(reason) = revoke {
        entry.coherent_pair.revoke(reason);
    }
    if load
        .and_then(|l| l.loads.first())
        .and_then(|r| r.timely_prefill_progress_source.as_deref())
        == Some("prefill_started_requests")
        && proof.is_none()
    {
        if revoke.is_none() {
            entry
                .coherent_pair
                .revoke("coherent_sg_complete_raw_proof_missing");
        }
        return;
    }
    let now = Instant::now();
    let Some(snapshot) = entry.decision_snapshot(config, now) else {
        return;
    };
    let original = decision::Admission {
        snapshot,
        predictions: entry.decision.predictions(
            snapshot,
            snapshot.at,
            snapshot.fresh && snapshot.native_ready,
        ),
    };
    let Some(pending) = entry.pending_work_admission(config, &original, now) else {
        return;
    };
    let sample = entry.sample.as_ref().unwrap().clone();
    entry.coherent_pair.issued = Some(RawContext {
        raw_expiry: proof.as_ref().map_or(
            diagnostic_time(started) + config.estimated_wait_max_snapshot_age_secs,
            SgRawEvidence::original_expiry,
        ),
        sample,
        snapshot,
        pending,
        native_raw: entry.survival.native.candidate_raw_state(),
        sg_proof: proof,
        calibration_epoch: entry.calibration_epoch,
        hard_generation: entry.coherent_pair.hard_generation,
        native_stamp: None,
        native_waiting: None,
        native_population: None,
        pair_available: None,
    });
}

#[cfg(test)]
mod tests {
    use openai_protocol::worker::WorkerStatus;
    use serde_json::json;

    use super::*;
    use crate::worker::{BasicWorkerBuilder, ConnectionMode, WorkerType};

    fn raw(second: u64, zero: bool, runtime: RuntimeType) -> WorkerLoadResponse {
        let source = if runtime == RuntimeType::Sglang {
            "prefill_started_requests"
        } else {
            "local_compute_tokens"
        };
        let counter = 2.0 * second as f64;
        serde_json::from_value(json!({"dp_rank_count":1,"loads":[{
            "dp_rank":0,"native_request_counts_available":true,"native_request_counts_profile":7,
            "num_waiting_reqs":2,"num_running_reqs":1,"num_waiting_uncached_tokens_available":false,
            "avg_request_prefill_kv_computed_tokens":1000.0,"prefill_size_sample_count":8,
            "timely_prefill_progress_source":source,"timely_prefill_progress":if zero {0.0}else{2000.0},
            "timely_prefill_progress_interval":1.0,"prefill_capacity_learning_active":second>0 && !zero,
            "prefill_capacity_sample_count":if zero {second.saturating_sub(1).min(32)}else{second.min(32)},
            "learned_prefill_capacity":(second>=8).then_some(2000.0),
            "prefill_metric_provenance":{"schema_version":1,"counter_epoch":0,"progress_source":source,
                "raw_counter":if zero {counter-2.0}else{counter},"previous_raw_counter":(second>0).then_some(counter-2.0),
                "interval_seconds":(second>0).then_some(1.0),"interval_started_time_seconds":(second>0).then_some(second as f64-1.0),
                "interval_ended_time_seconds":second as f64,"raw_rate":(second>0).then_some(if zero {0.0}else{2.0}),
                "previous_conversion_size_mean":1000.0,"current_unshrunk_size_mean":1000.0,
                "queue_size_observed_sample_count":8,"native_size_counter_count":8.0,
                "capacity_interval_eligible":second>0 && !zero,"counter_reset":false}
        }]})).unwrap()
    }
    fn native(seq: u64, runtime: RuntimeType, remaining: f64) -> WorkerLoadResponse {
        let profile = if runtime == RuntimeType::Sglang {
            "sglang-c9e5799:test"
        } else {
            "vllm:test"
        };
        let original_prefill = json!({
            "schema_version":1,"diagnostic_only":true,
            "method":"sg_original_prefill_observation_v1","source_commit":"c9e5799bd12fd5fefb0fb375ac537a6b7d53892e",
            "native_request_id":"native-owned","attempt_id":"owned","state_seq":seq,
            "observed_at_monotonic_secs":100.0+seq as f64,"available":true,"supported_mode":true,
            "original_alias_verified":true,"original_prompt_tokens":1000,"normal_return_frontier_tokens":0,
            "committed_output_tokens":0,"lifetime_retraction_count":0,"input_embeds_none":true,
            "session_none":true,"multimodal_inputs_none":true,"is_retracted":false,"abort_pending":false
        });
        serde_json::from_value(json!({"dp_rank_count":1,"loads":[{"dp_rank":0,"deadline_recovery":{
            "schema_version":1,"engine_epoch":"paired-test","profile_id":profile,"dp_rank":0,
            "snapshot_seq":seq,"captured_at_monotonic_secs":100.0+seq as f64,"snapshot_age_seconds":0.0,
            "active_inventory_complete":true,"native_population":1,"scheduler_semantics":"continuous_batching",
            "deadline_boundary":"first_nonempty_text","deadline_origin":"engine_http_receive_monotonic",
            "remaining_work_source":"native_uncached","effective_capacity_tokens_per_sec":null,"capacity_source":null,
            "hard_ttft_deadline_supported":true,"cancellation_ack_supported":true,"abort_enforcement_enabled":true,
            "abort_max_lag_seconds":null,"ack_floor_seq":0,"cancellation_acks":[],
            "active_requests":[{"attempt_id":"owned","native_request_id":"native-owned","state_seq":seq,
                "remaining_prefill_work_tokens":remaining,"hard_ttft_deadline_monotonic_secs":401.0,
                "queue_position":0,"state":"waiting","first_text_emitted":false,
                "original_prefill_observation":original_prefill}]
        }}]})).unwrap()
    }
    fn estimate() -> pending_work::Estimate {
        pending_work::Estimate {
            tokens: Some(100.0),
            source: "utf8_bytes_div4_proxy",
            text_bytes: Some(400),
            uncached_tokens_known: false,
            uncertainty: "test",
        }
    }
    fn publish(
        registry: &WorkerRegistry,
        worker: &Arc<dyn Worker>,
        report: Option<&WorkerLoadResponse>,
    ) -> (PollWatermark, Instant) {
        let api = registry.estimated_wait();
        let mut mark = api.poll_started(worker).unwrap();
        let started = Instant::now();
        mark.sg_raw_evidence = api.publish_with_sg_raw(worker, report, Some(mark.clone()), started);
        mark.coherent_raw = api.coherent_raw_context(worker, started);
        (mark, started)
    }
    fn ready(runtime: RuntimeType) -> (WorkerRegistry, Arc<dyn Worker>) {
        let registry = WorkerRegistry::new();
        registry.estimated_wait().configure(EstimatedWaitConfig {
            max_estimated_wait_secs: Some(300.0),
            estimated_wait_calibration_shadow: true,
            estimated_wait_feature_shadow: true,
            estimated_wait_provisional_baseline_secs: Some(0.2),
            estimated_wait_fallback_prefill_throughput: 10.0,
            ..Default::default()
        });
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://coherent-pair-test:1")
                .status(WorkerStatus::Ready)
                .runtime_type(runtime)
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Http)
                .build(),
        );
        registry.register(worker.clone()).unwrap();
        let mut last = None;
        for second in 0..=9 {
            last = Some(publish(
                &registry,
                &worker,
                Some(&raw(second, false, runtime)),
            ));
        }
        registry
            .estimated_wait()
            .entries
            .lock()
            .get_mut(worker.url())
            .unwrap()
            .survival
            .native
            .register(Some("owned"), 0.0, estimate());
        let (mark, started) = last.unwrap();
        registry.estimated_wait().publish_native_deadline(
            &worker,
            Some(&native(1, runtime, 100.0)),
            Some(mark),
            started,
            Instant::now(),
            None,
        );
        assert!(registry.estimated_wait().entries.lock()[worker.url()]
            .coherent_pair
            .published
            .is_some());
        (registry, worker)
    }
    fn admission(registry: &WorkerRegistry, worker: &Arc<dyn Worker>) -> survival::Admission {
        let mut guard = registry.estimated_wait().begin().unwrap();
        let mut forecast = guard.capture_calibration(worker, None);
        assert!(guard.attach_feature_context(worker, &mut forecast));
        *forecast.unwrap().survival_admission.unwrap()
    }

    fn emit_fixture(
        label: &str,
        runtime: RuntimeType,
        configured_floor: f64,
        value: &survival::Admission,
    ) {
        if let Some(folder) = std::env::var_os("SMG_COHERENT_FIXTURE_DIR") {
            let folder = std::path::PathBuf::from(folder);
            std::fs::create_dir_all(&folder).unwrap();
            let backend = if runtime == RuntimeType::Sglang {
                "sglang"
            } else {
                "qwen"
            };
            std::fs::write(
                folder.join(format!("{backend}-{label}.json")),
                serde_json::to_vec_pretty(
                    &json!({"backend":backend,"configured_floor":configured_floor,
                    "admission":value}),
                )
                .unwrap(),
            )
            .unwrap();
        }
    }
    #[test]
    fn current_pair_exact_parity_and_zero_mismatch_both_backends() {
        for runtime in [RuntimeType::Sglang, RuntimeType::Vllm] {
            let (registry, worker) = ready(runtime);
            let original = admission(&registry, &worker);
            emit_fixture(
                "selected-current",
                runtime,
                registry
                    .estimated_wait()
                    .config()
                    .estimated_wait_fallback_prefill_throughput,
                &original,
            );
            let selected = original.paired_native_candidate.as_ref().unwrap();
            assert!(selected.selected, "{}", selected.reason);
            let expected = if runtime == RuntimeType::Sglang {
                original
                    .sg_raw_basis_deadline
                    .as_ref()
                    .unwrap()
                    .total_ttft_seconds
            } else {
                original.projection.total_ttft_seconds.unwrap()
            };
            assert_eq!(selected.total_ttft_seconds, Some(expected));
            publish(&registry, &worker, Some(&raw(10, true, runtime)));
            let held_admission = admission(&registry, &worker);
            emit_fixture(
                "selected-held",
                runtime,
                registry
                    .estimated_wait()
                    .config()
                    .estimated_wait_fallback_prefill_throughput,
                &held_admission,
            );
            let held = held_admission.paired_native_candidate.unwrap();
            assert!(held.selected, "{}", held.reason);
            assert_eq!(held.reason, "coherent_held_pair_live_ledger_projection");
            assert!(held.raw_poll_started.unwrap() < held.latest_raw_poll_started.unwrap());
        }
    }
    #[test]
    fn missing_counts_and_malformed_counter_are_sticky_across_positive_and_old_callback() {
        for runtime in [RuntimeType::Sglang, RuntimeType::Vllm] {
            for mutation in 0..3 {
                let (registry, worker) = ready(runtime);
                let (old, started) = publish(&registry, &worker, Some(&raw(10, true, runtime)));
                let mut bad = raw(11, false, runtime);
                match mutation {
                    0 => bad.loads[0].native_request_counts_available = Some(false),
                    1 => {
                        bad.loads[0]
                            .prefill_metric_provenance
                            .as_mut()
                            .unwrap()
                            .schema_version = 99
                    }
                    _ => {
                        let p = bad.loads[0].prefill_metric_provenance.as_mut().unwrap();
                        p.raw_counter = None;
                        p.previous_raw_counter = None;
                        p.raw_rate = Some(0.0);
                        p.capacity_interval_eligible = false;
                    }
                }
                publish(&registry, &worker, Some(&bad));
                publish(&registry, &worker, Some(&raw(12, false, runtime)));
                for second in 13..=45 {
                    publish(&registry, &worker, Some(&raw(second, false, runtime)));
                }
                assert!(registry.estimated_wait().entries.lock()[worker.url()]
                    .coherent_pair
                    .issued
                    .is_some());
                registry.estimated_wait().publish_native_deadline(
                    &worker,
                    Some(&native(2, runtime, 50.0)),
                    Some(old),
                    started,
                    Instant::now(),
                    None,
                );
                assert!(registry.estimated_wait().entries.lock()[worker.url()]
                    .coherent_pair
                    .published
                    .is_none());
            }
        }
    }
    #[test]
    fn transport_timeout_revokes_pair_both_backends() {
        for runtime in [RuntimeType::Sglang, RuntimeType::Vllm] {
            let (registry, worker) = ready(runtime);
            let (mark, started) = publish(&registry, &worker, Some(&raw(10, true, runtime)));
            registry.estimated_wait().publish_native_deadline_timeout(
                &worker,
                Some(mark),
                started,
                Instant::now(),
                None,
            );
            let entries = registry.estimated_wait().entries.lock();
            assert!(entries[worker.url()].coherent_pair.published.is_none());
            assert_eq!(
                entries[worker.url()].coherent_pair.reason,
                "coherent_native_timeout"
            );
            drop(entries);
            emit_fixture(
                "unavailable",
                runtime,
                registry
                    .estimated_wait()
                    .config()
                    .estimated_wait_fallback_prefill_throughput,
                &admission(&registry, &worker),
            );
        }
    }
    #[test]
    fn zero_to_positive_keeps_context_and_accepts_its_own_late_native_pair() {
        for runtime in [RuntimeType::Sglang, RuntimeType::Vllm] {
            let (registry, worker) = ready(runtime);
            let (mark, started) = publish(&registry, &worker, Some(&raw(10, true, runtime)));
            let generation = registry.estimated_wait().entries.lock()[worker.url()]
                .coherent_pair
                .hard_generation;
            let mut positive = raw(11, false, runtime);
            positive.loads[0].prefill_capacity_sample_count = Some(10);
            {
                let p = positive.loads[0]
                    .prefill_metric_provenance
                    .as_mut()
                    .unwrap();
                p.raw_counter = Some(20.0);
                p.previous_raw_counter = Some(18.0);
            }
            publish(&registry, &worker, Some(&positive));
            assert_eq!(
                registry.estimated_wait().entries.lock()[worker.url()]
                    .coherent_pair
                    .hard_generation,
                generation
            );
            registry.estimated_wait().publish_native_deadline(
                &worker,
                Some(&native(2, runtime, 50.0)),
                Some(mark),
                started,
                Instant::now(),
                None,
            );
            let entries = registry.estimated_wait().entries.lock();
            assert!(entries[worker.url()].coherent_pair.published.is_some());
            assert_eq!(
                entries[worker.url()]
                    .coherent_pair
                    .published
                    .as_ref()
                    .unwrap()
                    .sample
                    .started,
                started
            );
        }
    }
    #[test]
    fn genuine_capacity_drop_and_unexplained_queue_growth_revoke_immediately() {
        for runtime in [RuntimeType::Sglang, RuntimeType::Vllm] {
            for capacity_drop in [true, false] {
                let (registry, worker) = ready(runtime);
                let mut new = raw(10, false, runtime);
                if capacity_drop {
                    new.loads[0].learned_prefill_capacity = Some(1000.0);
                } else {
                    new.loads[0].num_waiting_reqs = 100;
                }
                publish(&registry, &worker, Some(&new));
                let entries = registry.estimated_wait().entries.lock();
                let state = &entries[worker.url()].coherent_pair;
                assert!(state.published.is_none());
                assert_eq!(
                    state.reason,
                    if capacity_drop {
                        "coherent_new_raw_capacity_drop"
                    } else {
                        "coherent_unexplained_native_queue_growth"
                    }
                );
            }
        }
    }
    #[test]
    fn local_dispatch_and_first_text_remain_live_while_pair_is_held() {
        for runtime in [RuntimeType::Sglang, RuntimeType::Vllm] {
            let (registry, worker) = ready(runtime);
            publish(&registry, &worker, Some(&raw(10, true, runtime)));
            let before = admission(&registry, &worker)
                .paired_native_candidate
                .unwrap();
            {
                let mut entries = registry.estimated_wait().entries.lock();
                let entry = entries.get_mut(worker.url()).unwrap();
                entry.total_dispatched += 100;
                entry.total_dispatched_requests += 1;
                entry.survival.native.register(
                    Some("new"),
                    diagnostic_time(Instant::now()),
                    estimate(),
                );
            }
            let grown = admission(&registry, &worker)
                .paired_native_candidate
                .unwrap();
            assert!(grown.selected, "{}", grown.reason);
            assert_eq!(
                grown.projection.as_ref().unwrap().predecessors,
                before.projection.as_ref().unwrap().predecessors + 1
            );
            {
                let mut entries = registry.estimated_wait().entries.lock();
                entries
                    .get_mut(worker.url())
                    .unwrap()
                    .survival
                    .native
                    .first_text(Some("owned"), diagnostic_time(Instant::now()));
            }
            let text = admission(&registry, &worker)
                .paired_native_candidate
                .unwrap();
            assert!(text.selected);
            assert_eq!(
                text.projection.unwrap().predecessors,
                before.projection.unwrap().predecessors
            );
        }
    }
    #[test]
    fn foreign_running_growth_is_not_hidden_but_owned_dispatch_growth_is_covered() {
        for runtime in [RuntimeType::Sglang, RuntimeType::Vllm] {
            let (registry, worker) = ready(runtime);
            let mut foreign = raw(10, false, runtime);
            foreign.loads[0].num_running_reqs = 100;
            publish(&registry, &worker, Some(&foreign));
            assert_eq!(
                registry.estimated_wait().entries.lock()[worker.url()]
                    .coherent_pair
                    .reason,
                "coherent_unexplained_native_population_growth"
            );
            let (registry, worker) = ready(runtime);
            {
                let mut entries = registry.estimated_wait().entries.lock();
                let entry = entries.get_mut(worker.url()).unwrap();
                entry.total_dispatched += 100;
                entry.total_dispatched_requests += 1;
                entry.survival.native.register(
                    Some("own-new"),
                    diagnostic_time(Instant::now()),
                    estimate(),
                );
            }
            let mut own = raw(10, false, runtime);
            own.loads[0].num_waiting_reqs = 3;
            publish(&registry, &worker, Some(&own));
            assert!(registry.estimated_wait().entries.lock()[worker.url()]
                .coherent_pair
                .published
                .is_some());
            let mut movement = raw(11, false, runtime);
            movement.loads[0].num_waiting_reqs = 1;
            movement.loads[0].num_running_reqs = 2;
            publish(&registry, &worker, Some(&movement));
            assert!(registry.estimated_wait().entries.lock()[worker.url()]
                .coherent_pair
                .published
                .is_some());
        }
    }
    #[test]
    fn native_rebound_clears_pair_and_cannot_reuse_old_inventory() {
        for runtime in [RuntimeType::Sglang, RuntimeType::Vllm] {
            let (registry, worker) = ready(runtime);
            let (mark, started) = publish(&registry, &worker, Some(&raw(10, true, runtime)));
            registry.estimated_wait().publish_native_deadline(
                &worker,
                Some(&native(2, runtime, 150.0)),
                Some(mark),
                started,
                Instant::now(),
                None,
            );
            assert!(registry.estimated_wait().entries.lock()[worker.url()]
                .coherent_pair
                .published
                .is_none());
        }
    }
    #[test]
    fn native_ack_release_is_query_live_in_the_replacement_pair() {
        for runtime in [RuntimeType::Sglang, RuntimeType::Vllm] {
            let (registry, worker) = ready(runtime);
            let (mark, started) = publish(&registry, &worker, Some(&raw(10, true, runtime)));
            let mut released = native(2, runtime, 0.0);
            let lane = released.loads[0].deadline_recovery.as_mut().unwrap();
            lane.active_requests.clear();
            lane.native_population = 0;
            lane.cancellation_acks =
                vec![openai_protocol::native_deadline::NativeCancellationAck {
                    attempt_id: "owned".into(),
                    native_request_id: Some("native-owned".into()),
                    event_seq: 2,
                    ack_monotonic_secs: 102.0,
                    reason: "client_abort".into(),
                }];
            registry.estimated_wait().publish_native_deadline(
                &worker,
                Some(&released),
                Some(mark),
                started,
                Instant::now(),
                None,
            );
            let selected = admission(&registry, &worker)
                .paired_native_candidate
                .unwrap();
            assert!(selected.selected, "{}", selected.reason);
            assert_eq!(selected.native_snapshot_seq, Some(2));
            assert_eq!(selected.projection.unwrap().predecessors, 0);
        }
    }
    #[test]
    fn original_pair_expiry_is_not_renewed_by_new_raw_zero_polls() {
        let (registry, worker) = ready(RuntimeType::Sglang);
        let before = admission(&registry, &worker);
        let expiry = before
            .paired_native_candidate
            .as_ref()
            .unwrap()
            .original_native_expiry;
        publish(
            &registry,
            &worker,
            Some(&raw(10, true, RuntimeType::Sglang)),
        );
        let held = admission(&registry, &worker);
        assert_eq!(
            held.paired_native_candidate
                .as_ref()
                .unwrap()
                .original_native_expiry,
            expiry
        );
        let entries = registry.estimated_wait().entries.lock();
        let entry = &entries[worker.url()];
        let late = select(
            entry,
            registry.estimated_wait().config(),
            Instant::now() + Duration::from_secs(31),
            &held,
            true,
        );
        assert!(!late.selected);
        assert_eq!(late.reason, "coherent_pair_stale_or_identity_changed");
    }
    #[test]
    fn calibration_baseline_and_generation_exhaustion_fail_closed() {
        let (registry, worker) = ready(RuntimeType::Sglang);
        let original = admission(&registry, &worker);
        {
            let mut entries = registry.estimated_wait().entries.lock();
            let entry = entries.get_mut(worker.url()).unwrap();
            entry.calibration_baseline.0 = Some(0.3);
            assert!(
                !select(
                    entry,
                    registry.estimated_wait().config(),
                    Instant::now(),
                    &original,
                    true
                )
                .selected
            );
            entry.coherent_pair.hard_generation = u64::MAX;
        }
        publish(
            &registry,
            &worker,
            Some(&raw(10, false, RuntimeType::Sglang)),
        );
        let entries = registry.estimated_wait().entries.lock();
        let state = &entries[worker.url()].coherent_pair;
        assert!(state.issued.is_none() && state.published.is_none());
        assert_eq!(state.reason, "coherent_generation_exhausted");
    }
    #[test]
    fn cold_missing_and_over_age_provenance_never_select() {
        for runtime in [RuntimeType::Sglang, RuntimeType::Vllm] {
            let (registry, worker) = ready(runtime);
            let original = admission(&registry, &worker);
            let mut entries = registry.estimated_wait().entries.lock();
            let entry = entries.get_mut(worker.url()).unwrap();
            entry.coherent_pair = State::default();
            let missing = select(
                entry,
                registry.estimated_wait().config(),
                Instant::now(),
                &original,
                runtime == RuntimeType::Sglang,
            );
            assert!(!missing.selected);
            assert_eq!(missing.reason, "coherent_pair_cold_missing");
            let mut long = raw(10, false, runtime);
            let p = long.loads[0].prefill_metric_provenance.as_mut().unwrap();
            p.interval_seconds = Some(31.0);
            p.interval_started_time_seconds = Some(0.0);
            p.interval_ended_time_seconds = Some(31.0);
            p.previous_raw_counter = Some(0.0);
            p.raw_counter = Some(62.0);
            assert!(!entry.survival.native.candidate_raw_eligible(
                Some(&long),
                diagnostic_time(entry.sample.as_ref().unwrap().started),
                30.0
            ));
        }
    }
    #[test]
    fn calibration_reset_fences_an_already_issued_callback() {
        for runtime in [RuntimeType::Sglang, RuntimeType::Vllm] {
            let (registry, worker) = ready(runtime);
            let (old, started) = publish(&registry, &worker, Some(&raw(10, true, runtime)));
            registry
                .estimated_wait()
                .entries
                .lock()
                .get_mut(worker.url())
                .unwrap()
                .reset_calibration(Instant::now(), "coherent_test_reset");
            publish(&registry, &worker, Some(&raw(11, false, runtime)));
            registry.estimated_wait().publish_native_deadline(
                &worker,
                Some(&native(2, runtime, 50.0)),
                Some(old),
                started,
                Instant::now(),
                None,
            );
            assert!(registry.estimated_wait().entries.lock()[worker.url()]
                .coherent_pair
                .published
                .is_none());
        }
    }
    #[test]
    fn pending_generation_change_clears_every_size_selection_flag() {
        let (registry, worker) = ready(RuntimeType::Vllm);
        let entries = registry.estimated_wait().entries.lock();
        let mut pending = entries[worker.url()]
            .coherent_pair
            .published
            .as_ref()
            .unwrap()
            .pending
            .clone();
        pending.queue.applied = true;
        pending.selected_pending_applied = true;
        let next = pending.queue.summary.generation.checked_add(1).unwrap();
        guard_pending_generation(&mut pending, next);
        assert!(!pending.queue.applied && !pending.selected_pending_applied);
        assert_eq!(
            pending.queue.candidate_tokens,
            pending.queue.original_tokens
        );
        assert_eq!(pending.queue.reason, "inventory_generation_changed");
        assert_eq!(pending.source_gate_reason, "pending_queue_fallback");
    }
    #[test]
    fn changed_duplicate_original_facts_revoke_the_candidate_pair() {
        let runtime = RuntimeType::Sglang;
        let (registry, worker) = ready(runtime);
        let (mark, started) = publish(&registry, &worker, Some(&raw(10, true, runtime)));
        let mut changed = native(1, runtime, 100.0);
        changed.loads[0]
            .deadline_recovery
            .as_mut()
            .unwrap()
            .active_requests[0]
            .original_prefill_observation
            .as_mut()
            .unwrap()
            .original_prompt_tokens = Some(1001);
        registry.estimated_wait().publish_native_deadline(
            &worker,
            Some(&changed),
            Some(mark),
            started,
            Instant::now(),
            None,
        );
        let entries = registry.estimated_wait().entries.lock();
        assert!(entries[worker.url()].coherent_pair.published.is_none());
        assert_eq!(
            entries[worker.url()].coherent_pair.reason,
            "coherent_native_duplicate_fact_change"
        );
    }
    #[test]
    fn unchanged_duplicate_pairs_new_raw_without_renewing_native_origin() {
        for runtime in [RuntimeType::Sglang, RuntimeType::Vllm] {
            let (registry, worker) = ready(runtime);
            let before = admission(&registry, &worker)
                .paired_native_candidate
                .unwrap();
            let (mark, started) = publish(&registry, &worker, Some(&raw(10, false, runtime)));
            registry.estimated_wait().publish_native_deadline(
                &worker,
                Some(&native(1, runtime, 100.0)),
                Some(mark),
                started,
                Instant::now(),
                None,
            );
            let after = admission(&registry, &worker)
                .paired_native_candidate
                .unwrap();
            assert!(after.selected, "{}", after.reason);
            assert!(after.raw_poll_started.unwrap() > before.raw_poll_started.unwrap());
            assert_eq!(
                after.original_native_receipt,
                before.original_native_receipt
            );
            assert_eq!(after.original_native_expiry, before.original_native_expiry);
            assert!(after.original_native_receipt.unwrap() < after.raw_poll_started.unwrap());
            assert!(
                after.pair_available_at_gateway_seconds.unwrap()
                    > before.pair_available_at_gateway_seconds.unwrap()
            );
        }
    }
}

pub(super) fn issued(entry: &Entry, started: Instant) -> Option<RawContext> {
    entry
        .coherent_pair
        .issued
        .as_ref()
        .filter(|c| c.sample.started == started)
        .cloned()
}

fn guard_pending_generation(pending: &mut pending_work::Admission, generation: u64) {
    if pending.queue.summary.generation != generation {
        pending.queue.applied = false;
        pending.queue.candidate_tokens = pending.queue.original_tokens;
        pending.queue.reason = "inventory_generation_changed";
        pending.selected_pending_applied = false;
        pending.source_gate_reason = "pending_queue_fallback";
    }
}

/// The current native inventory/ACK/rebound observer runs first. Only raw-side
/// state is exchanged to validate this task's own proof, under the same lock.
pub(super) fn native_publication(
    entry: &mut Entry,
    context: Option<RawContext>,
    received: f64,
    timeout: bool,
    sg: bool,
    duplicate_facts_changed: bool,
) {
    entry.coherent_pair.published = None;
    if timeout
        || duplicate_facts_changed
        || entry.survival.native.candidate_native_stamp().is_none()
    {
        let reason = if timeout {
            "coherent_native_timeout"
        } else if duplicate_facts_changed {
            "coherent_native_duplicate_fact_change"
        } else {
            entry.survival.native.candidate_native_reason()
        };
        entry.coherent_pair.revoke(reason);
        return;
    }
    let Some(mut c) = context else {
        entry.coherent_pair.reason = "coherent_callback_context_missing";
        return;
    };
    if entry.coherent_pair.hard_generation == u64::MAX
        || c.hard_generation != entry.coherent_pair.hard_generation
        || c.calibration_epoch != entry.calibration_epoch
        || c.snapshot.baseline != entry.calibration_baseline.0.unwrap_or(f64::NAN)
        || received > c.raw_expiry
        || received < diagnostic_time(c.sample.started)
    {
        entry.coherent_pair.reason = "coherent_callback_hard_fence_or_expired";
        return;
    }
    let stamp = entry.survival.native.candidate_native_stamp().unwrap();
    if stamp.1 != diagnostic_time(c.sample.started) {
        entry
            .coherent_pair
            .revoke("coherent_callback_native_raw_mismatch");
        return;
    }
    let latest = entry
        .survival
        .native
        .replace_candidate_raw_state(c.native_raw);
    if sg {
        entry.survival.native.commit_sg_raw(
            c.sg_proof.clone(),
            stamp.1,
            c.calibration_epoch,
            received,
        );
    }
    c.native_raw = entry.survival.native.replace_candidate_raw_state(latest);
    c.native_stamp = Some(stamp);
    c.native_waiting = entry.survival.native.candidate_native_waiting();
    c.native_population = entry.survival.native.candidate_native_population();
    c.pair_available = Some(received);
    entry.coherent_pair.last_stamp = Some((stamp.1, c.raw_expiry, stamp));
    entry.coherent_pair.published = Some(c);
    entry.coherent_pair.reason = "coherent_pair_ready";
}

pub(super) fn select(
    entry: &Entry,
    config: &EstimatedWaitConfig,
    now: Instant,
    original: &survival::Admission,
    sg: bool,
) -> Diagnostic {
    let at = diagnostic_time(now);
    let mut d = diagnostic(entry, at);
    let Some(c) = entry.coherent_pair.published.as_ref() else {
        return d;
    };
    let Some(stamp) = entry.survival.native.candidate_native_stamp() else {
        d.reason = "coherent_live_native_unavailable";
        return d;
    };
    if entry.coherent_pair.hard_generation == u64::MAX
        || c.native_stamp != Some(stamp)
        || c.hard_generation != entry.coherent_pair.hard_generation
        || c.calibration_epoch != entry.calibration_epoch
        || c.snapshot.baseline != entry.calibration_baseline.0.unwrap_or(f64::NAN)
        || at > c.raw_expiry
        || at > stamp.3
        || at < stamp.2
        || c.pair_available.is_none_or(|received| at < received)
    {
        d.reason = "coherent_pair_stale_or_identity_changed";
        return d;
    }
    d.baseline_seconds = Some(c.snapshot.baseline);
    // With no publication mismatch this is exactly the original selected913
    // output, including its guards and fallback decisions.
    if entry
        .sample
        .as_ref()
        .is_some_and(|s| s.started == c.sample.started)
    {
        let selected = if sg {
            original
                .sg_raw_basis_deadline
                .as_ref()
                .map(|s| (&s.projection, Some(s.clone())))
        } else {
            original
                .projection
                .supported
                .then_some((&original.projection, None))
        };
        if let Some((projection, sg_projection)) = selected {
            d.selected = true;
            d.reason = "coherent_current_pair_original_output";
            d.total_ttft_seconds = projection.total_ttft_seconds;
            d.projection = Some(projection.clone());
            d.sg_raw_basis_deadline = sg_projection;
            d.training_snapshot = Some(original.training_snapshot);
            d.normalization = original.normalization;
            d.original_projection = Some(original.projection.clone());
        } else {
            d.reason = original
                .sg_raw_basis_rejection_reason
                .unwrap_or(original.projection.reason);
        }
        return d;
    }
    let mut snapshot = c.snapshot;
    snapshot.at = at;
    snapshot.fresh = at <= c.raw_expiry;
    let dispatch = entry.total_dispatched.saturating_sub(c.sample.watermark);
    let queue = c.sample.prepared.queue_excess_seconds(dispatch);
    let growth = (queue - snapshot.original_queue_extra).max(0.0);
    snapshot.x += growth;
    snapshot.original_queue_extra = queue;
    snapshot.live_queue_extra = snapshot.live_queue_extra.map(|v| v + growth);
    snapshot.context = entry.decision.context(
        at,
        c.sample.prepared.timely_progress,
        c.sample.prepared.progress_capacity,
        c.sample.prepared.has_backlog,
    );
    let original_query = decision::Admission {
        snapshot,
        predictions: entry.decision.predictions(
            snapshot,
            at,
            snapshot.fresh && snapshot.native_ready,
        ),
    };
    let mut pending = c.pending.clone();
    guard_pending_generation(&mut pending, entry.pending_work.generation());
    pending.at = at;
    pending.debt_seconds = (queue - pending.queue.original_tokens / pending.capacity).max(0.0);
    let (control_x, fraction) = pending_work::candidate_x(
        pending.queue,
        pending.capacity,
        pending.debt_seconds,
        snapshot.x,
        pending.recovery_cap_seconds,
    );
    pending.control_x = control_x;
    pending.recovery_fraction = fraction;
    pending.original_x = snapshot.x;
    pending.candidate_x = if pending.selected_pending_applied {
        control_x
    } else {
        snapshot.x
    };
    let control_snapshot = decision::Snapshot {
        x: pending.candidate_x,
        ..snapshot
    };
    pending.predictions = entry.decision.predictions(control_snapshot, at, true);
    let control = decision::Admission {
        snapshot: control_snapshot,
        predictions: pending.predictions.clone(),
    };
    let basis =
        entry
            .basis_decision
            .admission(&original_query, &control, pending_work::METHOD, at, true);
    let native = entry
        .survival
        .native
        .candidate_with_live_ledger(c.native_raw.clone());
    let Some(mut candidate) = entry.survival.paired_admission(
        &basis,
        &pending,
        Some(snapshot.baseline + snapshot.x),
        &native,
    ) else {
        d.reason = "coherent_current_fit_or_reference_unavailable";
        return d;
    };
    let work = survival::FrozenRawWork {
        poll_started: diagnostic_time(c.sample.started),
        source: c.sample.prepared.size_basis.map(|p| p.progress_source),
        key: c.sample.prepared.started_capacity_source_key,
        token_work: c.sample.prepared.estimate.token_work(dispatch),
        original_capacity: c.sample.prepared.effective_prefill_capacity,
        observed_delay_floor: c.sample.prepared.observed_delay_floor,
        measured_delay_only: c.sample.prepared.measured_delay_only(),
    };
    candidate.select_raw_fallback(Some(survival::RawFallback {
        baseline_seconds: snapshot.baseline,
        baseline_source: entry.calibration_baseline.1.as_str(),
        raw_extra_seconds: c.sample.prepared.excess_seconds(dispatch),
        total_ttft_seconds: snapshot.baseline + c.sample.prepared.excess_seconds(dispatch),
        capacity_basis: None,
    }));
    native.rebase_selected_raw_fallback(
        &mut candidate,
        work,
        config.estimated_wait_fallback_prefill_throughput,
    );
    if sg {
        native.select_sg_raw_basis(&mut candidate, entry.calibration_epoch);
    }
    let selected = if sg {
        candidate
            .sg_raw_basis_deadline
            .as_ref()
            .map(|s| (&s.projection, Some(s.clone())))
    } else {
        candidate
            .projection
            .supported
            .then_some((&candidate.projection, None))
    };
    if let Some((projection, sg_projection)) = selected {
        d.selected = true;
        d.reason = "coherent_held_pair_live_ledger_projection";
        d.total_ttft_seconds = projection.total_ttft_seconds;
        d.projection = Some(projection.clone());
        d.sg_raw_basis_deadline = sg_projection;
        d.capacity_basis = Some(work);
        d.training_snapshot = Some(candidate.training_snapshot);
        d.normalization = candidate.normalization;
        d.original_projection = Some(candidate.projection.clone());
    } else {
        d.reason = candidate
            .sg_raw_basis_rejection_reason
            .unwrap_or(candidate.projection.reason);
    }
    d
}
